from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from fed_baselines.server_base import FedServer
import copy

from fedsd2c.util import WrapperDataset, lr_cosine_policy
from utils import Logger

augment = transforms.Compose([
    transforms.RandomHorizontalFlip()
])


class FedCVAEServer(FedServer):
    def __init__(self, args, client_list, dataset_id, model_name):
        super().__init__(args, client_list, dataset_id, model_name)
        self.z_dim = args.cvae_z_dim
        self.beta = args.cvae_beta
        self.should_weight = True
        self.num_users = len(self.client_list)
        self.uniform_range = (-1, 1)

        self._cls_record = None
        self.pmf_dict = {}
        self.data_amt_dict = {}

        # Training Parameters
        self.num_train_samples = 10000
        self._epoch = args.server_n_epoch
        self._batch_size = args.server_bs
        self._lr = args.server_lr
        self._momentum = args.server_momentum
        self._num_workers = args.server_n_worker
        self.optim_name = args.server_optimizer

        self.client_dict = {}

    def generate_dataset_from_user_decoders(self, users, num_train_samples):
        """
        Generate a novel dataset from user decoders

        :param users: List of selected users to use as teacher models
        :param num_train_samples: How many samples to add to our new dataset
        """

        X_vals = torch.Tensor()
        y_vals = torch.Tensor()
        z_vals = torch.Tensor()

        total_train_samples = 0
        count_user = 0
        for u in users:
            u.model.to(self._device)
            u.model.eval()

            # Sample a proportional number of samples to the amount of data the current user has seen
            if self.should_weight:
                user_num_train_samples = int(self.data_amt_dict[u.name] * num_train_samples)
            else:
                user_num_train_samples = int(num_train_samples / self.num_users)

            if count_user == self.num_users - 1:
                user_num_train_samples = num_train_samples - total_train_samples
            else:
                total_train_samples += user_num_train_samples
                count_user += 1

            z = u.model.sample_z(
                user_num_train_samples, "truncnorm", width=self.uniform_range
            ).to(self._device)

            # Sample y's according to each user's target distribution
            classes = np.arange(self._num_class)
            y = torch.from_numpy(
                np.random.choice(classes, size=user_num_train_samples, p=self.pmf_dict[u.name])
            ).long()
            y_hot = torch.nn.functional.one_hot(y, self._num_class).to(self._device)

            # Detaching ensures that there aren't issues w/trying to calculate the KD grad WRT this net's params - not needed!
            X = u.model.decoder(z, y_hot).detach()

            X, y_hot, z = X.cpu(), y_hot.cpu(), z.cpu()

            X_vals = torch.cat((X_vals, X), 0)
            y_vals = torch.cat((y_vals, y_hot), 0)
            z_vals = torch.cat((z_vals, z), 0)

        # Normalize "target" images to ensure reconstruction loss works correctly
        X_vals = torch.sigmoid(X_vals)

        decoder_dataset = WrapperDataset(X_vals, y_vals, z_vals, augment)

        return decoder_dataset

    def train(self):
        dset = self.generate_dataset_from_user_decoders(self.client_dict.values(), self.num_train_samples)
        train_loader = DataLoader(dset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        if self.optim_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        elif self.optim_name == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        elif self.optim_name == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        else:
            raise ValueError("Optimizer error")

        loss_func = nn.CrossEntropyLoss()
        lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)
        accs = []
        for epoch in range(self._epoch):
            loss_accumulator = 0
            lr_scheduler(optimizer, epoch, epoch)
            for step, (x, y, _) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_accumulator += loss.item()

            loss_accumulator /= len(train_loader)

            # Recording the train loss during the training
            acc = self.test()
            accs.append(acc)
            L = Logger()
            logger = L.get_logger()
            logger.info('Epoch: %d' % epoch + ' / %d ' % self._epoch +
                        '| Train loss: %.4f ' % loss.data.cpu().numpy() +
                        '| Accuracy: %.4f ' % acc +
                        '| Max Acc: %.4f ' % np.max(np.array(accs)))

    def load_cls_record(self, cls_record):
        """
        Client loads the statistic of local label.
        :param cls_record: class number record
        """
        self._cls_record = {}
        self._cls_record = {int(k): v for k, v in cls_record.items()}
        self.pmf_dict = defaultdict(lambda: np.zeros(self._num_class))
        self.data_amt_dict = {}
        for client_id, record in self._cls_record.items():
            for p in range(self._num_class):
                self.pmf_dict[client_id][p] = record.get(float(p), 0) / sum(record.values())

            self.data_amt_dict[client_id] = sum(record.values()) / sum(
                [sum(_r.values()) for _r in self._cls_record.values()])
