import os
import json
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from json import JSONEncoder
from tqdm import tqdm

from fedsd2c.util import means, stds, lr_cosine_policy
from postprocessing.recorder import Recorder
from utils import Logger, AverageMeter
from utils.fed_utils import assign_dataset
from utils.models import *
from copy import deepcopy

from fed_baselines.server_base import FedServer


class CentralServer(FedServer):
    def __init__(self, args, client_list, dataset_id, model_name):
        """
        Server in the federated learning for Central Learning
        """
        super().__init__(args, client_list, dataset_id, model_name)
        # Server Properties
        self._id = "server"
        self._dataset_id = dataset_id
        self._model_name = model_name
        self._i_seed = args.sys_i_seed

        # Training related parameters
        self._epoch = args.server_n_epoch
        self._batch_size = args.server_bs
        self._lr = args.server_lr
        self._momentum = args.server_momentum
        self._num_workers = args.server_n_worker
        self.optim_name = args.server_optimizer

        # Global test dataset
        self._test_data = None

        # Global distilled dataset
        self.trainset = None

        # Following private parameters are defined by dataset.
        # self.model = None
        _, self._image_length, self._image_channel = assign_dataset(dataset_id)
        self._image_width = self._image_length


        self.recorder = Recorder()

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)

    def train(self):
        """
        Client trains the model on local dataset
        :return: Local updated model, number of local data points, training loss
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True, drop_last=True)

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum,
                                    weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)
        loss_func = nn.CrossEntropyLoss()

        # Training process
        loss_accumulator = AverageMeter()
        pbar = tqdm(range(self._epoch))
        accs = []
        for epoch in pbar:
            epoch_loss = AverageMeter()
            lr_scheduler(optimizer, epoch, epoch)
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_accumulator.update(loss.data.cpu().item())
                epoch_loss.update(loss.data.cpu().item())
            pbar.set_description('Epoch: %d' % epoch +
                                 '| Train loss: %.4f ' % epoch_loss.avg +
                                 '| lr: %.4f ' % optimizer.state_dict()['param_groups'][0]['lr'])
            # Test process
            acc = self.test()
            accs.append(acc)
            L = Logger()
            logger = L.get_logger()
            logger.info('Epoch: %d' % epoch + ' / %d ' % self._epoch +
                        '| Train loss: %.4f ' % loss.data.cpu().numpy() +
                        '| Accuracy: %.4f ' % acc +
                        '| Max Acc: %.4f ' % np.max(accs))

    def load_cls_record(self, cls_record):
        """
        Client loads the statistic of local label.
        :param cls_record: class number record
        """
        pass
