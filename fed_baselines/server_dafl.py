import os.path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from fed_baselines.server_base import FedServer

from fedsd2c.util import WrapperDataset, lr_cosine_policy, BNFeatureHook, means, stds
from utils import Logger, AverageMeter
from utils.models_gan import Generator_ACGan, Generator, LargeGenerator
from utils.util import Ensemble_A, MultiTransform, kldiv, ImagePool

augment = transforms.Compose([
    transforms.RandomHorizontalFlip()
])


def sample_zy(batch_size, num_class, z_dim, device):
    z = torch.tensor(np.random.normal(0, 1, (batch_size, z_dim)), dtype=torch.float32).to(device)
    y = torch.tensor(np.random.choice(num_class, batch_size), dtype=torch.long).to(device)
    # z = Variable(np.random.normal(0, 1, (batch_size, z_dim))).to(self._device)
    # gen_labels = Variable(torch.LongTensor(np.random.choice(num_class, batch_size))).to(self._device)
    return z, y


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class DAFLServer(FedServer):
    def __init__(self, args, client_list, dataset_id, model_name):
        super().__init__(args, client_list, dataset_id, model_name)
        self.generator = LargeGenerator(
            nz=args.dfkd_z_dim,
            ngf=64,
            img_size=args.dfkd_img_size,
            nc=3
        )
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
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

        # DAFL Parameters
        self.temperature = args.dfkd_temp
        self.r_bn = args.dfkd_r_bn
        self.r_adv = args.dfkd_r_adv
        self.r_balance = args.dfkd_r_bal
        self.gen_iteration = args.dfkd_giter
        self.model_iteration = args.dfkd_miter
        self.epoch_iteration = args.dfkd_eiter
        self.z_dim = args.dfkd_z_dim

        # Clients' model and dataset
        self.client_dict = {}
        self.client_model = {}
        self.client_data = {}
        self.cls_2_client = defaultdict(list)
        self.client_cls_score = {}

        self.normalizer = transforms.Compose([
            # augmentation.RandomCrop(size=(self._image_dim, self._image_dim), padding=4),
            # augmentation.RandomHorizontalFlip(),
            transforms.Normalize(mean=means[self._dataset_id], std=stds[self._dataset_id])
        ])

        # =======================
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=means[self._dataset_id], std=stds[self._dataset_id])
            ])

        self.data_pool = ImagePool(root=os.path.join(self._args.sys_res_root, args.save_name, "syn_data"))

        # Teacher model
        self.teacher = None
        self.loss_r_feature_layers = []

    def train_generator(self):
        """
        Generate a novel dataset from user decoders

        """
        if self.teacher is None:
            self.teacher = Ensemble_A(list(self.client_model.values()))
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.loss_r_feature_layers = []
            for module in self.teacher.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.loss_r_feature_layers.append(BNFeatureHook(module))
            self.teacher.to(self._device)

        self.teacher.eval()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.generator.train()
        for p in self.generator.parameters():
            p.requires_grad = True

        criterion_ce = nn.CrossEntropyLoss().to(self._device)

        l_ce, l_adv, l_bn, l_bal, l = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        pbar = tqdm(range(self.gen_iteration))
        for step in pbar:
            self.optimizer.zero_grad()
            z = torch.randn(size=(self._batch_size, self.z_dim), device=self._device)  #
            inputs = self.generator(z)
            inputs = self.normalizer(inputs)  # normalize
            t_out = self.teacher(inputs)

            loss_r_bn = sum([mod.r_feature for mod in self.loss_r_feature_layers]) / len(self.client_model)
            loss_ce = criterion_ce(t_out, t_out.max(1)[1])
            if self.r_adv > 0:
                s_out = self.model(inputs)
                loss_adv = -kldiv(s_out, t_out)  # decision adversarial distillation
            else:
                loss_adv = loss_ce.new_zeros(1)
            p = F.softmax(t_out, dim=1).mean(0)
            loss_balance = (p * torch.log(p)).sum()  # maximization

            loss = loss_ce + self.r_bn * loss_r_bn + self.r_adv * loss_adv + self.r_balance * loss_balance
            loss.backward()
            self.optimizer.step()

            l_adv.update(loss_adv.item())
            l_ce.update(loss_ce.item())
            l_bn.update(loss_r_bn.item())
            l_bal.update(loss_balance.item())
            l.update(loss.item())

            pbar.set_description(f'(Training Generator) Step: {step}' +
                                 f'| Train loss: {l.avg:.4f} ' +
                                 f'| ADV loss: {self.r_adv:.3f}*{l_adv.avg:.4f} ' +
                                 f'| BN loss: {self.r_bn:.3f}*{l_bn.avg:4f} ' +
                                 f'| Balance loss: {self.r_balance:.3f}*{l_bal.avg:.4f} ' +
                                 f'| CE loss: {1.00}*{l_ce.avg:.4f} ' +
                                 '| lr: %.4f ' % self.optimizer.state_dict()['param_groups'][0]['lr'])
        return {
            "loss": l.avg,
            "ADV loss": l_adv.avg,
            "BN loss": l_bn.avg,
            "CE loss": l_ce.avg,
            "Balance loss": l_bal.avg
        }

    def train_model(self, epoch, optimizer):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

        loss_func = nn.KLDivLoss("batchmean").to(self._device)
        lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)
        lr_scheduler(optimizer, epoch, epoch)

        loss_accumulator = AverageMeter()
        acc_accumulator = 0
        pbar = tqdm(enumerate(range(self.model_iteration)))
        for step in pbar:
            images = self.sample()
            with torch.no_grad():
                images = images.to(self._device)
                t_out = self.teacher(images)
                t_out = F.softmax(t_out / self.temperature, dim=1).detach()

            with torch.enable_grad():
                s_out = self.model(images)
                s_out = F.log_softmax(s_out / self.temperature, dim=1)

                loss = loss_func(s_out, t_out)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = s_out.argmax(dim=1)
                target = t_out.argmax(dim=1)
                acc_accumulator += pred.eq(target.view_as(pred)).sum().item()
                acc = acc_accumulator / (self.model_iteration * self._batch_size) * 100

                loss_accumulator.update(loss.item())

            pbar.set_description(
                '(Training Server model) Epoch/Step: {}/{}'.format(epoch, step) +
                '| Train loss: %.4f ' % loss_accumulator.avg +
                '| ST acc: %.4f ' % acc +
                '| lr: %.4f ' % optimizer.state_dict()['param_groups'][0]['lr']
            )

        return loss_accumulator.avg

    @torch.no_grad()
    def sample(self):
        self.generator.eval()
        z = torch.randn(size=(self._batch_size, self.z_dim), device=self._device)
        inputs = self.normalizer(self.generator(z))
        return inputs

    def train(self):
        accs = []
        self.model.to(self._device)
        self.generator.to(self._device)
        if self.optim_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum,
                                        weight_decay=1e-4)
        elif self.optim_name == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        elif self.optim_name == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        else:
            raise ValueError("Optimizer error")

        for epoch in range(self._epoch):
            for _ in range(self.epoch_iteration // self.model_iteration):
                gl_dict = self.train_generator()
                model_l = self.train_model(epoch, optimizer)

            # Recording the train loss during the training
            acc = self.test()
            accs.append(acc)
            L = Logger()
            logger = L.get_logger()
            logger.info('Epoch: %d' % epoch + ' / %d ' % self._epoch +
                        '| Model loss: %.4f ' % model_l +
                        " ".join([f"| Gen {k}: {v:.4f}" for k, v in gl_dict.items()]) +
                        '| Accuracy: %.4f ' % acc +
                        '| Max Acc: %.4f ' % np.max(np.array(accs)))

        torch.save(self.model.state_dict(), os.path.join(self._args.sys_res_root, self._args.save_name, "model.pt"))

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
    
    def load_testset(self, testset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        super().load_testset(testset)
        self.testset.transform = transforms.Compose([
            # transforms.Resize(32),
            self.testset.transform
        ])
