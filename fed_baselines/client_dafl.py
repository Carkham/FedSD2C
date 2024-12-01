import collections
import math
import random
from copy import deepcopy
from tqdm import tqdm
import torch
import os
import wandb
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from fed_baselines.client_base import FedClient

from utils.models import RandomMixup, RandomCutmix
from torch.utils.data import DataLoader, default_collate
from fedsd2c.util import *
from utils import AverageMeter


class DAFLClient(FedClient):

    def __init__(self, args, client_id, epoch, dataset_id, model_name):
        super().__init__(args, client_id, epoch, dataset_id, model_name)
        """
        Client in the federated learning for FedD3
        :param client_id: Id of the client
        :param dataset_id: Dataset name for the application scenario
        """
        self.args = args
        self.local_step = 0

    def train(self, model: nn.Module):
        """
        Client trains the model on local dataset
        :param model: model waited to be trained
        :return: Local updated model
        """
        model.train()
        model.to(self._device)
        mixup_transforms = []
        collate_fn = None
        if self.mixup_alpha > 0.0:
            mixup_transforms.append(RandomMixup(self._num_class, p=1.0, alpha=self.mixup_alpha))
        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(RandomCutmix(self._num_class, p=1.0, alpha=self.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):
                return mixupcutmix(*default_collate(batch))
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)

        optimizer = torch.optim.SGD(model.parameters(), lr=self._lr, momentum=self._momentum, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)
        loss_func = nn.CrossEntropyLoss()

        # Training process
        loss_accumulator = AverageMeter()
        pbar = tqdm(range(self._epoch))
        for epoch in pbar:
            epoch_loss = AverageMeter()
            lr_scheduler(optimizer, epoch, epoch)
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    output = model(b_x)
                    loss = loss_func(output, b_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_accumulator.update(loss.data.cpu().item())
                epoch_loss.update(loss.data.cpu().item())
                if self.args.using_wandb:
                    wandb.log({
                        f"{self.name}C local_loss": loss.item(),
                        "iteration": self.local_step,
                    })
                    self.local_step += 1
            pbar.set_description('Epoch: %d' % epoch +
                                 '| Train loss: %.4f ' % epoch_loss.avg +
                                 '| lr: %.4f ' % optimizer.state_dict()['param_groups'][0]['lr'])

        return loss_accumulator.avg
