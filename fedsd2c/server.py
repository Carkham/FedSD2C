import collections
import os
import random
from copy import deepcopy

import numpy as np
import wandb
from torch import optim

from fedsd2c.util import lr_cosine_policy, mix_aug, lr_no_policy, BNFeatureHook, clip_tiny, denormalize_tiny, \
    DistillCIDDataset, clip, denormalize, DistilledDataset, means, stds
from postprocessing.recorder import Recorder
from utils.models import *
import torch
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model
from fed_baselines.server_base import FedServer
from utils import Logger
from collections import defaultdict

augment = transforms.Compose([
    transforms.RandomHorizontalFlip()
])


class FedSD2CServer(FedServer):
    def __init__(self, args, client_list, epoch, batch_size, lr, momentum=0.9, num_workers=2, dataset_id='mnist',
                 server_id='server', model_name="LeNet", i_seed=0):
        """
        Server in the federated learning for FedSD2C
        :param epoch: Number of total training epochs in the server
        :param batch_size: Batch size for the training in the server
        :param lr: Learning rate for the training in the server
        :param momentum: Learning momentum for the training in the server
        :param num_workers: Number of the workers for the training in the server
        :param dataset_id: Dataset name for the application scenario
        :param server_id: Id of the server
        :param model_name: Machine learning model name for the application scenario
        :param i_seed: Index of the seed used in the experiment
        :param test_on_gpu: Flag: 1: Run testing on GPU after every epoch, otherwise 0.
        """
        super().__init__(args, client_list, dataset_id, model_name)
        # Server Properties
        self._id = server_id

        # Training related parameters
        self._epoch = epoch
        self._batch_size = batch_size
        self._lr = lr
        self._momentum = momentum
        self._num_workers = num_workers
        self.optim_name = args.server_optimizer

        # Global distilled dataset
        self._distill_data = None
        # Recording results
        self.recorder = Recorder()
        # Run on the GPU
        gpu = args.gpu_id
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # Clients' model and dataset
        self.client_model = {}
        self.client_data = {}
        self.cls_2_client = defaultdict(list)
        self.client_2_cls = {}
        self.client_cls_score = {}

        # wandb
        self.train_step = 0

        # FastDD parameters
        self.input_size = 64
        self.num_crop = self._args.fedsd2c_num_crop
        self.factor = 1
        self.mipc = self._args.fedsd2c_mipc
        self.ipc = self._args.fedsd2c_ipc
        self.temperature = 1

        self.iterations_per_layer = self._args.fedsd2c_iteration
        self.jitter = self._args.fedsd2c_jitter
        self.sre2l_lr = self._args.fedsd2c_lr
        self.l2_scale = self._args.fedsd2c_l2_scale
        self.tv_l2 = self._args.fedsd2c_tv_l2
        self.r_bn = self._args.fedsd2c_r_bn
        self.r_c = self._args.fedsd2c_r_c
        self.first_bn_multiplier = 10.
        self.beta = 0

        self.syn_step = 0

    def load_distill(self, data):
        """
        Server loads the decentralized distilled dataset.
        :param data: Dataset for training.
        """
        self._distill_data = {}
        self._distill_data = deepcopy(data)

    def load_state_dict(self, state_dict):
        """
        Server model load state dict.
        :return: Global model dict
        """
        self.model.load_state_dict(state_dict)

    def rec_distill(self, name, model, data, cls):
        self.client_model[name] = model
        self.client_data[name] = data
        for c in cls:
            self.cls_2_client[c].append(name)
        self.client_2_cls[name] = cls

    def train_distill(self, shuffle=True):
        central_x = []
        central_y = []
        central_ids = [] 
        for cid, dset in self.client_data.items():
            central_x.extend(dset.x)
            central_y.extend(dset.y)
            central_ids.extend([cid] * len(dset))
        client_dataloaders = {
            "central": DataLoader(DistillCIDDataset(central_x, central_y, central_ids, augment),
                                    batch_size=self._batch_size, shuffle=True)
        }

        self.model.to(self._device)
        if self.optim_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum,
                                        weight_decay=1e-4)
        elif self.optim_name == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        elif self.optim_name == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        else:
            raise ValueError("Optimizer error")
        if self._args.server_lr_scheduler == "cos":
            lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)
        else:
            lr_scheduler = lr_no_policy(self._lr)

        loss_func = nn.KLDivLoss(reduction="batchmean")
        for epoch in range(self._epoch):
            loss_accumulator = 0
            lr_scheduler(optimizer, epoch, epoch)

            client_data_iters = {}
            client_orders = []
            for k, v in client_dataloaders.items():
                client_data_iters[k] = iter(v)
                client_orders.extend(len(v) * [k])
            if shuffle:
                random.shuffle(client_orders)
            for step, cur_client_name in enumerate(client_orders):
                x, y, cids = next(client_data_iters[cur_client_name])
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    # b_x, rand_index, lam, _ = mix_aug(b_x, self._args, device=self._device)
                    b_y = None

                    assert cids is not None
                    weight_sum = 0
                    b_y = 0
                    for teacher_name in self.client_model:
                        teacher_model = self.client_model[teacher_name]
                        teacher_model.eval()
                        teacher_output = teacher_model(b_x)
                        # if lam is not None and rand_index is not None:
                        #     weight = torch.where(torch.eq(cids, teacher_name), 0, 1)  # (B)
                        #     weight = (lam * weight + (1 - lam) * weight[rand_index]).unsqueeze(-1).expand(
                        #         weight.shape[0], self._num_class).to(teacher_output)
                        # else:
                        weight = torch.where(torch.eq(cids, teacher_name), 1, 0).unsqueeze(-1).expand(
                            b_x.shape[0], self._num_class).to(teacher_output)
                        b_y = b_y + teacher_output * weight.to(teacher_output)  # (B, num_class)
                        weight_sum = weight_sum + weight
                    b_y = F.softmax(b_y / weight_sum / self.temperature, dim=1)

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    soft_output = F.log_softmax(output / self.temperature, dim=1)
                    loss = loss_func(soft_output, b_y) * (self.temperature ** 2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_accumulator += loss.item()

            loss_accumulator /= len(client_orders)
            # Recording the train loss during the training
            self.recorder.res['server']['train_loss'].append(loss.data.cpu().numpy())

            acc = self.test()
            self.recorder.res['server']['iid_accuracy'].append(acc)
            if self._args.using_wandb:
                wandb.log({
                    "Server Train loss": loss_accumulator,
                    "Server Accuracy": acc,
                    "iteration": self.train_step
                })
                self.train_step += 1
            L = Logger()
            logger = L.get_logger()
            logger.info('Epoch: %d' % epoch + ' / %d ' % self._epoch +
                        '| Train loss: %.4f ' % loss_accumulator +
                        '| Accuracy: %.4f ' % acc +
                        '| Max Acc: %.4f ' % np.max(np.array(self.recorder.res['server']['iid_accuracy'])))

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.client_model = {}
        self.client_data = {}
        self.cls_2_client = defaultdict(list)

    def get_images(self):
        images = []
        soft_labels = []
        print("generating IPC images (200)")
        for i in range(self.ipc):
            i, sf = self._sre2l(i)
            images.extend(i)
            soft_labels.extend(sf)
        return images, soft_labels

    def _sre2l(self, ipc_id):
        args = self._args
        model_teacher = deepcopy(self.model).to(self._device)
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False
        save_every = 100
        batch_size = 100

        loss_r_feature_layers = []
        for module in model_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(BNFeatureHook(module))

        # setup target labels
        # targets_all = torch.LongTensor(np.random.permutation(200))
        targets_all = torch.LongTensor(np.arange(200))

        saved_best_inputs = []
        saved_best_soft_labels = []
        for kk in range(0, 200, batch_size):
            targets = targets_all[kk:min(kk + batch_size, 200)].to(self._device)

            data_type = torch.float
            inputs = torch.randn((targets.shape[0], 3, 64, 64), requires_grad=True, device=self._device,
                                 dtype=data_type)

            iterations_per_layer = self.iterations_per_layer

            optimizer = optim.Adam([inputs], lr=self.sre2l_lr, betas=[0.5, 0.9], eps=1e-8)
            lr_scheduler = lr_cosine_policy(self.sre2l_lr, 0, iterations_per_layer)  # 0 - do not use warmup
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(self._device)

            best_cost = 1e4
            for iteration in range(iterations_per_layer):
                # learning rate scheduling
                lr_scheduler(optimizer, iteration, iteration)

                aug_function = transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                ])
                inputs_jit = aug_function(inputs)

                # apply random jitter offsets
                off1 = random.randint(-self.jitter, self.jitter)
                off2 = random.randint(-self.jitter, self.jitter)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # forward pass
                optimizer.zero_grad()
                outputs = model_teacher(inputs_jit)

                # R_cross classification loss
                loss_ce = criterion(outputs, targets)

                # R_feature loss
                rescale = [self.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                loss_r_bn_feature = sum(
                    [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                # final loss
                loss = loss_ce + self.r_bn * loss_r_bn_feature

                if iteration % save_every == 0:
                    L = Logger()
                    logger = L.get_logger()
                    logger.info(
                        "------------------------IPC {} iteration {}----------------------".format(ipc_id, iteration))
                    logger.info("Total loss: {} | loss_r_bn_feature: {} | Main criterion: {}".format(
                        loss.item(), loss_r_bn_feature.item(), criterion(outputs, targets).item()))
                    # comment below line can speed up the training (no validation process)

                if self._args.using_wandb:
                    wandb.log({
                        "Total loss": loss.item(),
                        "loss_r_bn_feature": loss_r_bn_feature.item(),
                        "Main criterion": criterion(outputs, targets).item(),
                        "syn_step": self.syn_step
                    })
                    self.syn_step += 1

                # do image update
                loss.backward()
                optimizer.step()

                # clip color outlayers
                inputs.data = clip_tiny(inputs.data)

                if best_cost > loss.item() or iteration == 0:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

            optimizer.state = collections.defaultdict(dict)
            saved_best_inputs.extend([d.squeeze() for d in best_inputs])
            saved_best_soft_labels.extend([d.squeeze() for d in model_teacher(best_inputs)])
            # if args.store_last_images:
            #     save_inputs = inputs.data.clone()  # using multicrop, save the last one
            # save_inputs = denormalize_tiny(save_inputs)
            # save_images(args, save_inputs, targets, ipc_id)

            return saved_best_inputs, saved_best_soft_labels

    def syn_data(self):
        logger = Logger()
        logger = logger.get_logger()

        # Initialize client models
        for teacher_name in self.client_model:
            model = self.client_model[teacher_name].to(self._device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.client_model[teacher_name] = model

        loss_dict = {
            "l": defaultdict(list),
            "ce": defaultdict(list),
            "bn": defaultdict(list),
            "ce_c": defaultdict(list)
        }
        loss_ce_list = loss_dict["ce"]
        loss_bn_list = loss_dict["bn"]
        loss_ce_c_list = loss_dict["ce_c"]
        loss_list = loss_dict["l"]
        # Synthesize data for each client
        for teacher_name, teacher_model in self.client_model.items():
            save_every = 500
            batch_size = 100
            loss_r_feature_layers = []
            for module in teacher_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    loss_r_feature_layers.append(BNFeatureHook(module))
            targets_all = torch.tensor(self.client_2_cls[teacher_name]).repeat(10)
            targets_all = targets_all[torch.randperm(len(targets_all))].to(self._device)

            saved_best_inputs = []
            saved_labels = []
            for kk in range(0, len(targets_all), batch_size):
                targets = targets_all[kk:min(kk + batch_size, len(targets_all))].to(self._device)
                inputs = torch.randn((targets.shape[0], 3, 64, 64), device=self._device, dtype=torch.float)
                for c in range(3):
                    m, s = means[self._dataset_id][c], stds[self._dataset_id][c]
                    inputs[:, c] = inputs[:, c] * s + m
                inputs.requires_grad = True

                optimizer = optim.Adam([inputs], lr=self.sre2l_lr, betas=[0.5, 0.9], eps=1e-8)
                lr_scheduler = lr_cosine_policy(self.sre2l_lr, 0, self.iterations_per_layer)
                criterion = nn.CrossEntropyLoss().to(self._device)
                criterion_c = nn.CrossEntropyLoss(reduction='none').to(self._device)

                best_inputs = None
                best_cost = 1e4
                logger.info(targets)
                for iteration in range(self.iterations_per_layer):
                    # learning rate scheduling
                    lr_scheduler(optimizer, iteration, iteration)

                    aug_function = transforms.Compose([
                        transforms.RandomResizedCrop(self.input_size),
                        transforms.RandomHorizontalFlip(),
                    ])
                    inputs_jit = aug_function(inputs)

                    # forward pass
                    optimizer.zero_grad()
                    outputs = teacher_model(inputs_jit)

                    # R_cross classification loss
                    loss_ce = criterion(outputs, targets)

                    loss_ce_c = 0
                    for c_name in self.client_model:
                        if c_name != teacher_name:
                            c_model = self.client_model[c_name]
                            c_outputs = c_model(inputs_jit)
                            c_loss = criterion_c(c_outputs, targets)
                            weight = torch.where(torch.isin(targets,
                                                            torch.tensor(self.client_2_cls[c_name], device=self._device,
                                                                         dtype=targets.dtype)), 1, 0)
                            loss_ce_c += (c_loss * weight).mean()

                    # R_feature loss
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                    loss_r_bn_feature = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                    # final loss
                    loss = loss_ce + self.r_bn * loss_r_bn_feature + self.r_c * loss_ce_c

                    if self._args.using_wandb:
                        loss_list[iteration].append(loss.item())
                        loss_bn_list[iteration].append(loss_r_bn_feature.item())
                        loss_ce_list[iteration].append(loss_ce.item())
                        loss_ce_c_list[iteration].append(loss_ce_c.item())

                    if iteration % save_every == 0 or iteration == save_every - 1:
                        logger.info(
                            "------------batch idx: {} / {} iteration {} / {}----------".format(kk, len(targets_all), iteration, self.iterations_per_layer))
                        logger.info("Total loss: {} | loss_r_bn_feature: {} | Main criterion: {} | Other criterion: {}".format(
                            loss.item(), loss_r_bn_feature.item(), loss_ce.item(), loss_ce_c.item()))

                    # do image update
                    loss.backward()
                    optimizer.step()

                    # clip color outlayers
                    inputs.data = clip(inputs.data, dataset=self._dataset_id)

                    if best_cost > loss.item() or iteration == 0:
                        best_inputs = inputs.data.cpu().clone()
                        best_cost = loss.item()

                optimizer.state = collections.defaultdict(dict)
                saved_best_inputs.extend([d.squeeze() for d in best_inputs])
                saved_labels.extend([d.squeeze() for d in targets.cpu()])

            if self._args.fedsd2c_store_images:
                dir_path = os.path.join(self._args.sys_res_root, self._args.save_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                path = os.path.join(dir_path, f"client{self._id}_")
                torch.save(torch.stack(saved_best_inputs), path + "best.pt")
                torch.save(targets_all.cpu().clone(), path + "best_label.pt")

            if self._args.using_wandb:
                wandb.log({
                    f"C{self._id} synthetic image": wandb.Image(
                        denormalize(saved_best_inputs[0].cpu()).numpy().transpose((1, 2, 0))),
                    "iteration": 0,
                })

            for hook in loss_r_feature_layers:
                hook.close()

            augment = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=64,
                    scale=(1, 1),
                    antialias=True
                ),
                transforms.RandomHorizontalFlip()
            ])
            self.client_data[teacher_name] = DistilledDataset(saved_best_inputs, saved_best_inputs, augment)

        if self._args.using_wandb:
            loss_list_dict = {k: [] for k in loss_dict}
            for i in range(self.iterations_per_layer):
                for k in loss_list_dict:
                    loss_list_dict[k].append(loss_dict[k][i])
            loss_mean_dict = {k: np.array(l_list).mean(axis=0).tolist() for k, l_list in loss_list_dict.items()}
            loss_std_dict = {k: np.array(l_list).std(axis=0).tolist() for k, l_list in loss_list_dict.items()}
            for i in range(len(loss_mean_dict["ce"])):
                _dict = {}
                for k in loss_mean_dict:
                    _dict[f"loss {k} mean"] = loss_mean_dict[k][i]
                    _dict[f"loss {k} std"] = loss_std_dict[k][i]
                _dict["iteration"] = i
                wandb.log(_dict)
