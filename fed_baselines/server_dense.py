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

from fedsd2c.util import WrapperDataset, lr_cosine_policy, BNFeatureHook, means, stds, denormalize
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


class DENSEServer(FedServer):
    def __init__(self, args, client_list, dataset_id, model_name):
        super().__init__(args, client_list, dataset_id, model_name)
        self.generator = Generator(
            args.dfkd_z_dim,
            ngf=64,
            img_size=args.dfkd_img_size,
            nc=3
        )
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

        # DENSE Parameters
        self.temperature = args.dfkd_temp
        self.r_bn = args.dfkd_r_bn
        self.r_adv = args.dfkd_r_adv
        self.r_balance = args.dfkd_r_bal
        self.r_oh = args.dfkd_r_oh
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
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=means[self._dataset_id], std=stds[self._dataset_id])
            ])

        self.data_pool = ImagePool(root=os.path.join(self._args.sys_res_root, args.save_name, "syn_data"), remove=args.dfkd_syn_data)

        # Teacher model
        self.teacher = None
        self.loss_r_feature_layers = []

    def train_generator(self, epoch):
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
        
        batch_size = self._args.dfkd_batch_size if self._args.dfkd_syn_data else 0
        l_ce, l_adv, l_bn, l = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        import math
        ratio = math.ceil(self._batch_size / batch_size)
        for idx, kk in enumerate(range(0, self._batch_size, batch_size)):
            best_cost = 1e6
            best_inputs = None
            z = torch.randn(size=(batch_size, self.z_dim), device=self._device)  #
            z.requires_grad = True
            targets = torch.randint(low=0, high=self._num_class, size=(batch_size,), device=self._device)
            targets = targets.sort()[0]
            # targets = None
            reset_model(self.generator)
            optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], lr=1e-3,
                                        betas=(0.5, 0.999))

            criterion_ce = nn.CrossEntropyLoss().to(self._device)
            pbar = tqdm(range(self.gen_iteration))
            # lr_scheduler = lr_cosine_policy(0.001, 0, self.gen_iteration)
            for step in pbar:
                # lr_scheduler(optimizer, step, step)
                inputs = self.generator(z)
                inputs = self.normalizer(inputs)  # crop and normalize
                t_out = self.teacher(inputs)
                # if targets is None:
                # targets = t_out.max(1)[1].detach().clone()

                if len(self.loss_r_feature_layers) == 0 or self.r_bn == 0:
                    loss_r_bn = torch.tensor(0).cuda()
                else:
                    loss_r_bn = sum([mod.r_feature for mod in self.loss_r_feature_layers]) / len(self.client_model)
                loss_ce = criterion_ce(t_out, targets)
                s_out = self.model(inputs)
                mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, T=3, reduction='none').sum(
                    1) * mask).mean()  # decision adversarial distillation

                loss = self.r_oh * loss_ce + self.r_bn * loss_r_bn + self.r_adv * loss_adv
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data.cpu().clone()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.generator.parameters(), max_norm=10)
                optimizer.step()

                l_adv.update(loss_adv.item())
                l_ce.update(loss_ce.item())
                l_bn.update(loss_r_bn.item())
                l.update(loss.item())

                pbar.set_description(f'(Training Generator) Step: {step}' +
                                    f'| Train loss: {l.avg:.4f} ' +
                                    f'| ADV loss: {self.r_adv:.3f}*{loss_adv.item():.4f} ' +
                                    f'| BN loss: {self.r_bn:.3f}*{loss_r_bn.item():4f} ' +
                                    f'| CE loss: {self.r_oh:.3f}*{loss_ce.item():.4f} ' +
                                    '| lr: %.4f ' % optimizer.state_dict()['param_groups'][0]['lr'])

            best_inputs = denormalize(best_inputs)
            self.data_pool.add(best_inputs, batch_id=int(epoch * ratio + idx), targets=targets.cpu())  # 生成了一个batch的数据
        return {
            "loss": l.avg,
            "ADV loss": l_adv.avg,
            "BN loss": l_bn.avg,
            "CE loss": l_ce.avg
        }

    def train_model(self, epoch, optimizer):
        self.teacher.eval()
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

        loss_func = nn.KLDivLoss('batchmean').to(self._device)
        lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)
        lr_scheduler(optimizer, epoch, epoch)

        loss_accumulator = AverageMeter()
        acc_accumulator = 0
        dataloader = self.get_data()
        pbar = tqdm(enumerate(dataloader))
        cnts = 0
        for step, (images) in pbar:
            with torch.no_grad():
                images = images.to(self._device)
                t_out = self.teacher(images)
                t_out = F.softmax(t_out / self.temperature, dim=1).detach()

            with torch.enable_grad():
                s_out = self.model(images)
                s_out = F.log_softmax(s_out / self.temperature, dim=1)

                loss = loss_func(s_out, t_out)  * (self.temperature * self.temperature)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                optimizer.step()

                pred = s_out.argmax(dim=1)
                target = t_out.argmax(dim=1)
                acc_accumulator += pred.eq(target.view_as(pred)).sum().item()
                cnts += pred.numel()
                acc = acc_accumulator / (cnts) * 100

                loss_accumulator.update(loss.item())

            pbar.set_description(
                '(Training Server model) Epoch/Step: {} / {}'.format(epoch, step) +
                '| Train loss: %.4f ' % loss.item() +
                '| ST acc: %.4f ' % acc +
                '| lr: %.4f ' % optimizer.state_dict()['param_groups'][0]['lr']
            )

        return loss_accumulator.avg

    def get_data(self):
        datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        data_loader = torch.utils.data.DataLoader(datasets, batch_size=self._batch_size, shuffle=True)
        return data_loader

    def train(self):
        eval_only = False
        if eval_only:
            self.model.load_state_dict(torch.load(os.path.join(self._args.sys_res_root, self._args.save_name, "model.pt"), map_location="cpu"))
            acc = self.test()
            L = Logger()
            logger = L.get_logger()
            logger.info(f"Accuracy: {acc:.4f}")
            return
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
            gl_dict = self.train_generator(epoch)

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
            self.testset.transform
        ])