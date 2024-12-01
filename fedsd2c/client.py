import collections
import math
import random
from copy import deepcopy

from tqdm import tqdm
import os
import wandb
import torchvision.transforms as transforms
import torch.nn as nn

from utils.models import MultiRandomCrop, RandomMixup, RandomCutmix
from torch.utils.data import DataLoader, default_collate, TensorDataset
from fedsd2c.util import *
from utils.logger import Logger
from collections import defaultdict
from utils import AverageMeter
from utils.fed_utils import assign_dataset, init_model
from utils.models_gan import LargeGenerator
from utils.models import ConvNet
from torchvision.models import ResNet
from diffusers import AutoencoderKL


class FedSD2CClient(object):

    def __init__(self, args, client_id, dataset_id='MNIST'):
        """
        Client in the federated learning for FedD3
        :param client_id: Id of the client
        :param dataset_id: Dataset name for the application scenario
        """
        # Metadata
        self._id = client_id
        self._dataset_id = dataset_id
        self.args = args

        # Following private parameters are defined by dataset.
        self._image_length = -1
        self._image_width = -1
        self._image_channel = -1
        self._n_class, self._image_length, self._image_channel = assign_dataset(dataset_id)
        self._image_width = self._image_length

        # Initialize the parameters in the local client
        self._epoch = args.client_instance_n_epoch
        self._batch_size = args.client_instance_bs
        self._lr = args.client_instance_lr
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0
        self.mixup_alpha = args.client_instance_mixup_alpha
        self.cutmix_alpha = args.client_instance_cutmix_alpha

        # Local dataset
        self._train_data = None
        self._test_data = None
        self._sd_data = None

        # Local distilled dataset
        self._distill_data = {'x': [], 'y': []}
        self._rest_data = {'x': [], 'y': [], 'dist': [], 'pred': []}
        self.coreset_select_indxs = []

        # FastDD parameters
        self.input_size = self._image_width
        self.num_crop = self.args.fedsd2c_num_crop
        self.factor = 1
        self.mipc = self.args.fedsd2c_mipc
        self.ipc = self.args.fedsd2c_ipc
        self.iter_mode = self.args.fedsd2c_iter_mode

        self.iterations_per_layer = self.args.fedsd2c_iteration
        self.jitter = self.args.fedsd2c_jitter
        self.sre2l_lr = self.args.fedsd2c_lr
        self.l2_scale = self.args.fedsd2c_l2_scale
        self.tv_l2 = self.args.fedsd2c_tv_l2
        self.r_bn = self.args.fedsd2c_r_bn
        self.r_c = self.args.fedsd2c_r_c
        self.r_adv = 0
        self.first_bn_multiplier = 10.
        self.inputs_init = self.args.fedsd2c_inputs_init

        self.noise_type = self.args.fedsd2c_noise_type
        self.noise_s = self.args.fedsd2c_noise_s
        self.noise_p = self.args.fedsd2c_noise_p

        self.normalizer = transforms.Normalize(means[self._dataset_id], stds[self._dataset_id])

        self._cls_record = None

        # Training on GPU
        gpu = args.gpu_id
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_train(self, data):
        """
        Client loads the decentralized dataset, it can be Non-IID across clients.
        :param data: Local dataset for training.
        """
        self._train_data = {}
        # self._train_data = deepcopy(data)
        self._train_data = data
        self.n_data = len(data)

    def load_test(self, data):
        """
        Client loads the test dataset.
        :param data: Dataset for testing.
        """
        self._test_data = {}
        self._test_data = deepcopy(data)

    def load_cls_record(self, cls_record):
        """
        Client loads the statistic of local label.
        :param cls_record: class number record
        """
        self._cls_record = {}
        self._cls_record = {int(k): v for k, v in cls_record.items()}

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
            mixup_transforms.append(RandomMixup(self._n_class, p=1.0, alpha=self.mixup_alpha))
        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(RandomCutmix(self._n_class, p=1.0, alpha=self.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):
                return mixupcutmix(*default_collate(batch))
        train_loader = DataLoader(self._train_data, batch_size=self._batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)

        optimizer = torch.optim.SGD(model.parameters(), lr=self._lr, momentum=self._momentum, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)
        loss_func = nn.CrossEntropyLoss()

        # Training process
        loss_accumulator = AverageMeter()
        pbar = tqdm(range(self._epoch))
        local_step = 0
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
                        f"{self._id}C local_loss": loss.item(),
                        "iteration": local_step,
                    })
                    local_step += 1
            pbar.set_description('Epoch: %d' % epoch +
                                 '| Train loss: %.4f ' % epoch_loss.avg +
                                 '| lr: %.4f ' % optimizer.state_dict()['param_groups'][0]['lr'])

        return model, loss_accumulator.avg

    def test(self, model):
        """
        Server tests the model on test dataset.
        """
        test_loader = DataLoader(self._test_data, batch_size=self._batch_size, shuffle=False)
        model.to(self._device)
        accuracy_collector = 0
        for step, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                b_x = x.to(self._device)  # Tensor on GPU
                b_y = y.to(self._device)  # Tensor on GPU

                test_output = model(b_x)
                pred_y = torch.max(test_output, 1)[1].to(self._device).data.squeeze()
                accuracy_collector = accuracy_collector + sum(pred_y == b_y)
        accuracy = accuracy_collector / len(self._test_data)

        return accuracy.cpu().numpy()

    def get_ipc(self, label):
        return self.ipc

    def coreset_stage(self, model):
        model = deepcopy(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        _dataset = deepcopy(self._train_data)
        _dataset.dataset = deepcopy(_dataset.dataset)
        _dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([self._image_length, self._image_width]),
            transforms.Normalize(mean=means[self._dataset_id], std=stds[self._dataset_id]),
        ])
        _dataset = CategoryDataset(_dataset, mipc=self.mipc, ipc=self.ipc * self.factor, shuffle=True, seed=self.args.sys_i_seed)

        ret_x = []
        ret_y = []

        mrc = MultiRandomCrop(self.num_crop, self.input_size, 1, 1)
        model.to(self._device)

        for c, (images, labels) in enumerate(_dataset):
            with torch.no_grad():
                images = mrc(images)
                ipc = self.get_ipc(labels[0].item())
                images, dists, rest_images, rest_dists, rest_preds, selected_indices = selector_coreset(
                    ipc * self.factor,
                    model,
                    images,
                    labels,
                    self.input_size,
                    device=self._device,
                    m=self.num_crop,
                    descending=False,
                    ret_all=True
                )
                self._rest_data['x'].extend([data.squeeze() for data in torch.split(rest_images.cpu(), 1)])
                self._rest_data['y'].extend([labels[0].cpu().item() for _ in range(rest_images.shape[0])])
                self._rest_data['dist'].extend([data.squeeze() for data in torch.split(rest_dists.cpu(), 1)])
                self._rest_data['pred'].extend([data.squeeze() for data in torch.split(rest_preds.cpu(), 1)])
                selected_indice_in_dset = []
                for indice in selected_indices.cpu().numpy().tolist():
                    selected_indice_in_dset.append(_dataset.class_indices[c][indice])
                self.coreset_select_indxs.extend(selected_indice_in_dset)
                images = mix_images(images, self.input_size, 1, images.shape[0]).cpu()

            # (ipc, 3, H, W)
            ret_x.extend([data.squeeze() for data in torch.split(images.cpu(), 1)])
            ret_y.extend([labels[0].cpu().clone() for _ in range(images.shape[0])])
        # ret_y = [0] * len(ret_x)
        self._distill_data['x'] = ret_x
        self._distill_data['y'] = ret_y

        return ret_x, ret_y

    def random_stage(self, model):
        _dataset = deepcopy(self._train_data)
        _dataset.dataset = deepcopy(_dataset.dataset)
        _dataset.dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([self._image_length, self._image_width]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _dataset = CategoryDataset(_dataset, mipc=self.mipc, ipc=self.ipc, shuffle=True,
                                   seed=self.args.sys_i_seed)

        ret_x = []
        ret_y = []
        ret_score = {}

        for c, (images, labels) in enumerate(_dataset):
            with torch.no_grad():
                ipc = self.get_ipc(labels[0].item())
                indices = torch.randperm(len(images))[:ipc]
                images = images[indices]
                images = images.cpu()

            ret_x.extend([data.squeeze() for data in torch.split(images.cpu(), 1)])
            ret_y.extend([labels[0].cpu().clone() for _ in range(self.ipc)])
            # ret_y.extend([0] * int(images.shape[1]))
            ret_score[labels[0].item()] = 0
            self._rest_data['x'].extend([data.squeeze() for data in torch.split(images.cpu(), 1)])
            self._rest_data['y'].extend([labels[0].cpu().item() for _ in range(images.shape[0])])
            self._rest_data['dist'].extend([labels[0].cpu() for _ in range(images.shape[0])])
            self._rest_data['pred'].extend([labels[0].cpu() for _ in range(images.shape[0])])
        # ret_y = [0] * len(ret_x)
        self._distill_data['x'] = ret_x
        self._distill_data['y'] = ret_y

        return ret_x, ret_y, ret_score

    def synthesis_stage(self, model):
        logger = Logger()
        logger = logger.get_logger()

        ret_x = []
        ori_x = []
        ret_y = []
        ret_z = []
        loss_list = []
        loss_dict_list = {}

        # model init
        model = deepcopy(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        for p in vae.parameters():
            p.requires_grad = False

        # hook for loss computation init
        loss_r_feature_layers = []
        if isinstance(model, ResNet):
            loss_r_feature_layers.append(OutputHook(model.maxpool))
            for name, module in model.named_modules():
                if name in [f"layer{j}" for j in range(1, 5)]:
                    # print(f"Adding hook to {name}")
                    loss_r_feature_layers.append(OutputHook(module))
            loss_r_feature_layers.append(OutputHook(model.avgpool))
        elif isinstance(model, ConvNet):
            for j in range(4):
                # print(f"Adding hook to {j} pool")
                loss_r_feature_layers.append(OutputHook(model.layers["pool"][j]))
        else:
            raise NotImplementedError()
        loss_r_bn_layers = []
        if self.r_bn > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    loss_r_bn_layers.append(BNFeatureHook(module))

        synset, batch_size = self._build_synset()
        synloader = torch.utils.data.DataLoader(synset, batch_size=batch_size, shuffle=False)
        for i, batch in enumerate(synloader):
            original_img, perturbed_img, y = batch

            # distillate initialization with fourier transformation
            with torch.no_grad():
                vae.to(self._device)
                if "fourier" in self.inputs_init:
                    z = vae.encode(denormalize(perturbed_img).to(self._device)).latent_dist.mode().clone().detach()
                else:
                    z = vae.encode(denormalize(original_img).to(self._device)).latent_dist.mode().clone().detach()
            targets = y.to(self._device)
            entropy_criterion = nn.CrossEntropyLoss()
            z.requires_grad = True
            optimizer = torch.optim.AdamW([z], lr=self.sre2l_lr, betas=(0.5, 0.9), eps=1e-8)
            lr_scheduler = lr_cosine_policy(self.sre2l_lr, 0, self.iterations_per_layer)

            best_inputs = None
            best_z = None
            best_cost = 1e4
            losses = []
            loss_dicts = {}
            for iteration in range(self.iterations_per_layer):
                lr_scheduler(optimizer, iteration, iteration)

                # inputs gen
                inputs = vae.decode(z).sample
                inputs = self.normalizer(inputs)

                im = original_img.clone().to(self._device)
                _inputs = torch.cat([inputs, im], dim=0)

                aug_function = transforms.Compose([
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                ])
                # _inputs = aug_function(_inputs)

                im = _inputs[inputs.shape[0]:]
                with torch.no_grad():
                    model(im)
                    target_feat_lists = [mod.r_feature.clone().detach() for mod in loss_r_feature_layers]

                # _inputs = aug_function(inputs)
                _inputs = _inputs[:inputs.shape[0]]

                outputs = model(_inputs)
                input_feat_lists = [mod.r_feature for mod in loss_r_feature_layers]
                key_words = self.args.fedsd2c_loss.split("_")
                loss = 0
                loss_dict = {}
                for key_word in key_words:
                    cf = key_word.split("-")
                    if "gram" in cf:
                        loss_fn = gram_mse_loss
                    elif "factorization" in cf:
                        loss_fn = factorization_loss
                    else:
                        loss_fn = mse_loss

                    loss_feat = loss_fn(input_feat_lists[-1], target_feat_lists[-1], reduction="mean")
                    loss += loss_feat
                    loss_dict["feat"] = loss_feat.item()

                if self.r_bn > 0:
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(loss_r_bn_layers) - 1)]
                    loss_r_bn = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_bn_layers)])
                    loss += self.r_bn * loss_r_bn
                    loss_dict["r_bn"] = loss_r_bn.item()
                if self.r_c > 0:
                    loss_r_c = entropy_criterion(outputs, targets)
                    loss += self.r_c * loss_r_c
                    loss_dict["r_ce"] = loss_r_c.item()
                if self.r_adv > 0:
                    loss_r_adv = -mse_loss(inputs, original_img.clone().to(self._device), reduction="mean")
                    loss += self.r_adv * loss_r_adv
                    loss_dict["r_adv"] = loss_r_adv.item()
                assert loss != 0

                if best_cost > loss.item() or iteration >= 0:
                    best_inputs = inputs.data.cpu().clone()
                    if z is not None:
                        best_z = z.data.detach().cpu().clone()
                    best_cost = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                inputs.data = clip(inputs.data, dataset=self._dataset_id)
                losses.append(loss.item())
                for k, v in loss_dict.items():
                    if k not in loss_dicts:
                        loss_dicts[k] = [v]
                    else:
                        loss_dicts[k].append(v)
            if len(losses) == 0:
                losses = [0]

            # To simplify the implementation, we pass the decoded best_inputs directly back to the server,
            # skipping the process of decoding and decoding on the server
            ret_x.extend([data.squeeze() for data in torch.split(best_inputs, 1)])
            ori_x.extend([data.squeeze() for data in torch.split(original_img.data.cpu().clone(), 1)])
            ret_y.extend([data.squeeze() for data in torch.split(y.clone(), 1)])
            if "vae" in self.inputs_init:
                ret_z.extend([data.squeeze() for data in torch.split(best_z, 1)])

            logger.info("------------idx {} / {}----------".format(i * batch_size, len(self._distill_data['x'])))
            logger.info("loss avg: {}, final: {}, ".format(np.mean(losses), losses[-1]) + ", ".join(
                [f"{k}: {v[-1]}" for k, v in loss_dicts.items()]))
            loss_list.append(losses)
            for k, v in loss_dicts.items():
                if k not in loss_dict_list:
                    loss_dict_list[k] = [v]
                else:
                    loss_dict_list[k].append(v)

        if self.args.using_wandb:
            loss_mean = np.array(loss_list).mean(axis=0).tolist()
            loss_std = np.array(loss_list).std(axis=0).tolist()
            for i, loss in enumerate(loss_mean):
                wandb.log({
                    f"C{self._id} comp loss avg": loss,
                    f"C{self._id} comp loss std": loss_std[i],
                    "iteration": i,
                })
            for k, v in loss_dict_list.items():
                lm = np.array(v).mean(axis=0).tolist()
                ls = np.array(v).std(axis=0).tolist()
                for i, loss in enumerate(lm):
                    wandb.log({
                        f"C{self._id} {k} avg": loss,
                        f"C{self._id} {k} std": ls[i],
                        "iteration": i,
                    })
        del vae
        torch.cuda.empty_cache()
        return ret_x, ret_y

    def decode_latents(self, latents):
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()
        vae.to(self._device)
        bs = self.ipc * self.factor
        samples = []
        rng = np.random.default_rng(self.args.sys_i_seed)
        with torch.no_grad():
            for kk in range(0, len(latents), bs):
                z = latents[kk:kk + bs].to(self._device)
                if self.noise_type == "gaussian":
                    noise = torch.tensor(rng.normal(size=z.numel()), dtype=z.dtype).reshape(z.shape).to(
                        self._device) * self.noise_s
                    z = (1 - self.noise_p) * z + noise
                elif self.noise_type == "laplace":
                    noise = torch.tensor(rng.laplace(size=z.numel()), dtype=z.dtype).reshape(z.shape).to(
                        self._device) * self.noise_s
                    z = (1 - self.noise_p) * z + noise
                elif self.noise_type == "None":
                    pass
                else:
                    raise NotImplementedError()
                sample = vae.decode(z).sample.detach().clone().cpu()
                sample = self.normalizer(sample)
                samples.extend([data.squeeze() for data in torch.split(sample, 1)])

        return samples

    @property
    def all_select(self):
        """
        The client uploads all of the original dataset
        :return: All of the original images
        """
        return self._train_data

    def save_distilled_dataset(self, exp_dir='client_models', res_root='results'):
        """
        The client saves the distilled images in corresponding directory
        :param exp_dir: Experiment directory name
        :param res_root: Result directory root for saving the result files
        """
        agent_name = 'clients'
        model_save_dir = os.path.join(res_root, exp_dir, agent_name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(self._distill_data, os.path.join(model_save_dir, self._id + '_distilled_img.pt'))

    def _build_synset(self):
        dx1, dx2, dy = [], [], []
        for i in range(0, len(self._distill_data['x']), self.ipc * self.factor):
            idxs = np.random.permutation(self.ipc * self.factor).tolist()
            subset_x = torch.stack([self._distill_data['x'][i + idx] for idx in idxs])
            subset_y = torch.stack([self._distill_data['y'][i + idx] for idx in idxs])

            corres_idxs = np.where(np.array(self._rest_data['y']) == subset_y[0].item())[0]
            rest_x = torch.stack([self._rest_data['x'][idx] for idx in corres_idxs])
            rest_dists = torch.stack([self._rest_data['dist'][idx] for idx in corres_idxs])
            rest_preds = torch.stack([self._rest_data['pred'][idx] for idx in corres_idxs])

            indices = np.where(torch.argmax(rest_preds).numpy() == subset_y[0].item())[0]
            if indices.shape[0] != 0:
                rest_x, rest_dists = rest_x[indices], rest_dists[indices]
            indices = torch.argsort(rest_dists, descending=True)[:subset_x.shape[0]]
            if indices.shape[0] < subset_x.shape[0]:
                indices = indices.repeat((subset_x.shape[0] // indices.shape[0]) + 1)[:subset_x.shape[0]]
            rest_x = rest_x[indices]

            dx1.append(subset_x)
            dx2.append(rest_x)
            dy.append(subset_y)
        dx1 = torch.stack(dx1, dim=0)
        dx2 = torch.stack(dx2, dim=0)
        dy = torch.stack(dy, dim=0)

        if self.iter_mode == "random" or self.iter_mode == "label":
            bs = self.ipc * self.factor
        elif self.iter_mode == "ipc":
            bs = dx1.shape[0]

        return SynDataset(dx1, dx2, dy, self.iter_mode, fourier="fourier" in self.inputs_init,
                          fourier_lambda=self.args.fourier_lambda, dataset=self._dataset_id), bs
