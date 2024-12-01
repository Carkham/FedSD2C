import copy
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self) -> str:
        return f"avg: {self.avg} cnt: {self.cnt}"


def reconstruction_loss(num_channels, x, x_recon):
    """Compute the reconstruction loss comparing input and reconstruction
    using an appropriate distribution given the number of channels.

    :param x: the input image
    :param x_recon: the reconstructed image produced by the decoder

    :return: reconstruction loss
    """

    batch_size = x.size(0)
    assert batch_size != 0

    # Use w/one-channel images
    if num_channels == 1:
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, reduction="sum"
        ).div(batch_size)
    # Multi-channel images
    elif num_channels == 3:
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum").div(batch_size)
    else:
        raise NotImplementedError("We only support 1 and 3 channel images.")

    return recon_loss


def kl_divergence(mu, logvar):
    """Compute KL Divergence between the multivariate normal distribution of z
    and a multivariate standard normal distribution.

    :param mu: the mean of the predicted distribution
    :param logvar: the log-variance of the predicted distribution

    :return: total KL divergence loss
    """

    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    # Shortcut: KL divergence w/N(0, I) prior and encoder dist is a multivariate normal
    # Push from multivariate normal --> multivariate STANDARD normal X ~ N(0,I)
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)

    return total_kld


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


def jsdiv( logits, targets, T=1.0, reduction='batchmean' ):
    P = F.softmax(logits / T, dim=1)
    Q = F.softmax(targets / T, dim=1)
    M = 0.5 * (P + Q)
    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    return 0.5 * F.kl_div(torch.log(P), M, reduction=reduction) + 0.5 * F.kl_div(torch.log(Q), M, reduction=reduction)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def save_image_batch(imgs, output, batch_id=None, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).transpose(1, 2, 0).squeeze()
        imgs = Image.fromarray(imgs)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output
        for idx, img in enumerate(imgs):
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + '%d-%d.png' % (batch_id, idx))


def save_image_batch_labeled(imgs, targets, batch_dir, batch_id=None, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(batch_dir)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack == False:
        output_filename = batch_dir
        for idx, img in enumerate(imgs):
            os.makedirs(batch_dir + str(targets[idx].item()) + "/", exist_ok=True)
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))
            img.save(output_filename + str(targets[idx].item()) + "/" + '%d-%d.png' % (batch_id, idx))


def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root)  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s' % (
            self.root, len(self), self.transform)


class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.categories = [int(f) for f in os.listdir(root)]
        images = []
        targets = []
        for c in self.categories:
            category_dir = os.path.join(self.root, str(c))
            _images = [os.path.join(category_dir, f) for f in os.listdir(category_dir)]
            images.extend(_images)
            targets.extend([c for _ in range(len(_images))])
        self.images = images
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        img, target = Image.open(self.images[idx]), self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


class ImagePool(object):
    def __init__(self, root, remove=False):
        self.root = os.path.abspath(root)
        self.batch_dir = None
        self.batch_id = 0
        if remove and os.path.exists(self.root):
            import shutil
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

    def add(self, imgs, batch_id=None, targets=None, his=True):
        self.batch_id = batch_id
        if targets is None:
            if not his:
                batch_dir = os.path.join(self.root, "%d" % (batch_id)) + "/"
                self.batch_dir = batch_dir
            else:
                batch_dir = os.path.join(self.root, "%d" % (0)) + "/"
                self.batch_dir = batch_dir
            os.makedirs(self.batch_dir, exist_ok=True)
            save_image_batch(imgs, batch_dir, batch_id=self.batch_id, pack=False)
        else:
            if not his:
                batch_dir = os.path.join(self.root, "%d" % (batch_id)) + "/"
                self.batch_dir = batch_dir
            else:
                batch_dir = os.path.join(self.root, "%d" % (0)) + "/"
                self.batch_dir = batch_dir

            os.makedirs(self.batch_dir, exist_ok=True)
            save_image_batch_labeled(imgs, targets, batch_dir, batch_id=self.batch_id, pack=False)

    def get_dataset(self, transform=None, labeled=False):
        if labeled == False:
            return UnlabeledImageDataset(self.root, transform=transform)
        else:
            return LabeledImageDataset(self.batch_dir, transform=transform)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        """

        :rtype: object
        """
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)


class Ensemble_A(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble_A, self).__init__()
        self.models = nn.ModuleList(model_list)

    def forward(self, x, mdl_w=None):
        logits_total = 0
        if mdl_w is None:
            mdl_w = torch.ones(len(self.models)).to(x.device)
        for i, model in enumerate(self.models):
            logits = model(x)
            logits_total += logits * mdl_w[i]
        logits_e = logits_total / torch.sum(mdl_w)

        return logits_e


class Ensemble_M(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble_M, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_list = []
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_list.append(logits)
        # 把list送入到mlp中
        logits_e = torch.stack((logits_list[0], logits_list[1],
                                logits_list[2], logits_list[3], logits_list[4]))
        data = logits_e.permute(1, 2, 0)  # [bs,num_cls,5]
        return data


class WEnsemble(torch.nn.Module):
    def __init__(self, model_list, mdl_w_list):
        super(WEnsemble, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.mdl_w_list = mdl_w_list

    def forward(self, x, return_features=False):
        logits_total = 0
        feat_total = 0
        for i in range(len(self.models)):
            if return_features:
                logits, feat = self.models[i](x, return_features=return_features)
                feat_total += self.mdl_w_list[i] * feat
            else:
                logits = self.models[i](x, return_features=return_features)
            logits_total += self.mdl_w_list[i] * logits
        logits_e = logits_total / torch.sum(self.mdl_w_list)
        if return_features:
            feat_e = feat_total / torch.sum(self.mdl_w_list)
            return logits_e, feat_e
        return logits_e

    def feat_forward(self, x):
        out_total = 0
        for i in range(len(self.models)):
            out = self.models[i].feat_forward(x)
            out_total += out
        return out_total / len(self.models)


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    """
    输入为logits,输出为weight矩阵
    [[bs,num_cls]*5]=[bs,num_cls,5]  ----> [[bs,1]*5], 搭配上[bs,num_cls]
    给每个样本配上一个权重，应该为[bs,1]*[bs,num_cls]
    """

    def __init__(self, dim_in=500, dim_hidden=100, dim_out=5):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        # --------------------
        bs = x.shape[0]  # x:[bs,num_cls,5]
        ori_data = x.permute(2, 0, 1)  # [5,bs,num_cls]
        logits_total = 0
        # --------------------
        x = x.reshape(-1, x.shape[2] * x.shape[1])  # [bs,num_cls*5]
        x = self.layer_input(x)  # [bs,dim_hidden]
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)  # [bs, dim_out]
        # ----------
        y_prob = F.softmax(x, dim=1)  # [bs, 5]
        for i in range(5):
            tmp = y_prob[:, i].reshape(bs, -1).cuda()
            logits = ori_data[i].mul(tmp)  # [bs,10] [bs,5]取第i列，对应点乘
            logits_total += logits
        logits_final = logits_total / 5.0
        return logits_final
