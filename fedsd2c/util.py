from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset


def patchnify(x, patch_size=16):
    # x -> B c h w
    bs, c, h, w = x.shape
    p = patch_size
    x = x.unfold(2, p, p).unfold(3, p, p)
    # x -> B c h w p p
    x = x.contiguous().view(bs, c, -1, p, p).permute(0, 2, 1, 3, 4)
    # a -> ( B no.of patches c p p )
    return x


def unpatchnify(x):
    # x -> B no.of patches c p p
    bs, n, c, p, _ = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs, c, int(np.sqrt(n)), int(np.sqrt(n)), p, p)
    # x -> B c h w
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(bs, c, p * int(np.sqrt(n)), p * int(np.sqrt(n)))
    return x


# use 0 to pad "other three picture"
def pad(input_tensor, target_height, target_width=None):
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(2)
    horizontal_padding = target_width - input_tensor.size(3)

    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    padded_tensor = F.pad(
        input_tensor, (left_padding, right_padding, top_padding, bottom_padding)
    )

    return padded_tensor


def batched_forward(model, tensor, batch_size):
    total_samples = tensor.size(0)

    all_outputs = []

    model.eval()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i: min(i + batch_size, total_samples)]

            output = model(batch_data)

            all_outputs.append(output)

    final_output = torch.cat(all_outputs, dim=0)

    return final_output


def cross_entropy(y_pre, y):
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def selector_coreset(n, model, images, labels, size, device, m=5, descending=False, ret_all=False):
    with torch.no_grad():
        # [mipc, m, 3, 224, 224]
        images = images.to(device)
        s = images.shape

        # [mipc * m, 3, 224, 224]
        images = images.permute(1, 0, 2, 3, 4)
        images = images.reshape(s[0] * s[1], s[2], s[3], s[4])

        # [mipc * m, 1]
        labels = labels.repeat(m).to(device).to(torch.int64)

        # [mipc * m, n_class]
        batch_size = 64  # Change it for small GPU memory
        preds = batched_forward(model, pad(images, size).to(device), batch_size)

        # [mipc * m]
        dist = cross_entropy(preds, labels)
        preds = torch.argmax(preds, dim=1)

        # [m, mipc]
        dist = dist.reshape(m, s[0])
        preds = preds.reshape(m, s[0])

        # [mipc]
        index = torch.argmin(dist, 0)
        dist = dist[index, torch.arange(s[0])]
        preds = preds[index, torch.arange(s[0])]

        # [mipc, 3, 224, 224]
        sa = images.shape
        images = images.reshape(m, s[0], sa[1], sa[2], sa[3])
        images = images[index, torch.arange(s[0])]

    indices = torch.argsort(dist, descending=descending)[:n]
    torch.cuda.empty_cache()
    if ret_all:
        rest_indices = torch.argsort(dist, descending=descending)[n:]
        return images[indices].detach(), dist[indices].detach(), images[rest_indices].detach(), dist[
            rest_indices].detach(), preds[rest_indices].detach(), indices
    return images[indices].detach(), dist[indices].detach(), indices


def mix_images(input_img, out_size, factor, n):
    s = out_size // factor
    remained = out_size % factor
    k = 0
    mixed_images = torch.zeros(
        (n, 3, out_size, out_size),
        requires_grad=False,
        dtype=torch.float,
    )
    h_loc = 0
    for i in range(factor):
        h_r = s + 1 if i < remained else s
        w_loc = 0
        for j in range(factor):
            w_r = s + 1 if j < remained else s
            img_part = F.interpolate(
                input_img.data[k * n: (k + 1) * n], size=(h_r, w_r)
            )
            mixed_images.data[
            0:n,
            :,
            h_loc: h_loc + h_r,
            w_loc: w_loc + w_r,
            ] = img_part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed_images


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None, device="cpu"):
    rand_index = torch.randperm(images.size()[0]).to(device)
    lam = np.random.beta(args.cutmix, args.cutmix)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None, device="cpu"):
    rand_index = torch.randperm(images.size()[0]).to(device)
    lam = np.random.beta(args.mixup, args.mixup)

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None, device="cpu"):
    if args.mix_type == "mixup":
        return mixup(images, args, rand_index, lam, device)
    elif args.mix_type == "cutmix":
        return cutmix(images, args, rand_index, lam, bbox, device)
    else:
        return images, None, None, None


class CategoryDataset(torch.utils.data.Dataset):
    def __init__(self, _dataset, mipc, ipc, shuffle=False, seed=42, **kwargs):
        super().__init__(**kwargs)
        if isinstance(_dataset, Subset):
            _indices = np.array(_dataset.indices)
            _dataset = _dataset.dataset
        else:
            _indices = np.array(range(len(_dataset)))
        self.class_indices = []
        self.dataset = _dataset

        targets = np.array(_dataset.targets)[_indices]
        unique_classes = np.unique(targets)
        for c in unique_classes:
            c_indices = np.where(targets == c)[0]
            if shuffle:
                g = np.random.default_rng(seed)
                g.shuffle(c_indices)
            if len(c_indices) > mipc:
                self.class_indices.append(_indices[c_indices[:mipc]])
            elif len(c_indices) > ipc:
                self.class_indices.append(_indices[c_indices])

    def __getitem__(self, idx):
        images = []
        labels = []
        for i in self.class_indices[idx]:
            img, label = self.dataset[i]
            images.append(img)
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.int64)
            labels.append(label)
        return torch.stack(images, 0), torch.stack(labels, 0)

    def __len__(self):
        return len(self.class_indices)


class DistilledDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        if self.transform is not None:
            x = self.transform(self.x[idx])
        else:
            x = self.x[idx]

        return x, self.y[idx]

    def __len__(self):
        return len(self.x)

    def filter_labels(self, clss):
        indices = []
        for i in range(len(self.y)):
            if self.y[i] in clss:
                indices.append(i)
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]


class DistillCIDDataset(DistilledDataset):
    def __init__(self, x, y, cids, transform=None):
        super().__init__(x, y, transform)
        self.cids = cids

    def __getitem__(self, idx):
        if self.transform is not None:
            x = self.transform(self.x[idx])
        else:
            x = self.x[idx]

        return x, self.y[idx], self.cids[idx]

    def filter_labels(self, clss):
        indices = []
        for i in range(len(self.y)):
            if self.y[i] in clss:
                indices.append(i)
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        self.cids = [self.cids[i] for i in indices]


class WrapperDataset(torch.utils.data.Dataset):
    """Wrapper dataset to put into a dataloader."""

    def __init__(self, X, y, z, transform=None):
        self.X = X
        self.y = y
        self.z = z
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.transform is not None:
            x = self.transform(self.X[idx])
        else:
            x = self.X[idx]
        return x, self.y[idx], self.z[idx]


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_no_policy(base_lr):
    def _lr_fn(iteration, epoch):
        return base_lr

    return lr_policy(_lr_fn)


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


class BNFeatureHook:
    def __init__(self, module):
        self.r_feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        # B C W H
        nch = inputs[0].shape[1]
        mean = inputs[0].mean([0, 2, 3])
        var = inputs[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class BNFeature2WayHook(BNFeatureHook):
    def __init__(self, module):
        super().__init__(module)
        self.r_feature = None
        self.hook = module.register_forward_hook(self.hook_fn)
        self.stored_mean = None
        self.stored_var = None
        self.storing = False

    def hook_fn(self, module, inputs, output):
        # B C W H
        nch = inputs[0].shape[1]
        mean = inputs[0].mean([0, 2, 3])
        var = inputs[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        if self.storing:
            self.stored_mean = mean
            self.stored_var = var
        else:
            r_feature = torch.norm(self.stored_var - var, 2) + torch.norm(self.stored_mean - mean, 2)
            self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def factorization_loss(f_a, f_b, reduction="mean"):
    # empirical cross-correlation matrix
    if f_a.ndim > 2:
        f_a = f_a.view(f_a.size(0), -1)
        f_b = f_b.view(f_b.size(0), -1)
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag

    return loss


def mse_loss(a, b, reduction="mean"):
    return F.mse_loss(a, b, reduction=reduction)


def gram_mse_loss(a, b, reduction="mean"):
    s = a.shape
    a, b = a.reshape(s[0], s[1], s[2] * s[3]), b.reshape(s[0], s[1], s[2] * s[3])
    gram_a = torch.bmm(a, a.transpose(1, 2)) / (s[2] * s[3])
    gram_b = torch.bmm(b, b.transpose(1, 2)) / (s[2] * s[3])
    return F.mse_loss(gram_a, gram_b, reduction=reduction)


means = {
    "CIFAR10": [0.4914, 0.4822, 0.4465],
    "TINYIMAGENET": [0.485, 0.456, 0.406],
    "tiny-imagenet": [0.4802, 0.4481, 0.3975],
    "imagenet-1k": [0.485, 0.456, 0.406],
    "Imagenette": [0.485, 0.456, 0.406],
    "CIFAR100": [0.5, 0.5, 0.5],
    # "openImg": [0.4914, 0.4822, 0.4465],
    "openImg": [0.485, 0.456, 0.406],
    "COVID": [0.5, 0.5, 0.5],
}
stds = {
    "CIFAR10": [0.2023, 0.1994, 0.2010],
    "tiny-imagenet": [0.2302, 0.2265, 0.2262],
    "TINYIMAGENET": [0.229, 0.224, 0.225],
    "imagenet-1k": [0.229, 0.224, 0.225],
    "Imagenette": [0.229, 0.224, 0.225],
    "CIFAR100": [0.5, 0.5, 0.5],
    # "openImg": [0.2023, 0.1994, 0.2010],
    "openImg": [0.229, 0.224, 0.225],
    "COVID": [0.5, 0.5, 0.5],
}


def clip(image_tensor, use_fp16=False, inplace=False, dataset="CIFAR10"):
    '''
    adjust the input based on mean and variance
    '''
    mean, std = means[dataset], stds[dataset]
    if use_fp16:
        mean = np.array(mean, dtype=np.float16)
        std = np.array(std, dtype=np.float16)
    else:
        mean = np.array(mean)
        std = np.array(std)
    if not inplace:
        image_tensor = image_tensor.clone()
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def clip_tiny(image_tensor):
    """
    adjust the input based on mean and variance, tiny-imagenet
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)

    return image_tensor


def denormalize_tiny(image_tensor):
    """
    convert floats back to input, tiny-imagenet
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def denormalize(image_tensor, use_fp16=False, inplace=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    if not inplace:
        image_tensor = image_tensor.clone()

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: tensor of [C, H, W]"""
    # lam = np.random.uniform(0.8, alpha)
    lam = alpha
    # img2 = np.random.randn(*img2.shape)

    img1 = img1.transpose(1, 2, 0)
    img2 = img2.transpose(1, 2, 0)
    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1))).transpose(2, 0, 1)
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1))).transpose(2, 0, 1)

    return img21, img12


class SynDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y, mode, dataset="TINYIMAGENET", fourier=False, fourier_src="img", fourier_lambda=0.9):
        """
        mode in ["random", "label", "ipc"]
        x1 [C_num, ipc, C, H, W]
        x2 [C_num, ipc, C, H, W]
        y [C_num, ipc]
        """
        assert mode in ["random", "label", "ipc"]
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.mode = mode
        self.fourier = fourier
        self.dataset = dataset
        self.fourier_lambda = fourier_lambda
        self.fourier_src = fourier_src

        s = self.x1.shape
        if self.mode == "random":
            idxs = np.random.permutation(s[0] * s[1])
            self.x1 = self.x1.reshape(s[0] * s[1], s[2], s[3], s[4])[idxs]
            self.x2 = self.x2.reshape(s[0] * s[1], s[2], s[3], s[4])[idxs]
            self.y = self.y.reshape(s[0] * s[1])[idxs]
        elif self.mode == "ipc":
            self.x1 = self.x1.permute(1, 0, 2, 3, 4).reshape(s[1] * s[0], s[2], s[3], s[4])
            self.x2 = self.x2.permute(1, 0, 2, 3, 4).reshape(s[1] * s[0], s[2], s[3], s[4])
            self.y = self.y.reshape(s[0] * s[1])
        elif self.mode == "label":
            self.x1 = self.x1.reshape(s[0] * s[1], s[2], s[3], s[4])
            self.x2 = self.x2.reshape(s[0] * s[1], s[2], s[3], s[4])
            self.y = self.y.reshape(s[0] * s[1])

    def __getitem__(self, idx):
        """
        if fourier
        """
        im1, im2 = self.x1[idx], self.x2[idx]
        if self.fourier:
            if self.fourier_src == "img":
                im2 = im2.numpy()
            elif self.fourier_src == "noise":
                im2 = np.random.randn(im1.numel()).reshape(*im1.shape)
            im2, _ = colorful_spectrum_mix(im1.numpy(), im2, self.fourier_lambda)
            # im1 = clip(torch.tensor(im1, dtype=torch.float), dataset=self.dataset)
            im2 = clip(torch.tensor(im2, dtype=torch.float), dataset=self.dataset)
        return im1, im2, self.y[idx]

    def __len__(self):
        return self.y.shape[0]

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def compute_psnr(real, syn, denorm=True):
    if denorm:
        real, syn = denormalize(real), denormalize(syn)
    
    return 10 * torch.log10(1 / (real - syn).pow(2).mean(dim=[-3,-2,-1]))


class OutputHook:
    def __init__(self, module):
        self.r_feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        # B C W H
        self.r_feature = output

    def close(self):
        self.hook.remove()