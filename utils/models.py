import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import functional as tF
import numpy as np
from collections import OrderedDict

import torchvision.transforms as transforms
from torch import Tensor

"""
We provide the models, which might be used in the experiments on FedD3, as follows:
    - AlexNet model customized for CIFAR-10 (AlexCifarNet) with 1756426 parameters
    - LeNet model customized for MNIST with 61706 parameters
    - Further ResNet models
    - Further Vgg models
"""


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=224, factor=2, stack_dim=0):
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor
        self.stack_dim = stack_dim

    def forward(self, image):
        cropper = transforms.RandomResizedCrop(
            self.size // self.factor,
            ratio=(1, 1),
            antialias=True,
        )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, self.stack_dim)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, H, W = tF.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

# AlexNet model customized for CIFAR-10 with 1756426 parameters
class AlexCifarNet(nn.Module):
    supported_dims = {32}

    def __init__(self):
        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 4096)
        out = self.classifier(out)
        return out


# LeNet model customized for MNIST with 61706 parameters
class LeNet(nn.Module):
    supported_dims = {28}

    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)  # 6 x 28 x 28
        out = F.max_pool2d(out, 2)  # 6 x 14 x 14
        out = F.relu(self.conv2(out), inplace=True)  # 16 x 7 x 7
        out = F.max_pool2d(out, 2)   # 16 x 5 x 5
        out = out.view(out.size(0), -1)  # 16 x 5 x 5
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)

        return out


# Further ResNet models
def generate_resnet(num_classes=10, in_channels=1, model_name="ResNet18"):
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=False)
    elif model_name == "ResNet34":
        model = models.resnet34(pretrained=False)
    elif model_name == "ResNet50":
        model = models.resnet50(pretrained=False)
    elif model_name == "ResNet101":
        model = models.resnet101(pretrained=False)
    elif model_name == "ResNet152":
        model = models.resnet152(pretrained=False)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    return model


# Further Vgg models
def generate_vgg(num_classes=10, in_channels=1, model_name="vgg11"):
    if model_name == "VGG11":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG11_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG13":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG13_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG16":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG16_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG19":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG19_bn":
        model = models.vgg11_bn(pretrained=True)

    # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    # first_conv_layer.extend(list(model.features))
    # model.features = nn.Sequential(*first_conv_layer)
    # model.conv1 = nn.Conv2d(num_classes, 64, 7, stride=2, padding=3, bias=False)

    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, num_classes)

    return model


class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(CNN, self).__init__()

        self.fp_con1 = nn.Sequential(OrderedDict([
            ('con0', nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)),
            ('relu0', nn.ReLU(inplace=True)),
            ]))

        self.ternary_con2 = nn.Sequential(OrderedDict([
            # Conv Layer block 1
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

            # Conv Layer block 2
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            # nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.fp_fc = nn.Linear(4096, num_classes, bias = False)

    def forward(self, x):
        x = self.fp_con1(x)
        x = self.ternary_con2(x)
        x = x.view(x.size(0), -1)
        x = self.fp_fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# Conv-3 model
class ConvNet(nn.Module):
    def __init__(
        self,
        num_classes,
        net_norm="batch",
        net_depth=3,
        net_width=128,
        channel=3,
        net_act="relu",
        net_pooling="avgpooling",
        im_size=(32, 32),
    ):
        # print(f"Define Convnet (depth {net_depth}, width {net_width}, norm {net_norm})")
        super(ConvNet, self).__init__()
        if net_act == "sigmoid":
            self.net_act = nn.Sigmoid()
        elif net_act == "relu":
            self.net_act = nn.ReLU()
        elif net_act == "leakyrelu":
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit("unknown activation function: %s" % net_act)

        if net_pooling == "maxpooling":
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == "avgpooling":
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == "none":
            self.net_pooling = None
        else:
            exit("unknown net_pooling: %s" % net_pooling)

        self.depth = net_depth
        self.net_norm = net_norm

        self.layers, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, return_features=False):
        for d in range(self.depth):
            x = self.layers["conv"][d](x)
            if len(self.layers["norm"]) > 0:
                x = self.layers["norm"][d](x)
            x = self.layers["act"][d](x)
            if len(self.layers["pool"]) > 0:
                x = self.layers["pool"][d](x)

        # x = nn.functional.avg_pool2d(x, x.shape[-1])
        out = x.view(x.shape[0], -1)
        logit = self.classifier(out)

        if return_features:
            return logit, out
        else:
            return logit

    def get_feature(
        self, x, idx_from, idx_to=-1, return_prob=False, return_logit=False
    ):
        if idx_to == -1:
            idx_to = idx_from
        features = []

        for d in range(self.depth):
            x = self.layers["conv"][d](x)
            if self.net_norm:
                x = self.layers["norm"][d](x)
            x = self.layers["act"][d](x)
            if self.net_pooling:
                x = self.layers["pool"][d](x)
            features.append(x)
            if idx_to < len(features):
                return features[idx_from : idx_to + 1]

        if return_prob:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            prob = torch.softmax(logit, dim=-1)
            return features, prob
        elif return_logit:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            return features, logit
        else:
            return features[idx_from : idx_to + 1]

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c * h * w)
        if net_norm == "batch":
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == "layer":
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == "instance":
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == "group":
            norm = nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == "none":
            norm = None
        else:
            norm = None
            exit("unknown net_norm: %s" % net_norm)
        return norm

    def _make_layers(
        self, channel, net_width, net_depth, net_norm, net_pooling, im_size
    ):
        layers = {"conv": [], "norm": [], "act": [], "pool": []}

        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]

        for d in range(net_depth):
            layers["conv"] += [
                nn.Conv2d(
                    in_channels,
                    net_width,
                    kernel_size=3,
                    padding=3 if channel == 1 and d == 0 else 1,
                )
            ]
            shape_feat[0] = net_width
            if net_norm != "none":
                layers["norm"] += [self._get_normlayer(net_norm, shape_feat)]
            layers["act"] += [self.net_act]
            in_channels = net_width
            if net_pooling != "none":
                layers["pool"] += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        layers["conv"] = nn.ModuleList(layers["conv"])
        layers["norm"] = nn.ModuleList(layers["norm"])
        layers["act"] = nn.ModuleList(layers["act"])
        layers["pool"] = nn.ModuleList(layers["pool"])
        layers = nn.ModuleDict(layers)

        return layers, shape_feat


if __name__ == "__main__":
    model_name_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    for model_name in model_name_list:
        model = generate_resnet(num_classes=10, in_channels=1, model_name=model_name)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        param_len = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of model parameters of %s :' % model_name, ' %d ' % param_len)

