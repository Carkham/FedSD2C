import torch
from torch import nn


class View(nn.Module):
    """Class to change the size of a PyTorch tensor."""

    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ConditionalDecoder(nn.Module):
    """
    Decoder needs to take in a sampled z AND the true class passed from the forward method
    """

    def __init__(self, image_size, num_classes, num_channels, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(
                z_dim + num_classes, 64 * int(image_size / 16) * int(image_size / 16)
            ),  # B, 64 * (input_size / 16) * (input_size / 16)
            View(
                (-1, 64, int(image_size / 16), int(image_size / 16))
            ),  # B, 64, input_size / 16, input_size / 16
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 64, 4, 2, 1
            ),  # B,  64, input_size / 8, input_size / 8
            nn.BatchNorm2d(64, 1.0e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, 4, 2, 1
            ),  # B,  32, input_size / 4, input_size / 4
            nn.BatchNorm2d(32, 1.0e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 32, 4, 2, 1
            ),  # B,  32, input_size / 2, input_size / 2
            nn.BatchNorm2d(32, 1.0e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, num_channels, 4, 2, 1
            ),  # B, nc, input_size, input_size
        )

    def forward(self, z, y_hot):
        y_hot = y_hot.to(torch.float32)
        concat_vec = torch.cat((z, y_hot), 1)  # sampled z and a one-hot class vector

        return self.model(concat_vec)


class ConditionalDecoderAlt(nn.Module):
    """
    Decoder needs to take in a sampled z AND the true class passed from the forward method
    """

    def __init__(self, image_size, num_classes, num_channels, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, int(image_size / 2) * int(image_size / 2)),
            View((-1, 64, int(image_size / 16), int(image_size / 16))),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 6, 2, 1),
            nn.BatchNorm2d(64, 1.0e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 8, 2, 1),
            nn.BatchNorm2d(32, 1.0e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, num_channels, 4, 2, 1),
        )

    def forward(self, z, y_hot):
        y_hot = y_hot.to(torch.float32)
        concat_vec = torch.cat((z, y_hot), 1)  # sampled z and a one-hot class vector

        return self.model(concat_vec)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample):
        super().__init__()
        if upsample:
            self.conv1 = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.shortcut = nn.Sequential()

        self.conv2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        shortcut = self.shortcut(X)
        X = nn.ReLU()(self.bn1(self.conv1(X)))
        X = nn.ReLU()(self.bn2(self.conv2(X)))
        X = X + shortcut
        return nn.ReLU()(X)


class ConditionalDecoderResNet(nn.Module):
    """Decoder CNN with ResNet structure"""

    def __init__(self, num_classes, num_channels, z_dim):
        super().__init__()

        self.layer_final_reverse = nn.Sequential(
            nn.Linear(z_dim + num_classes, 1024),
            View((-1, 256, 2, 2)),
            nn.ReLU(True),
        )

        self.layer3_reverse = nn.Sequential(
            ResBlock(256, 256, upsample=False),
            ResBlock(256, 128, upsample=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=1),
        )

        self.layer2_reverse = nn.Sequential(
            ResBlock(128, 128, upsample=False),
            ResBlock(128, 64, upsample=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=1),
        )

        self.layer1_reverse = nn.Sequential(
            ResBlock(64, 64, upsample=False),
            ResBlock(64, 32, upsample=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=1),
        )

        self.layer0_reverse = nn.Sequential(
            nn.ConvTranspose2d(32, num_channels, kernel_size=6, stride=2, padding=2),
        )

    def forward(self, z, y_hot):
        y_hot = y_hot.to(torch.float32)
        concat_vec = torch.cat((z, y_hot), 1)  # sampled z and a one-hot class vector

        X = self.layer_final_reverse(concat_vec)
        X = self.layer3_reverse(X)
        X = self.layer2_reverse(X)
        X = self.layer1_reverse(X)
        X = self.layer0_reverse(X)

        return X


if __name__ == "__main__":
    img_size = 32
    num_classes = 10
    num_channels = 3
    z_dim = 50
    model = ConditionalDecoderResNet(num_classes, num_channels, z_dim)
    print(model(torch.randn((1, z_dim)), torch.randn((1, num_classes))).shape)
