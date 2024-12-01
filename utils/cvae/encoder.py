import torch
from torch import nn


class ConditionalEncoder(nn.Module):
    """Basic CNN"""

    def __init__(self, num_channels, image_size, z_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                num_channels, 32, 4, 2, 1
            ),  # B,  32, input_size / 2, input_size / 2
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, input_size / 4, input_size / 4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64, input_size / 8, input_size / 8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64, input_size / 16, input_size / 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),  # B, 64 * (input_size / 16) * (input_size / 16)
            nn.Linear(64 * int(image_size / 16) * int(image_size / 16), 500),
            # Multiply output dimension by 2, so that 0-z can parameterize mu and z-2z can parametereize log(var)
            nn.Linear(500, z_dim * 2),
        )

    def forward(self, X):
        return self.model(X)


class ConditionalEncoderAlt(nn.Module):
    """Basic CNN with a few layers different"""

    def __init__(self, num_channels, image_size, z_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 8, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(int(image_size / 2) * int(image_size / 2), 300),
            nn.Linear(300, z_dim * 2),
        )

    def forward(self, X):
        return self.model(X)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(
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


class ConditionalEncoderResNet(nn.Module):
    """CNN with ResNet structure"""

    def __init__(self, num_channels, image_size, z_dim):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False), ResBlock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True), ResBlock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True), ResBlock(256, 256, downsample=False)
        )

        # Final "layer"
        self.fc = nn.Linear(256 * int(image_size / 16) * int(image_size / 16), z_dim * 2)

    def forward(self, X):
        X = self.layer0(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = nn.Flatten()(X)
        X = self.fc(X)

        return X


if __name__ == "__main__":
    img_size = 32
    num_channels = 3
    z_dim = 50
    model = ConditionalEncoderResNet(num_channels, img_size, z_dim)
    # summary(model, (num_channels, img_size, img_size))
    print(model(torch.randn((3, num_channels, img_size, img_size))).shape)
