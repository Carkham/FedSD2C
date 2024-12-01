import torch
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from torch import nn
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.cvae.decoder import (ConditionalDecoder, ConditionalDecoderAlt,
                            ConditionalDecoderResNet)
from utils.cvae.encoder import (ConditionalEncoder, ConditionalEncoderAlt,
                            ConditionalEncoderResNet)

import matplotlib.pyplot as plt

class CVAE(nn.Module):
    """
    A slight modification of the model proposed in original Beta-VAE paper (Higgins et al, ICLR, 2017).

    Compatible with input images that are of spatial dimension divisible by 16, includes a classifier as a component
    of the pipeline, and allows image generation conditional on a chosen class.
    """

    def __init__(self, num_classes, num_channels, z_dim, image_size, version):
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.z_dim = z_dim

        if image_size % 16 != 0:
            raise Exception("Image size must be divisible by 16")

        self.image_size = image_size

        # Latent dist for further sampling: multivariate normal, z ~ N(0, I)
        self.mvn_dist = MultivariateNormal(
            torch.zeros(self.z_dim), torch.eye(self.z_dim)
        )

        # Define neural models needed for this implementation
        if version == 0:
            print("Standard model")
            self.encoder = ConditionalEncoder(
                num_channels=self.num_channels,
                image_size=self.image_size,
                z_dim=self.z_dim,
            )
            self.decoder = ConditionalDecoder(
                image_size=self.image_size,
                num_classes=self.num_classes,
                num_channels=self.num_channels,
                z_dim=self.z_dim,
            )
        elif version == 1:
            print("Alt model")
            self.encoder = ConditionalEncoderAlt(
                num_channels=self.num_channels,
                image_size=self.image_size,
                z_dim=self.z_dim,
            )
            self.decoder = ConditionalDecoderAlt(
                image_size=self.image_size,
                num_classes=self.num_classes,
                num_channels=self.num_channels,
                z_dim=self.z_dim,
            )
        elif version == 2:
            print("ResNet Model")
            self.encoder = ConditionalEncoderResNet(
                image_size=self.image_size,
                num_channels=self.num_channels,
                z_dim=self.z_dim,
            )
            self.decoder = ConditionalDecoderResNet(
                num_classes=self.num_classes,
                num_channels=self.num_channels,
                z_dim=self.z_dim,
            )
        else:
            raise NotImplementedError(
                "The model you specified has not been implemented."
            )

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                self.kaiming_init(m)

    def reparametrize(self, mu, logvar):
        """Re-paramaterization trick to make our CVAE fully-differentiable"""
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def sample_z(self, num_samples, dist, width=(-1, 1)):
        """Sample latent vectors"""
        if dist == "mvn":  # multivariate normal
            z = self.mvn_dist.sample((num_samples,))
        elif dist == "truncnorm":  # truncated multivariate normal
            truncnorm_tensor = torch.FloatTensor(
                truncnorm.rvs(a=width[0], b=width[1], size=num_samples * self.z_dim)
            )
            z = torch.reshape(truncnorm_tensor, (num_samples, self.z_dim))
        elif dist == "uniform":  # uniform
            z = torch.FloatTensor(num_samples, self.z_dim).uniform_(*width)

        else:
            raise NotImplementedError(
                "Only multivariate normal (mvn), truncated multivariate normal (truncnorm), and uniform (uniform) distributions supported."
            )

        return z

    def forward(self, X, y_hot, device):
        distributions = self.encoder(X)

        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]

        # Re-paramaterization trick, sample latent vector z
        z = self.reparametrize(mu, logvar).to(device)

        # Decode latent vector + class info into a reconstructed image
        x_recon = self.decoder(z, y_hot)

        return x_recon, mu, logvar
