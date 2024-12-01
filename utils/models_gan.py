import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator_ACGan(nn.Module):

    def __init__(self, num_classes, z_dim, noise_label_combine='mul'):
        super(Generator_ACGan, self).__init__()

        self.data_set = 'Tiny-Imagenet'
        self.latent_dim = z_dim
        self.noise_label_combine = noise_label_combine
        self.n_classes = num_classes

        if self.noise_label_combine in ['cat']:
            input_dim = 2 * self.latent_dim
        elif self.noise_label_combine in ['cat_naive']:
            input_dim = self.latent_dim + self.n_classes
        else:
            input_dim = self.latent_dim

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
                                    nn.ReLU(True))

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))

        self.layer4_1 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
                                      nn.Tanh())

        self.layer4_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))

        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                                    nn.Tanh())

        self.layer4_3 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
                                      nn.Tanh())

        self.layer4_4 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(True))

        self.layer4_5 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(True))

        self.layer4_6 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                                      nn.Tanh())
        self.embedding = nn.Embedding(self.n_classes, self.latent_dim)

    def forward(self, noise, label):

        if self.noise_label_combine == 'mul':
            label_embedding = self.embedding(label)
            h = torch.mul(noise, label_embedding)
        elif self.noise_label_combine == 'add':
            label_embedding = self.embedding(label)
            h = torch.add(noise, label_embedding)
        elif self.noise_label_combine == 'cat':
            label_embedding = self.embedding(label)
            h = torch.cat((noise, label_embedding), dim=1)
        elif self.noise_label_combine == 'cat_naive':
            label_embedding = Variable(torch.cuda.FloatTensor(len(label), self.n_classes))
            label_embedding.zero_()
            label_embedding.scatter_(1, label.view(-1, 1), 1)
            h = torch.cat((noise, label_embedding), dim=1)
        else:
            label_embedding = noise
            h = noise

        x = h.view(-1, h.shape[1], 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.data_set in ['Tiny-Imagenet', 'FOOD101']:
            x = self.layer4_2(x)
            x = self.layer5(x)
        elif self.data_set in ['FMNIST', 'MNIST']:
            x = self.layer4_3(x)
        elif self.data_set in ['']:
            x = self.layer4_4(x)
            x = self.layer4_5(x)
            x = self.layer4_6(x)
        else:
            x = self.layer4_1(x)

        return x, h, label_embedding


class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(z_dim, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ACGAN_Generator(nn.Module):
    def __init__(self, num_classes=200, z_dim=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.n_classes = num_classes

        self.label_emb = nn.Embedding(self.n_classes, z_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(z_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, nc, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class LargeGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(LargeGenerator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 4 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 4, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
