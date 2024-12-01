from tqdm import tqdm

from fed_baselines.client_base import FedClient
import copy

from fedsd2c.util import lr_cosine_policy
from utils.models import *
from utils.models_cvae import CVAE
from utils.util import reconstruction_loss, kl_divergence, AverageMeter

from torch.utils.data import DataLoader, default_collate

cvae_resize = transforms.Compose([
    transforms.Resize(32)
])


class FedCVAEClient(FedClient):
    def __init__(self, args, name, epoch, dataset_id, model_name):
        super().__init__(args, name, epoch, dataset_id, model_name)
        self.z_dim = args.cvae_z_dim
        self.beta = args.cvae_beta
        self._image_dim = 32
        self.model = CVAE(
            num_classes=self._num_class,
            num_channels=self._image_channel,
            z_dim=self.z_dim,
            image_size=self._image_dim,
            version=2
        )

    def train(self, model: nn.Module):
        """
        Client trains the model on local dataset using FedCVAE
        :return: Local updated model, number of local data points, training loss
        """
        model.train()
        model.to(self._device)
        mixup_transforms = []
        collate_fn = None
        if self.mixup_alpha > 0.0:
            mixup_transforms.append(RandomMixup(self._num_class, p=1.0, alpha=self.mixup_alpha))
        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(RandomCutmix(self._num_class, p=1.0, alpha=self.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):
                return mixupcutmix(*default_collate(batch))
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        lr_scheduler = lr_cosine_policy(self._lr, 0, self._epoch)

        # Training process
        recon_loss_accumulator = AverageMeter()
        kld_loss_accumulator = AverageMeter()
        pbar = tqdm(range(self._epoch))
        for epoch in pbar:
            epoch_recon_loss = AverageMeter()
            epoch_kld_loss = AverageMeter()
            lr_scheduler(optimizer, epoch, epoch)
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    x = cvae_resize(x)
                    b_x = x.to(self._device)  # Tensor on GPU
                    y_hot = torch.nn.functional.one_hot(y, num_classes=self._num_class).to(dtype=b_x.dtype)
                    b_y = y_hot.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    model.train()
                    X_recon, mu, logvar = model(b_x, b_y, self._device)
                    recon_loss = reconstruction_loss(self._image_channel, b_x, X_recon)
                    total_kld = kl_divergence(mu, logvar)
                    total_loss = recon_loss + self.beta * total_kld

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                recon_loss_accumulator.update(recon_loss.data.cpu().item())
                kld_loss_accumulator.update(total_kld.data.cpu().item())
                epoch_recon_loss.update(recon_loss.data.cpu().item())
                epoch_kld_loss.update(total_kld.data.cpu().item())

                pbar.set_description('Epoch: %d' % epoch +
                                     '| Recon loss: %.4f ' % epoch_recon_loss.avg +
                                     '| Kld loss: %.4f' % epoch_kld_loss.avg +
                                     '| lr: %.4f ' % optimizer.state_dict()['param_groups'][0]['lr'])

        return model, recon_loss_accumulator.avg, kld_loss_accumulator.avg
