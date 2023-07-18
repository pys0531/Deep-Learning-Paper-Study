import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from config import cfg
from model import UNet, Discriminator

from PIL import Image

exec("from " + cfg.dataset + " import " + cfg.dataset)


class Trainer:
    def __init__(self, ):
        pass

    def get_optimizer(self, generator, discriminator):
        optimizer_G = torch.optim.Adam(generator.parameters(), lr = cfg.lr, betas = cfg.betas)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = cfg.lr, betas = cfg.betas)
        return optimizer_G, optimizer_D

    def _make_barch_generator(self, ):
        transforms_train = transforms.Compose([
                           transforms.Resize(cfg.img_shape[:2], Image.BICUBIC), # h, w
                           transforms.ToTensor(), 
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])

        train_dataset = eval(cfg.dataset)(transforms_train, "train")
        val_dataset = eval(cfg.dataset)(transforms_train, "val")

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.train_batch_size, shuffle = True, num_workers = cfg.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = cfg.val_batch_size, shuffle = True, num_workers = cfg.num_workers)

    def _make_loss(self, ):
        adversarial_loss = nn.MSELoss().cuda()
        pixelwise_loss = nn.L1Loss().cuda()
        return adversarial_loss, pixelwise_loss

    def _make_model(self,):
        generator = UNet(cfg).cuda()
        discriminator = Discriminator(cfg).cuda()

        optimizer_G, optimizer_D = self.get_optimizer(generator, discriminator)

        adversarial_loss, pixelwise_loss = self._make_loss()

        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.adversarial_loss = adversarial_loss
        self.pixelwise_loss = pixelwise_loss
