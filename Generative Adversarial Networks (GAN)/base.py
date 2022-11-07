import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from model import Generator, Discriminator
from config import cfg

class Trainer:
    def __init__(self, ):
        pass

    def get_optimizer(self, generator, discriminator):
        optimizer_G = torch.optim.Adam(generator.parameters(), lr = cfg.lr, betas = cfg.betas)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = cfg.lr, betas = cfg.betas)
        return optimizer_G, optimizer_D

    def _make_barch_generator(self, ):
        transforms_train = transforms.Compose([
                           transforms.Resize(28), 
                           transforms.ToTensor(), 
                           transforms.Normalize([0.5], [0.5])
                       ])
        train_dataset = datasets.MNIST(root = cfg.dataset_dir, train = True, download = True, transform = transforms_train)
        self.data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4)

    def _make_loss(self, ):
        adversarial_loss = nn.BCELoss().cuda()
        return adversarial_loss

    def _make_model(self,):
        generator = Generator(cfg).cuda()
        discriminator = Discriminator(cfg).cuda()

        optimizer_G, optimizer_D = self.get_optimizer(generator, discriminator)

        adversarial_loss = self._make_loss()

        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.adversarial_loss = adversarial_loss
