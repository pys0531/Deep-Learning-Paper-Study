import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        def block(input_dim, output_dim, normalize = True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.cfg = cfg
        self.blocks = nn.Sequential(
                          *block(self.cfg.latent_dim, 128, normalize = False),
                          *block(128, 256),
                          *block(256, 512),
                          *block(512, 1024),
                          nn.Linear(1024, 1 * self.cfg.input_img_size * self.cfg.input_img_size),
                          nn.Tanh()
                       )

    def forward(self, z):
        G_img = self.blocks(z)
        G_img = G_img.view(z.size(0), 1, self.cfg.input_img_size, self.cfg.input_img_size)
        return G_img


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.cfg = cfg
        self.blocks = nn.Sequential(
                          nn.Linear(1 * self.cfg.input_img_size * self.cfg.input_img_size, 512),
                          nn.LeakyReLU(0.2),
                          nn.Linear(512, 256),
                          nn.LeakyReLU(0.2),
                          nn.Linear(256, 1),
                          nn.Sigmoid()
                       )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        img = self.blocks(img)
        return img
