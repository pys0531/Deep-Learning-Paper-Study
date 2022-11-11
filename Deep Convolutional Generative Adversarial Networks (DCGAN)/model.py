import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        def block(input_dim, output_dim, kernel_size, stride = 1, padding = 0, bias = True, normalize = True, activation = nn.ReLU(inplace=True)):
            layers = [nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias = bias)]
            if normalize:
                layers.append(nn.BatchNorm2d(output_dim))
            layers.append(activation)
            return layers

        self.cfg = cfg
        self.blocks = nn.Sequential(
                          *block(self.cfg.latent_dim, 1024, 4, 1, 0, bias = False),
                          *block(1024, 512, 4, 2, 1, bias = False),
                          *block(512, 256, 4, 2, 1, bias = False),
                          *block(256, 128, 4, 2, 1, bias = False),
                          *block(128, cfg.img_shape[2], 4, 2, 1, normalize = False, activation = nn.Tanh()), # 64
                       )

        self.weights_init()

    def forward(self, z):
        G_img = self.blocks(z)
        return G_img

    def weights_init(self,):
        m = self.modules()
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        def block(input_dim, output_dim, kernel_size, stride = 1, padding = 0, bias = True, normalize = True, activation = nn.LeakyReLU(0.2, inplace=True)):
            layers = [nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias = bias)]
            if normalize:
                layers.append(nn.BatchNorm2d(output_dim))
            layers.append(activation)
            return layers

        self.cfg = cfg
        self.blocks = nn.Sequential(
                          *block(cfg.img_shape[2], 128, 4, 2, 1, normalize = False),
                          *block(128, 256, 4, 2, 1, bias = False),
                          *block(256, 512, 4, 2, 1, bias = False),
                          *block(512, 1024, 4, 2, 1, bias = False),
                          *block(1024, 1, 4, 1, 0, normalize = False, activation = nn.Sigmoid()),
                       )

        self.weights_init()

    def forward(self, img):
        img = self.blocks(img)
        return img

    def weights_init(self,):
        m = self.modules()
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

