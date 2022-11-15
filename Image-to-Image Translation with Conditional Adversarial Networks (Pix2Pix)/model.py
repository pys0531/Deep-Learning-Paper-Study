import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride = 1, padding = 0, activation = nn.LeakyReLU(0.2, inplace=True), normalize = nn.InstanceNorm2d, dropout = 0.0):
        super(ConvBlock, self).__init__()

        bias = (normalize == nn.InstanceNorm2d) or not normalize
        layers = [nn.Conv2d(input_dim, output_dim, kernel_size, stride = stride, padding = padding, bias = bias)]
        if normalize:
            layers.append(normalize(output_dim))
        if activation:
            layers.append(activation)
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConvTranseBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride = 1, padding = 0, activation = nn.ReLU(inplace = True), normalize = nn.InstanceNorm2d, dropout = 0.0):
        super(ConvTranseBlock, self).__init__()

        bias = (normalize == nn.InstanceNorm2d) or not normalize
        layers = [nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride = stride, padding = padding, bias = bias)]
        if normalize:
            layers.append(normalize(output_dim))
        if activation:
            layers.append(activation)
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_connect):
        x = self.model(x)
        x = torch.cat((x, skip_connect), 1)
        return x


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()

        in_channel = cfg.img_shape[2]
        out_channel = cfg.img_shape[2]

        self.down1 = ConvBlock(in_channel, 64, 4, 2, 1, normalize = None)
        self.down2 = ConvBlock(64, 128, 4, 2, 1)
        self.down3 = ConvBlock(128, 256, 4, 2, 1)
        self.down4 = ConvBlock(256, 512, 4, 2, 1, dropout = 0.5)
        self.down5 = ConvBlock(512, 512, 4, 2, 1, dropout = 0.5)
        self.down6 = ConvBlock(512, 512, 4, 2, 1, dropout = 0.5)
        self.down7 = ConvBlock(512, 512, 4, 2, 1, dropout = 0.5)
        self.down8 = ConvBlock(512, 512, 4, 2, 1, normalize = None, dropout = 0.5)


        self.up1 = ConvTranseBlock(512, 512, 4, 2, 1, dropout = 0.5)
        self.up2 = ConvTranseBlock(1024, 512, 4, 2, 1, dropout = 0.5)
        self.up3 = ConvTranseBlock(1024, 512, 4, 2, 1, dropout = 0.5)
        self.up4 = ConvTranseBlock(1024, 512, 4, 2, 1, dropout = 0.5)
        self.up5 = ConvTranseBlock(1024, 256, 4, 2, 1)
        self.up6 = ConvTranseBlock(512, 128, 4, 2, 1)
        self.up7 = ConvTranseBlock(256, 64, 4, 2, 1)

        self.final = nn.Sequential(
                         nn.Upsample(scale_factor = 2),
                         nn.ZeroPad2d((1, 0, 1, 0)),
                         nn.Conv2d(128, out_channel, 4, padding = 1),
                         nn.Tanh()
                     )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)




class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        in_channel = cfg.img_shape[2]
        out_channel = cfg.img_shape[2]

        ## 실제 / 변환된 이미지를 판별 => dim * 2
        self.down1 = ConvBlock(in_channel * 2, 64, 4, 2, 1, normalize = None)
        self.down2 = ConvBlock(64, 128, 4, 2, 1)
        self.down3 = ConvBlock(128, 256, 4, 2, 1)
        self.down4 = ConvBlock(256, 512, 4, 2, 1)
        self.final = nn.Sequential(
                         nn.ZeroPad2d((1, 0, 1, 0)),
                         nn.Conv2d(512, 1, 4, padding = 1)
                     )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)

        d1 = self.down1(img_input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        return self.final(d4) # [1 X 16 X 16]








