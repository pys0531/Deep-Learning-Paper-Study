#import torch
#from torchvision import datasets
#import torchvision.transforms as transforms

#from model import Generator, Discriminator
from base import Trainer
from function import train
from config import cfg

def main():
    #transforms_train = transforms.Compose([
    #                       transforms.Resize(28), 
    #                       transforms.ToTensor(), 
    #                       transforms.Normalize([0.5], [0.5])
    #                   ])
    #train_dataset = datasets.MNIST(root = cfg.dataset_dir, train = True, download = True, transform = transforms_train)
    #data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True, num_worker = 4)

    #generator = Generator(cfg).cuda()
    #discriminator = Discriminator(cfg).cuda()

    #adversarial_loss = nn.BCELoss().cdua()

    #optimizer_G = torch.optim.Adam(generator.parameters(), lr = cfg.lr, betas = cfg.betas)
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = cfg.lr, betas = cfg.betas)

    trainer = Trainer()
    trainer._make_barch_generator()
    trainer._make_model()

    for epoch in range(cfg.epochs):
        train(epoch, trainer, cfg)


if __name__ == "__main__":
    main()
