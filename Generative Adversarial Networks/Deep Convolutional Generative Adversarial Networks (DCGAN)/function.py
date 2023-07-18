import torch
from torchvision.utils import save_image

def train(epoch, trainer, cfg):

    for i, (imgs, _) in enumerate(trainer.data_loader):
        real = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(1.0)
        fake = torch.cuda.FloatTensor(imgs.size(0), 1).fill_(0.0)

        real_imgs = imgs.cuda()

        #################################################################
        ## Generator train
        trainer.optimizer_G.zero_grad()

        ## random noise
        z = torch.normal(mean = 0, std = 1, size = (real_imgs.size(0), cfg.latent_dim, 1, 1)).cuda()

        ## generated img
        generated_imgs = trainer.generator(z)
        fake_imgs = trainer.discriminator(generated_imgs)

        ## loss function
        G_loss = trainer.adversarial_loss(fake_imgs.view(-1), real.view(-1))

        ## loss update
        G_loss.backward()
        trainer.optimizer_G.step()


        #################################################################
        ## Discriminator train
        trainer.optimizer_D.zero_grad()

        real_loss = trainer.adversarial_loss(trainer.discriminator(real_imgs).view(-1), real.view(-1))
        fake_loss = trainer.adversarial_loss(trainer.discriminator(generated_imgs.detach()).view(-1), fake.view(-1))
        D_loss = (real_loss + fake_loss) / 2

        D_loss.backward()
        trainer.optimizer_D.step()

        done = epoch * len(trainer.data_loader) + i
        if done % cfg.sample_interval == 0:
            save_image(generated_imgs.data[:25], f"{cfg.result_dir}/{done}.png", nrow = 5, normalize = True)

    print(f"[Epoch {epoch}/{cfg.epochs}] [D loss: {D_loss.item():.6f}] [G loss: {G_loss.item():.6f}]")
