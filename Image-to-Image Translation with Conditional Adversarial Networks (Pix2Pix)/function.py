import torch
from torchvision.utils import save_image

def train(epoch, trainer, cfg):

    for i, batch in enumerate(trainer.train_dataloader):
        gt = batch["gt"].cuda()
        label = batch["label"].cuda()

        real = torch.cuda.FloatTensor(label.size(0), *cfg.patch).fill_(1.0)
        fake = torch.cuda.FloatTensor(label.size(0), *cfg.patch).fill_(0.0)

        #################################################################
        ## Generator train
        trainer.optimizer_G.zero_grad()

        ## generated img
        generated_imgs = trainer.generator(label)
        fake_imgs = trainer.discriminator(generated_imgs, label)

        ## loss function
        GAN_loss = trainer.adversarial_loss(fake_imgs, real)
        Pixel_loss = trainer.pixelwise_loss(generated_imgs, gt)

        G_loss = GAN_loss + cfg.lambda_pixel * Pixel_loss

        ## loss update
        G_loss.backward()
        trainer.optimizer_G.step()


        #################################################################
        ## Discriminator train
        trainer.optimizer_D.zero_grad()

        real_loss = trainer.adversarial_loss(trainer.discriminator(gt, label), real)
        fake_loss = trainer.adversarial_loss(trainer.discriminator(generated_imgs.detach(), label), fake)
        D_loss = (real_loss + fake_loss) * 0.5

        D_loss.backward()
        trainer.optimizer_D.step()

        done = epoch * len(trainer.train_dataloader) + i
        if done % cfg.sample_interval == 0:
            imgs = next(iter(trainer.val_dataloader))
            gt = imgs["gt"].cuda()
            label = imgs["label"].cuda()
            generated_imgs = trainer.generator(label)
            img_sample = torch.cat((label.data, generated_imgs.data, gt.data), -2)
            save_image(img_sample, f"{cfg.result_dir}/{done}.png", nrow = 5, normalize = True)

    print(f"[Epoch {epoch}/{cfg.epochs}] [D loss: {D_loss.item():.6f}] [G loss: {G_loss.item():.6f}]")




