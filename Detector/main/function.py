import torch
from torch.utils.tensorboard import SummaryWriter

import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2

from config import cfg

def train(trainer):
    train_writer = SummaryWriter(log_dir=osp.join(cfg.model_dir, "log"))
    val_writer = SummaryWriter(log_dir=osp.join(cfg.model_dir, "log"))
    losses = AverageMeter()
    for epoch in range(trainer.start_epoch, cfg.num_epochs):
        trainer.model.train()
        
        total, correct = 0, 0
        epoch_iterator = tqdm(trainer.train_batch_generator,
                              desc="Training (X / X Steps) (lr=X.X) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for itr, (img, labels, bboxes, difficults) in enumerate(epoch_iterator):            
            trainer.optimizer.zero_grad()
            
            ## device to target
            img = img.to(cfg.device)
            labels = [label.to(cfg.device) for label in labels]
            bboxes = [bbox.to(cfg.device) for bbox in bboxes]
            difficults = [difficult.to(cfg.device) for difficult in difficults]
            
            
            p_locs, p_clss = trainer.model(img)
            loss = trainer.criterion(p_locs, p_clss, bboxes, labels)
                        
            loss.backward()
            trainer.optimizer.step()
            trainer.set_lr(epoch)
            
            losses.update(loss.item())

            epoch_iterator.set_description(
                        f"Training ({epoch} / {cfg.num_epochs} Steps) (lr={trainer.get_lr():.6f}) (loss={losses.val:2.5f})"
                    )
        print(f"Training ({epoch} / {cfg.num_epochs} Steps) : loss={losses.avg:2.5f}")
        train_writer.add_scalar("lr", scalar_value=trainer.get_lr(), global_step=epoch)
        train_writer.add_scalar("loss", scalar_value=losses.avg, global_step=epoch)
            
        if epoch%10 == 0:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)

        val_loss_avg = validation(trainer)
        val_writer.add_scalar("loss", scalar_value=val_loss_avg, global_step=epoch)
        
        losses.reset()
        del p_locs, p_clss, img, bboxes, labels  # free some memory since their histories may be stored
    
    trainer.save_model({
        'epoch': epoch,
        'network': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
    }, epoch)

        
def validation(trainer):
    losses = AverageMeter()
    trainer.model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for itr, (img, labels, bboxes, difficults) in enumerate(trainer.val_batch_generator):
            img = img.to(cfg.device)
            labels = [label.to(cfg.device) for label in labels]
            bboxes = [bbox.to(cfg.device) for bbox in bboxes]
            difficults = [difficult.to(cfg.device) for difficult in difficults]
            
            p_locs, p_clss = trainer.model(img)
            loss = trainer.criterion(p_locs, p_clss, bboxes, labels)
            losses.update(loss.item())

        print(f"val_loss: {losses.avg:.4f}")
    return losses.avg
            
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def data_test(img, bboxes, labels):
    for n, i in enumerate(img):
        i = i.detach().numpy().transpose(1,2,0).copy()
        for b in bboxes[n]:
            b = b.detach().numpy().copy()*300
            cv2.rectangle(i, b[:2].astype(np.int64), b[2:].astype(np.int64), (0, 255, 0), 3)
        cv2.imshow(f"i_{n}", i)
        print(f"labels_{n}: " , labels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()