import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import cfg
import os.path as osp
from tqdm import tqdm

def train(trainer, ):
    train_writer = SummaryWriter(log_dir=osp.join(cfg.model_dir, "log"))
    val_writer = SummaryWriter(log_dir=osp.join(cfg.model_dir, "log"))
    losses = AverageMeter()
    acces = AverageMeter()
    
    while trainer.global_step < cfg.num_steps: # for epoch in range(trainer.start_epoch, cfg.num_epochs):
        total, correct = 0, 0
        
        epoch_iterator = tqdm(trainer.train_batch_generator,
              desc="Training (X / X Steps) (lr=X.X) (loss=X.X)",
              bar_format="{l_bar}{r_bar}",
              dynamic_ncols=True)
        trainer.model.train()
        trainer.optimizer.zero_grad()
        for step, (img, label) in enumerate(epoch_iterator):
            img = img.to(trainer.device)
            label = label.to(trainer.device)
            
            output = trainer.model(img)
            loss = trainer.criterion(output, label)
            
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(trainer.train_batch_generator):
                losses.update(loss.item() * cfg.gradient_accumulation_steps)
                
                nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm = 1.0)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                trainer.set_lr()

                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += torch.sum(predicted.eq(label))

                acc = correct / total * 100
                acces.update(acc.item())
                
                trainer.global_step += 1
                epoch_iterator.set_description(
                        f"Training ({trainer.global_step} / {cfg.num_steps} Steps) (lr={trainer.get_lr():.6f}) (loss={losses.val:2.5f})"
                    )
                
                train_writer.add_scalar("lr", scalar_value=trainer.get_lr(), global_step=trainer.global_step)
                train_writer.add_scalar("loss", scalar_value=losses.val, global_step=trainer.global_step)
                
                if trainer.global_step % cfg.save_steps == 0:
                    val_acc, val_loss = validation(trainer)
                    train_writer.add_scalar("accuracy", scalar_value=acces.avg, global_step=trainer.global_step)
                    val_writer.add_scalar("accuracy", scalar_value=val_acc, global_step=trainer.global_step)
                    val_writer.add_scalar("loss", scalar_value=val_loss, global_step=trainer.global_step)
                    
                    file_path = osp.join(cfg.root_dir, 'model_dump', f'snapshot_{trainer.global_step}.pth.tar')
                    trainer.save_model({
                        'step': trainer.global_step,
                        'network': trainer.model.state_dict(),
                        'optimizer': trainer.optimizer.state_dict(),
                    }, trainer.global_step)
                    trainer.model.train()
                    
                if trainer.global_step >= cfg.num_steps:
                    break

        
        
def validation(trainer):
    trainer.model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        total, correct = 0, 0

        for itr, (img, label) in enumerate(trainer.val_batch_generator):
            img = img.to(trainer.device)
            label = label.to(trainer.device)
            
            output = trainer.model(img)
            loss = trainer.criterion(output, label)
            losses.update(loss.item())
            
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += torch.sum(predicted.eq(label))
            
        if total != 0:
            acc = correct / total * 100
        else:
            acc = 0
            
        print(f"\n Validating... (loss={losses.avg:2.5f}) (acc={acc:.4f})")
        return acc, losses.avg
            
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