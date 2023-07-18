import torch
import torch.backends.cudnn as cudnn

from config import cfg
from base import Trainer

def main():
    cudnn.fastest = True
    cudnn.benchmark = True
    
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
        
    for epoch in range(trainer.start_epoch, cfg.num_epochs):
        trainer.model.train()
        trainer.set_lr(epoch)
        
        total = 0
        correct = 0
        for itr, (img, label) in enumerate(trainer.train_batch_generator):
            trainer.optimizer.zero_grad()
            
            img = img.to(trainer.device)
            label = label.to(trainer.device)
            
            output = trainer.model(img)
            loss = trainer.criterion(output, label)
            
            loss.backward()
            trainer.optimizer.step()
            
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += torch.sum(predicted.eq(label))
            
        acc = correct / total * 100
        print(f"epoch: {epoch}/{cfg.num_epochs}  lr: {trainer.get_lr():.4f}  ==>  train_loss: {loss:.4f}, train_acc: {acc:.4f}", end = " || ")
        
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)
        
        validation(trainer)
        
        
def validation(trainer):
    trainer.model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for itr, (img, label) in enumerate(trainer.val_batch_generator):
            img = img.to(trainer.device)
            label = label.to(trainer.device)
            
            output = trainer.model(img)
            loss = trainer.criterion(output, label)
            
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += torch.sum(predicted.eq(label))
            
        if total != 0:
            acc = correct / total * 100
            print(f"val_loss: {loss:.4f}, val_acc: {acc:.4f}")
            
            
if __name__ == "__main__":
    main()