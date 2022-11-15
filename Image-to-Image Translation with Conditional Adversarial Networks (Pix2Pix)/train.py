from base import Trainer
from function import train
from config import cfg

def main():
    trainer = Trainer()
    trainer._make_barch_generator()
    trainer._make_model()

    for epoch in range(cfg.epochs):
        train(epoch, trainer, cfg)
        quit()

if __name__ == "__main__":
    main()
