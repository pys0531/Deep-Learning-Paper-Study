import os
import os.path as osp
from utils import make_folder, add_pypath

class Config:
    dataset = "CelebA" # "MNIST"

    latent_dim = 100
    img_shape = (64, 64, 3) # img.shape = h, w, 3

    batch_size = 128
    num_workers = 4

    lr = 0.0002
    betas = (0.5, 0.999)

    epochs = 200
    sample_interval = 2000

    cur_dir = osp.dirname(os.path.abspath(__file__))
    dataset_dir = osp.join(cur_dir, 'dataset')
    result_dir = osp.join(cur_dir, 'result')

cfg = Config()

make_folder(cfg.dataset_dir)
make_folder(cfg.result_dir)
