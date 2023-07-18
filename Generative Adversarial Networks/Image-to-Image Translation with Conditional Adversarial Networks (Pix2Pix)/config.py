import os
import os.path as osp
from utils import make_folder, add_pypath

class Config:
    dataset = "Facades"

    # latent_dim = 100 # => not used in pix2pix
    lambda_pixel = 100 # Loss weight L1 pixel wise loss between real img and translated img
    img_shape = (256, 256, 3) # img.shape = h, w, 3
    patch = (1, img_shape[0] // 2 ** 4, img_shape[1] // 2 ** 4) # [1, 16, 16]


    train_batch_size = 128
    val_batch_size = 10
    num_workers = 16

    lr = 0.0002
    betas = (0.5, 0.999)

    epochs = 200
    sample_interval = 200
    nrow = 5 # visual row num

    cur_dir = osp.dirname(os.path.abspath(__file__))
    data_dir = osp.join(cur_dir, 'data')
    result_dir = osp.join(cur_dir, 'result')

cfg = Config()

add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.result_dir)
