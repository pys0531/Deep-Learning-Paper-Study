import os
import os.path as osp
import sys

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network_list = {0: 'SSD'}
network = network_list[0]
exec(f'from configs.{network} import Config') 
cfg = Config()

## Setting Torch Cuda Device
cfg.device = device

## Directory Info
cfg.cur_dir = osp.dirname(osp.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, "..")
cfg.model_dir = osp.join(cfg.root_dir, "model_dump")
cfg.data_dir = osp.join(cfg.root_dir, "data")

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import make_folder, add_pypath
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.model_dir)