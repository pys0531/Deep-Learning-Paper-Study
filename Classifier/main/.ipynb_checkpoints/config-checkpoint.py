import os
import os.path as osp
import sys


network_list = {0: 'CNN', 1: "VGG", 2: 'Resnet', 3: 'MobileNetV2', 4: 'ViT'}
network = network_list[2]
exec(f'from configs.{network} import Config') 
cfg = Config()



## Directory Info
cfg.cur_dir = osp.dirname(osp.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, "..")
cfg.model_dir = osp.join(cfg.root_dir, "model_dump")
cfg.data_dir = osp.join(cfg.root_dir, "data")
cfg.vis_dir = osp.join(cfg.root_dir, "vis")

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import make_folder, add_pypath
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
add_pypath(cfg.vis_dir)
make_folder(cfg.model_dir)


