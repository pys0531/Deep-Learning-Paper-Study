from config import cfg
exec(f'from networks.{cfg.network} import {cfg.network}')

def get_network(pretrained = True):
    network = eval(cfg.network)()
    if pretrained:
        network.init_weights()
    return network