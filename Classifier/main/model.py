from config import cfg
exec(f'from networks.{cfg.network} import {cfg.network}')

def get_network():
    network = eval(cfg.network)()
    return network