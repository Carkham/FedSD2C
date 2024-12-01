import os
import copy
import yaml
from .logger import Logger
import wandb

L = Logger()


class Loader(yaml.Loader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as fr:
            return yaml.load(fr, Loader)


Loader.add_constructor('!include', Loader.include)


def read_config(config_name: str, args):
    with open(config_name, "r") as f:
        config = yaml.load(f, Loader=Loader)
        if 'base' in config:
            _config = copy.deepcopy(config["base"])
            del config["base"]
            _config.update(config)
            config = copy.deepcopy(_config)

    for k, v_ in config.items():
        if k == 'config':
            continue
        if getattr(args, k) is not None:
            continue
        print('Updating configure file', k, v_)
        args.__setattr__(k, v_)
    
    if args.using_wandb:
        wandb.init(
            project="{}_dir{}".format(args.sys_dataset_dir_alpha, args.wandb_proj_name),
            config=args,
        )

    args.save_name = "{}_{}_{}_{}_s{}".format(args.save_name, args.client_instance, args.sys_model, args.sys_dataset,
                                              args.sys_i_seed)
    return args


def log_config(config):
    logger = L.get_logger()
    configs = vars(config)
    logger.info('================= Config =================')
    for key in configs.keys():
        logger.info('\t{} = {}'.format(key, configs[key]))
    logger.info('================= ====== =================')
