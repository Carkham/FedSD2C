#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
from collections import defaultdict
from json import JSONEncoder

import torch
from tqdm import tqdm

from utils import Logger, fed_args, read_config, log_config
from fed_baselines.client_base import EnsembleClient
from fed_baselines.server_base import EnsembleServer
from fed_baselines.client_cvae import FedCVAEClient
from fed_baselines.server_cvae import FedCVAEServer
from fed_baselines.server_dense import DENSEServer
from fed_baselines.server_coboost import CoBoostServer
from fed_baselines.server_dafl import DAFLServer
from fed_baselines.server_adi import ADIServer
from fed_baselines.server_central import CentralServer
from fed_baselines.server_fedavg import FedAVGServer

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data, divide_data_with_dirichlet, divide_data_with_local_cls, load_data
from utils.models import *

json_types = (list, dict, str, int, float, bool, type(None))

# 初始化参数
args = fed_args()
args = read_config(args.config, args)

# 设置日志
if Logger.logger is None:
    logger = Logger()
    os.makedirs("train_records/", exist_ok=True)
    logger.set_log_name(os.path.join("train_records", f"train_record_{args.save_name}.log"))
    logger = logger.get_logger()
    log_config(args)

using_wandb = args.using_wandb

# 清理GPU缓存
torch.cuda.empty_cache()

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


def fed_run():
    """
    Main function for the baselines of federated learning
    """

    algo_list = ["FedCVAE", "DENSE", "ENSEMBLE", "CoBoost", "DAFL", "ADI", "Central", "FedAVG"]
    assert args.client_instance in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ["CIFAR10", 'TINYIMAGENET', "CIFAR100", 'Imagenette', "openImg", "COVID"]
    assert args.sys_dataset in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN", "Conv4",
                  "Conv5", "Conv6"]
    assert args.sys_model in model_list, "The model is not supported"

    random.seed(args.sys_i_seed)
    np.random.seed(args.sys_i_seed)
    torch.manual_seed(args.sys_i_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.sys_i_seed)
    torch.set_num_threads(3)

    client_dict = {}
    recorder = Recorder()

    logger.info('======================Setup Clients==========================')
    if args.sys_dataset_dir_alpha is not None:
        logger.info('Using divide data with dirichlet')
        trainset_config, testset, cls_record = divide_data_with_dirichlet(n_clients=args.sys_n_client,
                                                                          beta=args.sys_dataset_dir_alpha,
                                                                          dataset_name=args.sys_dataset,
                                                                          seed=args.sys_i_seed)
    else:
        raise NotImplementedError("sys_n_local_class and sys_dataset_dir_alpha are both None")
    logger.info('Clients in Total: %d' % len(trainset_config['users']))

    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    if args.client_instance == 'FedCVAE':
        server = FedCVAEServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'DENSE':
        server = DENSEServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'ENSEMBLE':
        server = EnsembleServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'CoBoost':
        server = CoBoostServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'DAFL':
        server = DAFLServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'ADI':
        server = ADIServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'FedAVG':
        server = FedAVGServer(args, trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    else:
        raise NotImplementedError('Server error')
    server.load_testset(testset)
    server.load_cls_record(cls_record)

    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    losses = defaultdict(list)
    logger.info('--------------------- configuration stage ---------------------')
    for client_id in trainset_config['users']:
        if args.client_instance == 'FedCVAE':
            client_dict[client_id] = FedCVAEClient(args, client_id, epoch=args.client_instance_n_epoch,
                                                   dataset_id=args.sys_dataset, model_name=args.sys_model)
            server.client_dict[client_id] = client_dict[client_id]
            client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])
            _, recon_loss, kld_loss = client_dict[client_id].train(client_dict[client_id].model)
            losses["recon_loss"].append(recon_loss)
            losses["kld_loss"].append(kld_loss)
        elif args.client_instance in ['DENSE', 'ENSEMBLE', "CoBoost", "DAFL", "ADI", "FedAVG"]:
            client_class = EnsembleClient
            client_dict[client_id] = client_class(args, client_id, epoch=args.client_instance_n_epoch,
                                                  dataset_id=args.sys_dataset, model_name=args.sys_model)
            server.client_dict[client_id] = client_dict[client_id]
            client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])
            server.client_model[client_id] = client_dict[client_id].model
            if args.client_model_root is not None and os.path.exists(
                    os.path.join(args.client_model_root, f"c{client_id}.pt")):
                weight = torch.load(os.path.join(args.client_model_root, f"c{client_id}.pt"), map_location="cpu")
                logger.info("Load Client {} from {}".format(client_id, os.path.join(args.client_model_root, f"c{client_id}.pt")))
                client_dict[client_id].model.load_state_dict(weight)
            else:
                c_loss = client_dict[client_id].train(client_dict[client_id].model)
                if not os.path.exists(os.path.join(args.sys_res_root, args.save_name)):
                    os.makedirs(os.path.join(args.sys_res_root, args.save_name))
                if args.save_client_model:
                    torch.save(client_dict[client_id].model.state_dict(), os.path.join(args.sys_res_root, args.save_name, f"c{client_id}.pt"))
                losses["client_loss"].append(c_loss)
        else:
            raise NotImplementedError("Client instance is not supported")

    for l_name, loss_list in losses.items():
        logger.info('{}: {}'.format(l_name, [float(l) for l in loss_list]))

    server.train()


if __name__ == "__main__":
    fed_run()
