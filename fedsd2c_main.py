import copy
import os
import random
import signal
import sys
from typing import Dict, Set, Tuple

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import Logger, fed_args, read_config, log_config, AverageMeter
from utils.fed_utils import init_model, assign_dataset
from preprocessing.baselines_dataloader import divide_data_with_dirichlet
from fedsd2c.client import FedSD2CClient
from fedsd2c.server import FedSD2CServer
from fedsd2c.util import DistilledDataset

# Initialize parameters
args = fed_args()
args = read_config(args.config, args)

# Set up logging
if Logger.logger is None:
    logger = Logger()
    os.makedirs("train_records/", exist_ok=True)
    logger.set_log_name(os.path.join("train_records", f"train_record_{args.save_name}.log"))
    logger = logger.get_logger()
    log_config(args)

using_wandb = args.using_wandb

# Clear GPU cache
torch.cuda.empty_cache()

def test_model(model: nn.Module, testset, batch_size: int, device: str) -> float:
    """Test model accuracy on test set"""
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model.to(device)
    correct = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            
    return correct / len(testset)

def test_specified(model: nn.Module, testset, batch_size: int, device: str, 
                  specified_labels: list) -> Tuple[float, Dict]:
    """Test model accuracy on specified labels"""
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model.to(device)
    correct = 0
    specified_acc = {label: AverageMeter() for label in specified_labels}
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            
            for label in specified_labels:
                mask = (y == label)
                if mask.any():
                    acc = (pred[mask] == y[mask]).float().mean().item()
                    specified_acc[label].update(acc, mask.sum().item())
                    
    return correct / len(testset), specified_acc

def get_specified_labels(client, ipc: int) -> Tuple[Set, Set]:
    """Get label sets that need to be distilled"""
    dataset = client._train_data
    indices = np.array(dataset.indices)
    targets = np.array(dataset.dataset.targets, dtype=np.int64)[indices]
    unique_classes = np.unique(targets)
    
    containing_labels = set()
    distilled_labels = set()
    
    for c in unique_classes:
        containing_labels.add(c)
        if (targets == c).sum() > ipc:
            distilled_labels.add(c)
            
    return distilled_labels, containing_labels

def setup_environment():
    """Set up runtime environment"""
    # Set random seeds
    random.seed(args.sys_i_seed)
    np.random.seed(args.sys_i_seed)
    torch.manual_seed(args.sys_i_seed)
    torch.cuda.manual_seed(args.sys_i_seed)
    
    # Set CUDNN
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Set number of threads
    torch.set_num_threads(10)

def main():
    """Main function"""
    setup_environment()
    
    # Initialize client dictionary
    client_dict = {}
    
    # Data partitioning
    logger.info('======================Setup Clients==========================')
    if args.sys_dataset_dir_alpha is None:
        raise NotImplementedError("sys_dataset_dir_alpha is None")
        
    logger.info('Using divide data with dirichlet')
    trainset_config, testset, cls_record = divide_data_with_dirichlet(
        n_clients=args.sys_n_client,
        beta=args.sys_dataset_dir_alpha,
        dataset_name=args.sys_dataset,
        seed=42
    )
    logger.info(f'Clients in Total: {len(trainset_config["users"])}')

    # Initialize server
    server = FedSD2CServer(
        args, trainset_config['users'], 
        epoch=args.server_n_epoch,
        batch_size=args.server_bs,
        lr=args.server_lr,
        momentum=args.server_momentum,
        num_workers=args.server_n_worker,
        dataset_id=args.sys_dataset,
        model_name=args.sys_model,
        i_seed=args.sys_i_seed
    )
    server.load_testset(testset)

    # Get dataset parameters
    num_class, img_dim, image_channel = assign_dataset(args.sys_dataset)

    # Client training loop
    for client_id in trainset_config['users']:
        # Initialize client
        client = FedSD2CClient(args, client_id, dataset_id=args.sys_dataset)
        client_dict[client_id] = client
        client.load_train(trainset_config['user_data'][client_id])
        client.load_cls_record(cls_record[client_id])

        # Initialize model
        model = init_model(args.sys_model, num_class, image_channel, im_size=img_dim)
        specified_labels, containing_labels = get_specified_labels(
            client,
            client.ipc * client.factor ** 2
        )

        # Load or train model
        if args.client_model_root is not None:
            model_path = os.path.join(args.client_model_root, f"c{client_id}.pt")
            weight = torch.load(model_path, map_location="cpu")
            logger.info(f"Load Client {client_id} from {model_path}")
            model.load_state_dict(weight)
            model = model.to(client._device)
        else:
            logger.info(f"Client {client_id} local training")
            model, _ = client.train(model)

        # Execute distillation
        if args.client_instance == "coreset":
            ret_x, ret_y = client.coreset_stage(model)
        elif args.client_instance == "random":
            ret_x, ret_y = client.random_stage(model)
        elif args.client_instance == "coreset+dist_syn":
            ret_x, ret_y = client.coreset_stage(model)
            ret_x, ret_y = client.synthesis_stage(model)
        elif args.client_instance == "random+dist_syn":
            ret_x, ret_y = client.random_stage(model)
            ret_x, ret_y = client.synthesis_stage(model)
        else:
            raise NotImplementedError("Not implemented yet.")

        
        # Server receives distillation results
        # Data augmentation
        augment = transforms.Compose([
            transforms.RandomResizedCrop(size=img_dim, scale=(1, 1), antialias=True),
            transforms.RandomHorizontalFlip()
        ])
        server.rec_distill(
            client._id,
            model,
            DistilledDataset(ret_x, ret_y, augment),
            list(specified_labels)
        )

    # Server training
    server.train_distill()

def term_sig_handler(signum, frame):
    """Signal handler function"""
    print(f'caught signal: {signum}')
    if using_wandb:
        wandb.finish()
    sys.exit()

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, term_sig_handler)
    signal.signal(signal.SIGINT, term_sig_handler)
    
    # Run main program
    main()
    
    if using_wandb:
        wandb.finish()
