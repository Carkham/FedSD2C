import os
import json
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from json import JSONEncoder
from tqdm import tqdm

from fedsd2c.util import means, stds, lr_cosine_policy
from postprocessing.recorder import Recorder
from utils import Logger, AverageMeter
from utils.fed_utils import assign_dataset
from utils.models import *
from copy import deepcopy
from collections import defaultdict

from fed_baselines.server_base import FedServer


class FedAVGServer(FedServer):
    def __init__(self, args, client_list, dataset_id, model_name):
        """
        Server in the federated learning for Central Learning
        """
        super().__init__(args, client_list, dataset_id, model_name)
        self.client_model = {}
        # self.model = None
        # Clients' model and dataset
        self.client_dict = {}
        self.client_model = {}
        self.client_data = {}
        self.cls_2_client = defaultdict(list)
        self.client_cls_score = {}

    def train(self):
        """
        Client trains the model on local dataset
        :return: Local updated model, number of local data points, training loss
        """
        self.model.to(self._device)

        state_dict = {}
        for k in self.model.state_dict():
            avg_p = 0
            for client in self.client_dict.values():
                avg_p += client.model.state_dict()[k]
            state_dict[k] = avg_p / len(self.client_dict)
        self.model.load_state_dict(state_dict)
        # Test process
        acc = self.test()
        L = Logger()
        logger = L.get_logger()
        logger.info('Accuracy: %.4f ' % acc)

    def load_cls_record(self, cls_record):
        """
        Client loads the statistic of local label.
        :param cls_record: class number record
        """
        pass
