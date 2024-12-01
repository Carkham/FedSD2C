import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from .Imagenette import Imagenette
from PIL import Image
import os
import PIL
import h5py
import numpy as np
import pandas as pd


class TinyImageNet(torchvision.datasets.VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

        self.targets = [s[1] for s in self.data]

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images


class OpenImage(torchvision.datasets.VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'openImg/'
    client_splits = None

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(OpenImage, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val", "test"))

        if self.client_splits is None:
            self._make_dataset()

        self.data = []
        self.cid2idx = {}
        k = 0
        for idx, (cid, data_list) in enumerate(self.client_splits[split].items()):
            self.data.extend(data_list)
            self.cid2idx[idx] = (k, k + len(data_list))
            k += len(data_list)

        self.targets = [s[1] for s in self.data]

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)
        # avoid channel error
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)

    def _make_dataset(self):
        print('Make client split for OpenImage')
        self.client_splits = {
            "train": {},
            "test": {},
            "val": {}
        }

        csv_file = pd.read_csv(os.path.join(self.dataset_path, f"openImg_train.csv"))
        _dict = self.client_splits["train"]
        for row in csv_file.itertuples():
            cid = getattr(row, "client_id")
            path, label_id = getattr(row, "sample_path"), getattr(row, "label_id")
            if cid not in _dict:
                _dict[cid] = []
            _dict[cid].append((os.path.join(self.dataset_path, "train", path), label_id))

        csv_file = pd.read_csv(os.path.join(self.dataset_path, f"openImg_val.csv"))
        _dict = self.client_splits["val"]
        for row in csv_file.itertuples():
            cid = getattr(row, "client_id")
            path, label_id = getattr(row, "sample_path"), getattr(row, "label_id")
            if cid not in _dict:
                _dict[cid] = []
            _dict[cid].append((os.path.join(self.dataset_path, "val", path), label_id))

        csv_file = pd.read_csv(os.path.join(self.dataset_path, f"openImg_test.csv"))
        _dict = self.client_splits["test"]
        for row in csv_file.itertuples():
            cid = getattr(row, "client_id")
            path, label_id = getattr(row, "sample_path"), getattr(row, "label_id")
            if cid not in _dict:
                _dict[cid] = []
            _dict[cid].append((os.path.join(self.dataset_path, "test", path), label_id))


def load_data(name, root='./data', download=True, save_pre_data=True, aug=False):
    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN', "IMAGENET", 'CIFAR100',
                 'TINYIMAGENET', 'Imagenette', 'openImg']
    # assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)

    elif name == 'EMNIST':
        # byclass, bymerge, balanced, letters, digits, mnist
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.EMNIST(root=root, train=True, split='letters', download=download,
                                               transform=transform)
        testset = torchvision.datasets.EMNIST(root=root, train=False, split='letters', download=download,
                                              transform=transform)

    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)

    elif name == 'CelebA':
        # Could not loaded possibly for google drive break downs, try again at week days
        target_transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CelebA(root=root, split='train', target_type=list, download=download,
                                               transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CelebA(root=root, split='test', target_type=list, download=download,
                                              transform=transform, target_transform=target_transform)

    elif name == 'CIFAR10':
        train_val_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        test_transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=train_val_transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
        trainset.targets = torch.tensor(trainset.targets, dtype=torch.int64)
        testset.targets = torch.tensor(testset.targets, dtype=torch.int64)

    elif name == 'CIFAR100':
        train_val_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, transform=train_val_transform, download=True)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, transform=test_transform, download=True)
        trainset.targets = torch.tensor(trainset.targets, dtype=torch.int64)
        testset.targets = torch.tensor(testset.targets, dtype=torch.int64)

    elif name == 'QMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.QMNIST(root=root, what='train', compat=True, download=download,
                                               transform=transform)
        testset = torchvision.datasets.QMNIST(root=root, what='test', compat=True, download=download,
                                              transform=transform)

    elif name == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=transform)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.labels)
        testset.targets = torch.Tensor(testset.labels)

    elif name == 'IMAGENET':
        train_val_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
        ])
        # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])
        trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train',
                                                    transform=train_val_transform)
        testset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)
    elif name == 'TINYIMAGENET':
        if not aug:
            train_val_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                #                      std=[0.2302, 0.2265, 0.2262]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomHorizontalFlip()
            ])
        else:
            print("Client Dataset Augment!")
            train_val_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                #                      std=[0.2302, 0.2265, 0.2262]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomResizedCrop(64),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
            #                      std=[0.2302, 0.2265, 0.2262]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trainset = TinyImageNet(root=root, split='train', download=True, transform=train_val_transform)
        testset = TinyImageNet(root=root, split='val', download=True, transform=test_transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)
    elif name == "Imagenette":
        if not aug:
            train_val_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                #                      std=[0.2302, 0.2265, 0.2262]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomHorizontalFlip()
            ])
        else:
            print("Client Dataset Augment!")
            train_val_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                #                      std=[0.2302, 0.2265, 0.2262]),
                transforms.Resize([128, 128]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
            #                      std=[0.2302, 0.2265, 0.2262]),
            transforms.Resize([128, 128]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trainset = Imagenette(root=root, split="train",
                                                   download=not os.path.exists(os.path.join(root, "imagenette2")),
                                                   transform=train_val_transform)
        testset = Imagenette(root=root, split="val", download=False, transform=test_transform)
        trainset.targets = torch.Tensor([s[1] for s in trainset._samples])
        testset.targets = torch.Tensor([s[1] for s in testset._samples])
    elif name == "openImg":
        train_val_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                      (0.2023, 0.1994, 0.2010)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.RandomResizedCrop((128,128)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                      (0.2023, 0.1994, 0.2010)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        trainset = OpenImage(root=root, split="train", transform=train_val_transform)
        testset = OpenImage(root=root, split="test", transform=test_transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)
    elif name == "COVID":
        train_val_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        trainset = ImageFolder(root=os.path.join(root, "COVID", "train"), transform=train_val_transform)
        print(trainset.class_to_idx)
        testset = ImageFolder(root=os.path.join(root, "COVID", "test"), transform=test_transform)
        print(trainset.class_to_idx)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    len_classes_dict = {
        'MNIST': 10,
        'EMNIST': 26,  # ByClass: 62. ByMerge: 814,255 47.Digits: 280,000 10.Letters: 145,600 26.MNIST: 70,000 10.
        'FashionMNIST': 10,
        'CelebA': 0,
        'CIFAR10': 10,
        'QMNIST': 10,
        'SVHN': 10,
        'IMAGENET': 200,
        'TINYIMAGENET': 200,
        'Imagenette': 10,
        'CIFAR100': 100,
        'openImg': 600,
        'COVID': 3
    }

    len_classes = len_classes_dict[name]

    return trainset, testset, len_classes


def divide_data(num_client=1, num_local_class=10, dataset_name='emnist', i_seed=0, aug=True):
    torch.manual_seed(i_seed)

    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False, aug=aug)

    num_classes = len_classes
    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': []}
    config_division = {}  # Count of the classes for division
    config_class = {}  # Configuration of class distribution in clients
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes

    for i in range(num_client):
        config_class['f_{0:05d}'.format(i)] = []
        for j in range(num_local_class):
            cls = (i + j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]

            else:
                config_division[cls] += 1
            config_class['f_{0:05d}'.format(i)].append(cls)

    # print(config_class)
    # print(config_division)

    for cls in config_division.keys():
        indexes = torch.nonzero(trainset.targets == cls)
        num_datapoint = indexes.shape[0]
        indexes = indexes[torch.randperm(num_datapoint)]
        num_partition = num_datapoint // config_division[cls]
        for i_partition in range(config_division[cls]):
            if i_partition == config_division[cls] - 1:
                config_data[cls][1].append(indexes[i_partition * num_partition:])
            else:
                config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

    for user in tqdm(config_class.keys()):
        user_data_indexes = torch.tensor([])
        for cls in config_class[user]:
            user_data_index = config_data[cls][1][config_data[cls][0]]
            user_data_indexes = torch.cat((user_data_indexes, user_data_index))
            config_data[cls][0] += 1
        user_data_indexes = user_data_indexes.squeeze().int().tolist()
        user_data = Subset(trainset, user_data_indexes)
        # user_targets = trainset.target[user_data_indexes.tolist()]
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = user_data
        trainset_config['num_samples'] = len(user_data)

    #
    # test_loader = DataLoader(trainset_config['user_data']['f_00001'])
    # for i, (x,y) in enumerate(test_loader):
    #     print(i)
    #     print(y)

    return trainset_config, testset


def divide_data_with_dirichlet(n_clients, dataset_name='CIFAR10', beta=0.4, seed=2022, aug=True):
    min_size = 0
    min_require_size = 8

    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False, aug=aug)
    y_train = trainset.targets
    n_classes = len_classes

    N = len(y_train)
    generator = np.random.default_rng(seed)
    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': []}
    net_dataidx_map = {}
    if dataset_name == "openImg":
        assert n_clients <= len(trainset.cid2idx)
        for j in range(n_clients):
            start, stop = trainset.cid2idx[j]
            idx_batch = list(range(start, stop))
            user_data = Subset(trainset, idx_batch)
            net_dataidx_map[j] = idx_batch
            trainset_config['users'].append(j)
            trainset_config['user_data'][j] = user_data
            trainset_config['num_samples'].append(len(user_data))

        cls_record = record_net_data_stats(y_train, net_dataidx_map, None)

        return trainset_config, testset, cls_record

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(y_train == k)[0]
            generator.shuffle(idx_k)
            proportions = generator.dirichlet(np.repeat(beta, n_clients))
            proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_clients):
        generator.shuffle(idx_batch[j])
        user_data = Subset(trainset, idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        trainset_config['users'].append(j)
        trainset_config['user_data'][j] = user_data
        trainset_config['num_samples'].append(len(user_data))

    cls_record = record_net_data_stats(y_train, net_dataidx_map, None)

    return trainset_config, testset, cls_record


def divide_data_with_local_cls(n_clients, n_local_cls, dataset_name='CIFAR10', seed=2022, aug=False):
    rng = np.random.default_rng(seed)
    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False, aug=aug)

    num_classes = len_classes
    if n_local_cls == -1:
        n_local_cls = num_classes
    assert 0 < n_local_cls <= num_classes, "number of local class should smaller than global number of class"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': []}
    config_division = {}  # Count of the classes for division
    config_class = {}  # Configuration of class distribution in clients
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes
    net_dataidx_map = {}

    local_clss = rng.permutation(num_classes).reshape(n_clients, -1)

    for i in range(n_clients):
        user = 'f_{0:05d}'.format(i)
        config_class[user] = local_clss[i].tolist()
        trainset_config['users'].append(user)
        user_data_indexes = []
        for cls in config_class[user]:
            indexes = np.where(np.asarray(trainset.targets) == cls)[0]
            user_data_indexes.append(indexes)
        user_data_indexes = np.concatenate(user_data_indexes).squeeze().tolist()
        user_data = Subset(trainset, user_data_indexes)
        net_dataidx_map[user] = user_data_indexes
        trainset_config['user_data'][user] = user_data
        trainset_config['num_samples'].append(len(user_data))

    #
    # test_loader = DataLoader(trainset_config['user_data']['f_00001'])
    # for i, (x,y) in enumerate(test_loader):
    #     print(i)
    #     print(y)
    y_train = trainset.targets

    cls_record = record_net_data_stats(y_train, net_dataidx_map, None)

    return trainset_config, testset, cls_record


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


if __name__ == "__main__":
    # 'MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN'
    # data_dict = ['CIFAR10']
    #
    # for name in data_dict:
    #     print(name)
    #     divide_data_with_dirichlet(n_clients=10)
    # train_dataset = TinyImageNet('../data', split='train', download=True)
    # test_dataset = TinyImageNet('../data', split='val', download=False)
    trainset_config, testset, records = divide_data_with_dirichlet(10, 'openImg', seed=42)
    ys = {client: [data[i][1] for i in range(len(data))] for client, data in trainset_config['user_data'].items()}
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    fig, ax = plt.subplots(1, 1, figsize=(4, 50))
    for i, client in tqdm(enumerate(ys)):
        labels = ys[client]
        xx = []
        yy = []
        ss = []
        for label_i in range(200):
            xx.append(i)
            yy.append(label_i)
            ss.append(len(np.where(np.array(labels) == label_i)[0]))
        ss = np.array(ss)
        plt.scatter(xx, yy, s=ss / 10)
        print(ss)
    plt.show()

    # print(len(train_dataset))
    # print(len(test_dataset))
    #
    # print(test_dataset.targets)
