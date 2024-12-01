from utils.models import *


def assign_dataset(dataset_name):
    """
    Assign the parameters to a dataset
    :param dataset_name: Dataset name
    :return: num_class: Number of classes in the dataset
    :return: image_dim: Image dimensions
    :return: image_channel: Number of image channels
    """
    num_class = -1
    image_dim = -1
    image_channel = -1

    if dataset_name == 'MNIST':
        num_class = 10
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'FashionMNIST':
        num_class = 10
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'EMNIST':
        num_class = 27
        image_dim = 28
        image_channel = 1

    elif dataset_name == 'CIFAR10':

        num_class = 10
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'CIFAR100':

        num_class = 100
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'SVHN':

        num_class = 10
        image_dim = 32
        image_channel = 3

    elif dataset_name == 'IMAGENET' or dataset_name == 'TINYIMAGENET':

        num_class = 200
        image_dim = 64
        image_channel = 3

    elif dataset_name == 'Imagenette':

        num_class = 10
        image_dim = 128
        image_channel = 3

    elif dataset_name == 'openImg':

        num_class = 600
        image_dim = 256
        image_channel = 3

    elif dataset_name == 'COVID':

        num_class = 3
        image_dim = 128
        image_channel = 3

    return num_class, image_dim, image_channel


def init_model(model_name, num_class, image_channel, im_size=64):
    """
    Initialize the model for a specific learning task.
    :param model_name: Model name
    :param num_class: Number of classes in the dataset
    :param image_channel: Number of image channels
    :return: The initialized model
    """
    model = None
    if model_name == "ResNet18":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet50":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet34":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet101":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "ResNet152":
        model = generate_resnet(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "LeNet":
        model = LeNet(num_classes=num_class, in_channels=image_channel)
    elif model_name == "CNN":
        model = CNN(num_classes=num_class, in_channels=image_channel)
    elif model_name == "VGG11":
        model = generate_vgg(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "VGG11_bn":
        model = generate_vgg(num_classes=num_class, in_channels=image_channel, model_name=model_name)
    elif model_name == "AlexCifarNet":
        model = AlexCifarNet()
    elif "Conv" in model_name:
        model = ConvNet(
            num_classes=num_class,
            net_norm="batch",
            net_act="relu",
            net_pooling="avgpooling",
            net_depth=int(model_name[-1]),
            net_width=128,
            channel=3,
            im_size=(im_size, im_size),
        )
    else:
        print('Model is not supported')
    if model_name == "ResNet18" and im_size == 64:
        # Modifid resnet-18 for Tiny-ImageNet
        # https://github.com/zeyuanyin/tiny-imagenet/blob/275c667a116a39d7dd735cda52a607597db546d4/classification/train.py#L198
        model.conv1 = nn.Conv2d(image_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()

    return model
