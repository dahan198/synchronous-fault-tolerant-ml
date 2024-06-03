from .resnet import *
from .simple_conv_mnist import SimpleConvMNIST
from .simple_conv_cifar10 import SimpleConvCIFAR


MODEL_REGISTRY = {
    'resnet18': ResNet18,
    'simple_mnist': SimpleConvMNIST,
    'simple_cifar10': SimpleConvCIFAR
}

