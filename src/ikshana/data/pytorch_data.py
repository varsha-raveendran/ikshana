# from typing import Literal Python 3.8
import operator
import numpy as np
import torch
from torchvision import datasets, transforms
from transformations.album_pipeline import album_transformation_support
import albumentations as A

class GetData:
    
    def __init__(self, name, path='../data'): # name:Literal['MNIST', 'CIFAR10'],

        self.path = path
        name = name.upper()
        self.gen_dataset = operator.attrgetter(name)(datasets)

    def build_data(self, transformations=None, *, train=True):
        '''
        Downloads MNIST Data from Torch Vision and performs trasnformations if given,
        or uses pre determined mean and standard deviation of MNIST to convert to Tensor.

        Args:
            trasnformations: List of Trasnformations Compose
            train: True to get Train Data, False for Test Data.
        Return:
            Train/Test Dataset of 60,000/10,000 MNIST Images. 
        '''

        if transformations is None:
            transformations = transforms.Compose([transforms.ToTensor()])

        if isinstance(transformations, A.core.composition.Compose):
            transformations = album_transformation_support(transformations)

        data = self.gen_dataset(self.path, train=train, download=True, transform=transformations)

        return data

    def build_loader(self, dataset:datasets, batch_size=64, **kwargs):
        '''
        Create the Train Loader for the given dataset, and batch_size
        Args:
            dataset: The MNIST Train Data Set
            batch_size: The batch size.
        '''
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
        return loader


