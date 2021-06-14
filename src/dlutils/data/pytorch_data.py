# from typing import Literal Python 3.8
import operator
import numpy as np
import torch
from torchvision import datasets, transforms
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

def album_transformation_support(trans):
    '''
    A Closure for Albumentation Transform, since Albumentation trasnform
    doesn't directly work on img = trasnform(img). where as Torchvision 
    uses it directly.
    For Albumentaions we have to create trasnform function which will
    return trasnform(image=img)['image']

    Args:
        trans: Albumentations Trasnforms 
    return:
        Function for Transformation which trasnforms image and returns
        image as per Albumentation requirement.
    '''
    def inner(img):
      img = np.array(img)
      return trans(image=img)['image']
    return inner
    
