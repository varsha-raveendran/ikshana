import torch
from torchvision import datasets, transforms

class MNISTData:
    
    def __init__(self, mean=(0.1307,), std=(0.3081,)):
        self.mean = mean
        self.std = std

    def build_data(self, transformations=None, *, train=True):
        '''
        Downloads MNIST Data from Torch Vision and performs trasnformations if given,
        or uses pre determined mean and standard deviation of MNIST to convert to Tensor and Normalize.

        Args:
            trasnformations: List of Trasnformations Compose
            train: True to get Train Data, False for Test Data.
        Return:
            Train/Test Dataset of 60,000/10,000 MNIST Images. 
        '''
        if not transformations:
            transformations = transforms.Compose(
                                [
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

        data = datasets.MNIST('../data', train=train, download=True, trasnform=transformations)

        return data

    def build_loader(self, dataset, batch_size=64, **kwargs):
        '''
        Create the Train Loader for the given dataset, and batch_size
        Args:
            dataset: The MNIST Train Data Set
            batch_size: The batch size.
        '''
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
        return loader

    
