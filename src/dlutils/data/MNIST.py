import torch
from torchvision import datasets, transforms

class Data:
    
    def __init__(self, mean=(0.1307,), std=(0.3081,)):
        self.mean = mean
        self.std = std

    def build_train(self, transformations=None):

        if not transformations:
            transformations = transforms.Compose(
                                [
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

        train_data = datasets.MNIST('../data', train=True, download=True, trasnform=transformations)

        