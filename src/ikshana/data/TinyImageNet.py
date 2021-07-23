from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A

from .transformations.album_pipeline import album_transformation_support

import os

__all__ = ['TinyImageNet']


class _TinyTestData(Dataset):
    def __init__(self, root, label_map, transform=None):
        self.root = root
        self.transform = transform
        self.label_map = label_map
        self.files = os.listdir(f'{root}/images')
        with open(f'{root}/val_annotations.txt', 'r') as f:
            self.labels = {line.split()[0]:line.split()[1] for line in f.readlines()}


    def __getitem__(self, index):
        img_name = self.files[index]
        img = Image.open(f'{self.root}/images/{img_name}')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[img_name]
        label = self.label_map[label]
        
        return img, label


    def __len__(self) -> int:
        return len(self.labels)


class TinyImageNet:
    
    def __init__(self, train_root, test_root, train_transform=None, 
                    test_transform=None):
        """
        Tiny ImageNet dataset
        :train_root:
        :test_root:
        :train_transform:
        :test_transform:
        """
        if isinstance(train_transform, A.core.composition.Compose):
            train_transform = album_transformation_support(train_transform)
        if isinstance(test_transform, A.core.composition.Compose):
            test_transform = album_transformation_support(test_transform)

        self._train_data = ImageFolder(train_root,
                        transform=train_transform)
        self.class_int_map = self._train_data.find_classes(train_root)
        self._test_data = _TinyTestData(test_root, self.class_int_map[1], transform=test_transform)

    def build_data(self, train=True):
        if train:
            return self._train_data
        else:
            return self._test_data
    
    def build_loader(self, data, batch_size, **kwargs):
        return DataLoader(dataset=data, batch_size=batch_size, shuffle=True, **kwargs)