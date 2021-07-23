from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

__all__ = ['TinyImageNet']


class _TinyTestData(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = os.listdir(root)
        with open(f'{root}/val_annotations.txt', 'r') as f:
            self.labels = {line.split()[0]:line.split()[1] for line in f.readlines()}


    def __getitem__(self, index):
        img_name = self.files[index]
        img = Image.open(f'{self.root}/{img_name}')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[img_name]
        
        return img, label


    def __len__(self) -> int:
        return len(self.labels)


class TinyImageNet:
    
    def __init__(self, train_root, test_root, train_transform=None, 
                    test_transform=None, batch_size=64, **kwargs):
        """
        Tiny ImageNet dataset
        :train_root: 
        :batch_size:
        :num_workers:
        :pin_memory:
        :return:
        """
        self.batch_size = batch_size
        self.kwargs = kwargs
        self._train_data = ImageFolder(train_root,
                        transform=train_transform)
        self._test_data = _TinyTestData(test_root, transform=test_transform)

    @property
    def train_dataset(self):
        return self._train_data
    
    def train_loader(self):
        return DataLoader(self._train_data, self.batch_size, **self.kwargs)

    @property
    def test_dataset(self):
        return self._test_data

    def test_loader(self):
        return DataLoader(self._test_data, self.batch_size, **self.kwargs)