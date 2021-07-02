import torch
import typing
from torch.utils.data import Dataset

__all__ = ["FindStat"]

class UnNormalize:
    '''
    A class to Un-normalize a image when given the actual (initial)
    mean and standard deviation

    .. math::
        y = tensor*std + mean
    '''
    def __init__(self, mean: typing.Tuple[float, ...], std: typing.Tuple[float, ...]):
        '''
        Args:
            mean: Tuple of Means of Each Channel
            std: Tuple of Standard Deviations of Each Channel
        '''
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        res = torch.tensor(())
        for t, m, s in zip(tensor, self.mean, self.std):
            chan = t.mul(s).add(m)
            res = torch.cat((res,chan.unsqueeze(0)), dim=0)
            # The normalize code -> t.sub_(m).div_(s)
        return res

class FindStat:
    '''
    A class to Find the mean and standard deviation of the dataset prvoided
    '''
    def __init__(self, dataset:Dataset):
        '''
        Args:
            dataset: Pytorch Dataset.
        '''
        self.dataset = dataset
        self.mean = 0
        self.std = 0

    def calculate(self):
        '''
        Iterates through the entire dataset and find mean and standard deviation
        for each channel of the images
        
        Return:
            mean: Tuple of Means of all Channels of Image
            std: Tuple of Standard Deviation of all Channels of Image
        '''
        for img, _ in self.dataset:
            self.mean += torch.mean(img, dim=(1,2)) # Findig Mean Along each Channel
            self.std += torch.std(img, dim=(1,2)) # Findig Stadnard Deviation Along each Channel
        
        return self.mean/len(self.dataset), self.std/len(self.dataset)