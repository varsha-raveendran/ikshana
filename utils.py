import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

def plot_loss_acc(train_loss, train_acc, test_loss, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_loss)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_loss)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def plot_combined(train_acc, test_acc):
    plt.axes(xlabel= 'epochs', ylabel= 'Accuracy')
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('Test vs Train Accuracy')
    plt.legend(['Train', 'Test'])

def data_stats(data):
    exp = data.data
    exp = data.transform(exp.numpy())

    print('Train Statistics')
    print(' - Numpy Shape:', data.data.cpu().numpy().shape)
    print(' - Tensor Shape:', data.data.size())
    print(' - min:', torch.min(exp))
    print(' - max:', torch.max(exp))
    print(' - mean:', torch.mean(exp))
    print(' - std:', torch.std(exp))
    print(' - var:', torch.var(exp))

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def plot_data_grid(train_loader, mean:list, std:list, class_list, ncol=6, nrow=6):

    images, labels = next(iter(train_loader))
    unNorm= UnNormalize(mean, std)

    fig,a =  plt.subplots(nrow,ncol,figsize=(10,10))
    for num in range(nrow*ncol):
        i = num//nrow
        j = num%ncol
        if images[num].size(0) == 1: #Single Channel
            img = unNorm(images[num])
            img = torch.squeeze(img,0)
            cmap='gray'
        else: # Multi-Channel
            img = unNorm(images[num])
            img = np.transpose(img, (1,2,0))
            cmap=None
        a[i][j].imshow(img, cmap)
        a[i][j].set_title(f'GT:{class_list[labels[num]]}')
        a[i,j].axis('off')
    fig.tight_layout()

def plot_results(pred_img, pred_lab, gt_lab, mean:list, std:list, class_list, ncol=6, nrow=6):
    
    unNorm= UnNormalize(mean, std)
    pred_img = unNorm(pred_img)

    fig,a =  plt.subplots(nrow,ncol,figsize=(10,10))
    for num in range(nrow*ncol):
        i = num//nrow
        j = num%ncol
        if pred_img[num].size(0) == 1: #Single Channel
            img = unNorm(pred_img[num])
            img = torch.squeeze(img,0)
            cmap='gray'
        else: # Multi-Channel
            img = unNorm(pred_img[num])
            img = np.transpose(img, (1,2,0))
            cmap=None
        a[i][j].imshow(img, cmap)
        a[i][j].set_title(f'GT:{class_list[gt_lab[num]]}')
        a[i][j].text(0.5,-0.2, f'Predicted: {class_list[pred_lab[num].item()]}', size=12, ha="center", transform=a[i][j].transAxes)
        a[i][j].axis('off')

    fig.tight_layout()