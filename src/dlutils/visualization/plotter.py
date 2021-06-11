import matplotlib.pyplot as plt
import numpy as np
import torch

from norm import UnNormalize

def plot_loss_acc(train_loss, train_acc, test_loss, test_acc):
    '''
    Plots 4 Grpahs each for Training Loss, Training Accuracy, Test Loss and Test Accuracy
    against epochs.
    Arguments:
        train_loss: The list of training losses end of each epoch.
        train_accuracy: The list of training accuracies of each epoch.
        test_loss: The list of testing losses end of each epoch.
        test_accuracy: The list of testing accuracies of each epoch.
    Returns:
        Plots 4 Grpahs.
        Returns None value.
    '''
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_loss)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_loss)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def plot_combined(list_of_plotters, *, x_label='epochs', y_label='Accuracy', title='Test vs Train', legend = ['Train', "Test"]):
    '''
    Plot multiple list of values, in a single plot.

    Parameters:
        list_of_plotters: A List of Accuracies/Loss which are arrays/lists.
        x_label: The X Axis Label (Optional Keywork Argument)
        y_label: The Y Axis Label (Optional Keywork Argument)
        title: The Title of the Graph (Optional Keywork Argument)
        legend: The Legend to be printed on Chart (Optional Keywork Argument) 
    '''
    plt.axes(xlabel= x_label, ylabel= y_label)
    for plotting_list in list_of_plotters:
        plt.plot(plotting_list)
    plt.title(title)
    plt.legend(legend)


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