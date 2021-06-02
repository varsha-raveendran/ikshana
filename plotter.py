import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import seaborn as sn
import pandas as pd

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


def plot_combined(train_acc, test_acc, x_label='epochs', y_label='Accuracy'):
    plt.axes(xlabel= x_label, ylabel= y_label)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('Test vs Train')
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

class Results():

    def __init__(self, model, loader, device, mean, std, class_list):
        self.model = model
        self.loader = loader
        self.device = device
        self.mean = mean
        self.std = std
        self.class_list = class_list
        self.results = self._forwad_pass()

    def _forwad_pass(self):
        nb_classes = len(self.class_list)
        confusion_matrix = torch.zeros(nb_classes, nb_classes, dtype=torch.long)
        pred_imgs, pred_lab, gt_lab = None, None, None
        incorrect_images, total_pred, total_gt_lab = None, None, None

        self.model.eval()
        with torch.no_grad():
            for batch in self.loader:
                images, labels = batch
                output = self.model(images.to(self.device))
                predicted = output.argmax(dim=1).cpu()

                # Confusion Matrix
                for l,p in zip(labels, predicted):
                    confusion_matrix[l, p] += 1

                # For Plot Results of one Batch
                if pred_imgs is None:
                    pred_imgs = images
                    pred_lab = predicted.cpu()
                    gt_lab = labels

                # Geeting the ids of "In"correct Classigied Images
                idx = ~predicted.eq(labels)
                if idx.sum().item() > 0: # If there are incorrect images
                    if incorrect_images is None:
                        incorrect_images = images[idx]
                        total_pred = predicted[idx]
                        total_gt_lab = labels[[idx]]
                    else:
                        incorrect_images = torch.cat((incorrect_images, images[idx]), dim=0)
                        total_pred = torch.cat((total_pred, predicted[idx].cpu()))
                        total_gt_lab = torch.cat((total_gt_lab, labels[[idx]]))

        cls_acc = (confusion_matrix.diag()/confusion_matrix.sum(1))*100

        return {'confusion':confusion_matrix, 'class_acc': cls_acc, 
                'incorrect_images':incorrect_images, 'total_pred': total_pred, 'total_gt':total_gt_lab,
                'pred_imgs':pred_imgs, 'pred_lab':pred_lab, 'gt_lab':gt_lab}

    def plot_batch(self, ncol=6, nrow=6):
        
        unNorm= UnNormalize(self.mean, self.std)

        fig,a =  plt.subplots(nrow,ncol,figsize=(10,10))
        for num in range(nrow*ncol):
            i = num//nrow
            j = num%ncol
            if self.results['pred_imgs'][num].size(0) == 1: #Single Channel
                img = unNorm(self.results['pred_imgs'][num])
                img = torch.squeeze(img,0)
                cmap='gray'
            else: # Multi-Channel
                img = unNorm(self.results['pred_imgs'][num])
                img = np.transpose(img, (1,2,0))
                cmap=None
            a[i][j].imshow(img, cmap)
            a[i][j].set_title(f"GT:{self.class_list[self.results['gt_lab'][num]]}")
            a[i][j].text(0.5,-0.2, f"Predicted: {self.class_list[self.results['pred_lab'][num].item()]}", size=12, ha="center", transform=a[i][j].transAxes)
            a[i][j].axis('off')

        fig.tight_layout()

    def plot_incorrect(self):

        unNorm= UnNormalize(self.mean, self.std)
        ncol = int(np.sqrt(self.results['incorrect_images'].size(0))) #Finding Total Number of Images Sqrt
        ncol = min(ncol, 6)
        nrow = ncol

        fig,a =  plt.subplots(nrow,ncol,figsize=(10,10))
        for num in range(nrow*ncol):
            i = num//nrow
            j = num%ncol
            if self.results['incorrect_images'][num].size(0) == 1: #Single Channel
                img = unNorm(self.results['incorrect_images'][num])
                img = torch.squeeze(img,0)
                cmap='gray'
            else: # Multi-Channel
                img = unNorm(self.results['incorrect_images'][num])
                img = np.transpose(img, (1,2,0))
                cmap=None
            a[i][j].imshow(img, cmap)
            a[i][j].set_title(f"GT:{self.class_list[self.results['total_gt'][num]]}")
            a[i][j].text(0.5,-0.2, f"Predicted: {self.class_list[self.results['total_pred'][num].item()]}", size=12, ha="center", transform=a[i][j].transAxes)
            a[i][j].axis('off')

        fig.tight_layout()

    def class_accuracy(self, confusion_heatmap=True, top_n=10):

        if confusion_heatmap:
            plt.figure(figsize=(8,8))
            sn.heatmap(self.results['confusion'].numpy(), 
                        xticklabels=self.class_list, yticklabels=self.class_list,
                        annot=True,cmap='Blues', fmt='d')
            plt.xlabel("Predicted") 
            plt.ylabel("Labels") 
            plt.show()
        
        sorted_class_acc = torch.sort(self.results['class_acc'])
        
        print(f'Accuracies of Top {top_n} Classes in Decreasing Order')
        for i in sorted_class_acc.indices[:top_n]:
            print(f"Accuracy of class {self.class_list[i]} is {self.results['class_acc'][i]:.2f}")