import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sn

from norm import UnNormalize

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

    def plot_incorrect(self, nrow, ncol):
        '''
        Plot those Incorrect classified Images in ncol*nrow matrix if given,
        or else Displays all incorrect classified Images.
        Parameters:
            nrow: The Number of Rows of Images
            ncol: The Number of Coloumns of Images
        '''
        unNorm= UnNormalize(self.mean, self.std)
        ncol_ = int(np.sqrt(self.results['incorrect_images'].size(0))) #Finding Total Number of Images Sqrt
        
        # All Images or Given ncol*nrow number of Images
        ncol = min(ncol_, ncol)
        nrow = min(ncol, nrow)

        fig,a =  plt.subplots(nrow,ncol,figsize=(10,10))
        for num in range(nrow*ncol):
            # num -> 0-(nrow*ncol)
            # i = 0, j=0,1,2,3,...,(ncol-1)
            # i = 1, j=0,1,2,3,...,(ncol-1)
            # .
            # .
            # .
            # i = (nrow-1), j=0,1,2,3,...,(ncol-1)
            i = num//ncol
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
        '''

        '''
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