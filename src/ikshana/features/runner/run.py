import torch
import torch.nn as nn
import copy

from .test import Test
from .train import Train
from .lr import lr_finder

class Run:

    def __init__(self, model, train_loader, test_loader, epochs, device,
                    optimizer, scheduler=None):
        '''
        '''
        self.epochs = epochs
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.model = model
        self.device = device

        self.train = Train(model, device, train_loader, optimizer)
        self.test = Test(model, device, test_loader)
        self.metrics = {'train_loss':self.train.loss, 'train_accuracy':self.train.accuracy,
                        'test_loss':self.test.loss, 'test_accuracy':self.test.accuracy}

    def __call__(self, **kwargs):
        '''
        '''
        for epoch in range(1,  self.epochs):
            print(f'Epoch: {epoch}')
            train_loss, train_acc = self.train.fit()
            test_loss, test_acc = self.test.predict()
            if self.scheduler:
                self.scheduler.step(**kwargs)

            print('TRAIN set: Average loss: {:.4f}, Train Accuracy: {:.2f}%'.format(train_loss,train_acc), end= ' | ')
            print('TEST set: Average loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss,test_acc))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    def find_lr(self, epochs = 1, init_value:float = 1e-8, final_value:float = 10, 
            beta:float = 0.98):
        '''
        '''
        model_clone = copy.deepcopy(self.model)
        optimizer_clone = copy.deepcopy(self.optimizer)
        log_lr, loss = lr_finder(self.train_loader, model_clone, optimizer_clone,
                            nn.NLLLoss, self.device, epochs,
                            init_value, final_value, beta)

        # Skipping First 10 and Last 5 Values.
        plt.plot(logs[10:-5],losses[10:-5])
        min_loss_idx = loss.index(min(loss))
        print(f'The minimum loss of {loss[min_loss_idx]} at LR of log_lr{10**log_lr[min_loss_idx]}'
        print('Run <object>.reset to reset the model and optimizer.')

    def 