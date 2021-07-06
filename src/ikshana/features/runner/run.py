import torch
from .test import Test
from .train import Train


class Run:

    def __init__(self, model, train_loader, test_loader, epochs, device,
                    optimizer, scheduler=None):
        '''
        '''
        self.epochs = epochs
        self.scheduler = scheduler

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