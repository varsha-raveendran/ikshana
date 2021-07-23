import torch
import torch.nn as nn

from .test import Test
from .train import Train

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
            sched_name = type(self.scheduler).__name__ 
            sched = self.scheduler if sched_name == 'OneCycleLR' else None
            train_loss, train_acc = self.train.fit(sched)
            test_loss, test_acc = self.test.predict()
            if self.scheduler and sched_name != 'OneCycleLR':
                if sched_name == 'ReduceLROnPlateau':
                    metrics = test_loss if kwargs['metrics'].lower() == 'loss' else test_acc
                    self.scheduler.step(metrics)
                else:
                    self.scheduler.step()

            print('TRAIN set: Average loss: {:.4f}, Train Accuracy: {:.2f}%'.format(train_loss,train_acc), end= ' | ')
            print('TEST set: Average loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss,test_acc))
            print('~'*60)