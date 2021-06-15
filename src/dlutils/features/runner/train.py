from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import typing
from torch.utils.data import DataLoader
from torch.optim import Optimizer

class Train():
    """
    Model training is performed considering the multiple inputs like
    model, device on which training has to happen, optimizer to be
    used while training.

    Parameters:
    -----------
        model: Model architecture to be used to train the model.
        device: Type of device (GPU/CPU) to be used while training model.
        train_loader: Defining which DataLoader to be used for training.
        optimizer: Defining which optimizer to be used.

    Returns:
    --------
        train_loss: The loss observed while training the model.
        accuracy: Accuracy of the model over the current training data.
    """
    def __init__(self, model: nn.Module, device, train_loader: DataLoader, optimizer: Optimizer):

        self.model = model
        self.progress_bar = tqdm(train_loader)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device

        self.accuracy = []
        self.loss = []
       

    def fit(self) -> typing.Tuple[float, float]:
        '''
        Runs the Model Inferenceing on the Test Dataloader provided.
        
        Returns:
            train_loss: The Average Loss of the Model for the Dataloader, 
            accuracy: The Accuracy of the Model for the Dataloader (correct_predicted/total_images)
        '''
        self.model.train() 
        correct_count = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()  # Setting optimizer value to zero to avoid accumulation of gradient values
            output = self.model(data)
            batch_loss = F.nll_loss(input=output, target=target, reduction='mean')
            batch_loss.backward()
            self.optimizer.step()

            train_loss += batch_loss.item()
            pred = output.argmax(dim=1) # Getting Indices of Class with Max Value for each Image
            correct_count += pred.eq(target).sum().item() # Equating Predicted and Label Tensors at each Index value

            self.progress_bar.set_description(desc= f'loss={batch_loss.item()} batch_id={batch_idx}')
            
        train_loss /= len(self.train_loader)
        accuracy = 100. * correct_count / len(self.train_loader.dataset)

        self.loss.append(train_loss)
        self.accuracy.append(accuracy)
        
        return train_loss, accuracy