import torch.nn.functional as F
import torch

class Test():
    """
    Performance of the trained model is evaluated on the DataLoader provided by
    the user.

    Parameters:
    -----------
    model: Model to be used to test the model.
    device: Type of device (GPU/CPU) to be used while testing model.
    test_loader: Defining which DataLoader to be used for testing.

    Returns:
    --------
    test_loss: The loss observed while testing the model.
    accuracy: Accuracy of the model over the testing data.
    """
    def __init__(self, model, device, test_loader):

        self.model = model
        self.test_loader = test_loader
        self.device = device
        
        self.loss = []
        self.accuracy = []

    def __call__(self):
        self.model.eval()
        correct_count = 0
        test_loss = 0
        with torch.no_grad():  # Setting the calculations to not involve any kind of gradient calculation
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='mean').item()
                pred = output.argmax(dim=1) # Getting Indices of Class with Max Value for each Image
                correct_count += pred.eq(target).sum().item() # Equating Predicted and Label Tensors at each Index value

        test_loss /= len(self.test_loader)
        accuracy = 100. * correct_count / len(self.test_loader.dataset)

        self.loss.append(test_loss)
        self.accuracy.append(accuracy)

        return test_loss, accuracy