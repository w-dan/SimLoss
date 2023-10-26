import torch
from torch import Tensor
from torchmetrics import JaccardIndex

class CustomJaccardLoss(torch.nn.Module):
    def __init__(self, device='cuda:0'):
        """
        Initializes the CustomJaccardLoss module.

        This loss function computes the Jaccard loss between the predicted output and the target, while excluding
        elements with 'nan' values.

        Methods:
            forward(output, target):
                Calculate the Jaccard loss for non-"nan" values between the output and target torch.Tensors.

        """
        super(CustomJaccardLoss, self).__init__()
        self.device = device
    
    def set_jaccard_classes(self, num_classes):
        self.jaccard = JaccardIndex(task='multiclass', num_classes=num_classes).to(self.device)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass for the CustomJaccardLoss module.

        Args:
            output (Tensor): The model's predicted output.
            target (Tensor): The target values.

        Returns:
            Tensor: The Jaccard loss computed for the non-'nan' values in the output and target torch.Tensors.
        """
        output = output.to(self.device)
        target = target.to(self.device)

        # Find the indices where there are 'nan' values
        nan_indices = torch.isnan(output) | torch.isnan(target)

        # Calculate the Jaccard loss only for non-'nan' values
        loss = self.jaccard(output[~nan_indices], 1 - target[~nan_indices])

        return loss


# Loss function and optimizer
# Custom loss function to handle "nan" values
class CustomMSELoss(torch.nn.Module):
    """
    CustomMSELoss is a PyTorch custom loss module that calculates the Mean Squared Error (MSE) loss for non-"nan" values
    between the output and target torch.Tensors. It masks out "nan" values to ensure a meaningful loss calculation.

    Methods:
        forward(output, target):
            Calculate the MSE loss for non-"nan" values between the output and target torch.Tensors.
    """

    def __init__(self, device='cuda:0'):
        super(CustomMSELoss, self).__init__()
        self.device = device

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the MSE loss for non-"nan" values between the output and target torch.Tensors.

        Args:
            output (Tensor): The output Tensor.
            target (Tensor): The target Tensor.

        Returns:
            loss (Tensor): The MSE loss for non-"nan" values between the output and target torch.Tensors.
        """
        output = output.to(self.device)
        target = target.to(self.device)

        # Find indices where there are "nan" values
        nan_indices = torch.isnan(output) | torch.isnan(target)

        # Calculate the MSE loss only for values that are not "nan"
        loss = torch.mean((output[~nan_indices] - target[~nan_indices]) ** 2)

        return loss
