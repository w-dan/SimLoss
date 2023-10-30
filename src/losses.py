import torch
from torch import Tensor

class JaccardSimilarity(torch.nn.Module):
    def __init__(self, device='cpu'):
        """
        Initializes the JaccardSimilarity module.

        Args:
            device (str): Device to use, e.g., 'cpu' or 'cuda:0'. Default is 'cuda:0'.

        Methods:
            forward(output, target):
                Calculate the Jaccard similarity between the predicted output and the target torch.Tensors.

        """
        super(JaccardSimilarity, self).__init__()
        self.device = device

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass for the JaccardSimilarity module.

        Args:
            output (Tensor): The model's predicted output.
            target (Tensor): The target values.

        Returns:
            Tensor: The Jaccard similarity computed for the output and target torch.Tensors.
        """
        # Ensure both tensors are on the same device
        output = output.to(self.device)
        target = target.to(self.device)

        intersection = (output * target).sum()
        union = (output + target).sum() - intersection

        jaccard = intersection / union

        return jaccard


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

    def __init__(self, device='cpu'):
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
        valid_indices = torch.nonzero(~nan_indices, as_tuple=True)
        valid_indices = valid_indices[0]

        # Calculate the MSE loss only for values that are not "nan"
        loss = torch.mean((output[valid_indices] - target[valid_indices]) ** 2)

        return loss
