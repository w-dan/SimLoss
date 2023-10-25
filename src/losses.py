import torch
from torchmetrics import JaccardIndex

class CustomJaccardLoss(torch.nn.Module):
    def __init__(self):
        """
        Initializes the CustomJaccardLoss module.

        This loss function computes the Jaccard loss between the predicted output and the target, while excluding
        elements with 'nan' values.

        Methods:
            forward(output, target):
                Calculate the Jaccard loss for non-"nan" values between the output and target tensors.

        """
        super(CustomJaccardLoss, self).__init__()

    def forward(self, output, target):
        """
        Forward pass for the CustomJaccardLoss module.

        Args:
            output (torch.Tensor): The model's predicted output.
            target (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The Jaccard loss computed for the non-'nan' values in the output and target tensors.
        """
        # Find the indices where there are 'nan' values
        nan_indices = torch.isnan(output) | torch.isnan(target)
        
        # Calculate the Jaccard loss only for non-'nan' values
        loss = JaccardIndex(output[~nan_indices], 1 - target[~nan_indices])

        return loss

# Función de pérdida y optimizador
# Función de pérdida personalizada para manejar valores "nan"
class CustomMSELoss(torch.nn.Module):
    """
    CustomMSELoss is a PyTorch custom loss module that calculates the Mean Squared Error (MSE) loss for non-"nan" values
    between the output and target tensors. It masks out "nan" values to ensure a meaningful loss calculation.

    Methods:
        forward(output, target):
            Calculate the MSE loss for non-"nan" values between the output and target tensors.
    """

    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, output, target):
        """
        Calculate the MSE loss for non-"nan" values between the output and target tensors.

        Args:
            output (torch.Tensor): The output tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            loss (torch.Tensor): The MSE loss for non-"nan" values between the output and target tensors.
        """
        # Find indices where there are "nan" values
        nan_indices = torch.isnan(output) | torch.isnan(target)
        
        # Calculate the MSE loss only for values that are not "nan"
        loss = torch.mean((output[~nan_indices] - target[~nan_indices]) ** 2)

        return loss
