import torch
import torch.nn.functional as F


@torch.inference_mode
def mean_absolute_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Absolute Error (MAE) between the input and target tensors.

    Parameters:
    y_pred (torch.Tensor): The predicted values tensor.
    y_true (torch.Tensor): The ground truth tensor.

    Returns:
    torch.Tensor: The computed mean absolute error as a scalar tensor.
    """
    return F.l1_loss(input=y_pred, target=y_true)


@torch.inference_mode
def mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Square Error (MSE) between the input and target tensors.

    Parameters:
    y_pred (torch.Tensor): The predicted values tensor.
    y_true (torch.Tensor): The ground truth tensor.

    Returns:
    torch.Tensor: The computed mean absolute error as a scalar tensor.
    """
    return F.mse_loss(input=y_pred.flatten(1), target=y_true.flatten(1))
