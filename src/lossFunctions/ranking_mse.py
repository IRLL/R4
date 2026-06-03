import torch
from fast_soft_sort.pytorch_ops import soft_rank
from torch.nn.functional import mse_loss


def ranking_mse(predictions, gt_returns, regularization_strength=1.0):
    """
    MSE between soft ranks of predictions and ground truth.

    Args:
        predictions (Tensor): predictions
        gt_returns (Tensor): ground truth returns
    """
    predictions = soft_rank(predictions.T, regularization_strength=regularization_strength)
    gt_returns = soft_rank(gt_returns.T, regularization_strength=regularization_strength).detach()
    loss = mse_loss(predictions, gt_returns)
    return loss
