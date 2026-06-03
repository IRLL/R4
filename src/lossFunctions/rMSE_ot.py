import torch
from torch.nn.functional import mse_loss
from lossFunctions.utils import Rank_Sort


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def ranking_mse_ot(predictions, gt_returns, eps=0.1, power=2, nu=1e-5):
    """
    MSE between OT-based soft ranks of predictions and ground truth.

    Args:
        predictions (Tensor): shape (N,) or (N, B)
        gt_returns (Tensor): shape (N,) or (N, B)
    """
    predictions = predictions.float().T
    gt_returns = gt_returns.float().T

    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
        gt_returns = gt_returns.unsqueeze(0)

    _, _ = predictions.shape
    device = predictions.device

    a = torch.ones(predictions.shape[1], device=device) / predictions.shape[1]
    b = torch.ones(predictions.shape[1], device=device) / predictions.shape[1]

    y = torch.linspace(0, 1, predictions.shape[1], device=device)

    losses = []

    for i in range(predictions.shape[0]):
        x_pred = normalize(predictions[i])
        x_gt = gt_returns[i]

        R_pred, _ = Rank_Sort(a, b, x_pred, y, eps, power, nu)

        R_gt = x_gt
        R_pred = R_pred - 1

        loss = mse_loss(R_pred, R_gt)
        losses.append(loss)

    return torch.stack(losses).mean()
