import torch
import torch.nn as nn
from typing_extensions import Literal

from nerfstudio.model_components.losses import MSELoss, L1Loss

SmoothL1Loss = nn.SmoothL1Loss


class FocalLoss(nn.Module):
    def __init__(self, alpha : float = 1.0, type : Literal["mse", "l1", "sl1"] = "mse"):
        super().__init__()
        self.alpha = alpha
        if type == "mse":
            self.criterion = MSELoss(reduction="none")
        elif type == "l1":
            self.criterion = L1Loss(reduction="none")
        elif type =="sl1":
            self.criterion = SmoothL1Loss(reduction="none", beta=0.01)
        else:
            raise ValueError(f"Invalid 'type' value: {type}.")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # (N, 3)
        error = self.criterion(input, target)
        weight = error.clone().detach().mean(dim=-1, keepdim=True) ** self.alpha
        weight = weight / weight.min()
        weight = torch.nan_to_num(weight, nan=0.0)
        weight = torch.clamp(weight, min=1.0, max=10.0)
        return torch.mean(weight * error)


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
