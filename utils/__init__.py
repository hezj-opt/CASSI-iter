import numpy as np
import torch

def At(Phi: torch.tensor, y: torch.tensor) -> torch.tensor:
    """ Computing Phi^T*y """
    # Phi: (C, H, W)
    # y: (H, W)
    return Phi * y

def A(Phi, x):
    """ Computing y = sum(Phi*x) """
    return torch.sum(Phi*x, dim=0)

def shift(x, step):
    C, H, W = x.shape
    x_shift = torch.zeros(C, H, W + (C - 1) * step).to(x.device)
    for i in range(C):
        x_shift[i, :, i*step:i*step+W] = x[i, :, :]

    return x_shift

def shift_back(x_shift, step):
    C, H, W = x_shift.shape
    x = torch.zeros(C, H, W - (C - 1) * step).to(x_shift.device)
    for i in range(C):
        x[i, :, :] = x_shift[i, :, i*step:i*step+W-(C - 1)*step]

    return x