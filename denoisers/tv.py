import numpy as np
import torch
import torch.nn as nn

class TV_denoiser(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.alpha = args.alpha
        self.iter_num = args.iter_num
        self.lam = args.lam

    def forward(self, x):
        #initialization
        C, H, W = x.shape
        device = x.device
        zh = torch.zeros(C, H-1, W).to(device)
        zv = torch.zeros(C, H, W-1).to(device)

        for i in range(self.iter_num):
            xh = x - (-torch.diff(zh, dim=1, prepend=torch.zeros(C, 1, W).to(device), append=torch.zeros(C, 1, W).to(device)))
            xv = x - (-torch.diff(zv, dim=2, prepend=torch.zeros(C, H, 1).to(device), append=torch.zeros(C, H, 1).to(device)))
            x_temp = (xh+xv) / 2
            zh = clip(zh + 1/self.alpha*torch.diff(x_temp, dim=1), self.lam/2)
            zv = clip(zv + 1/self.alpha*torch.diff(x_temp, dim=2), self.lam/2)

        return x_temp

def clip(x, T):
    return torch.clamp(x, min=-T, max=T)