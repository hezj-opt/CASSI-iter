import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('..')
from utils import At, A, shift, shift_back
from tqdm import tqdm
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr


def get_solver(args, denoiser, gt=None):
    if args.name.lower() == "gap":
        return GAP(args, denoiser, gt)
    elif args.name.lower() == "admm":
        return ADMM(args, denoiser, gt)
    else:
        ValueError(f"{args.name} not implemented yet")

class GAP(nn.Module):
    """GAP solver"""
    def __init__(self, args, denoiser, gt=None) -> None:
        super().__init__()
        self.step = args.step
        self.iter_num = args.iter_num
        self.accelerate = args.accelerate
        self.denoiser = denoiser
        self.denoiser_name = denoiser._get_name()
        self.gt = gt if gt is not None else None

    def iter(self, v, y, Phi, denoiser):
        """iter one step"""
        PhiPhi_T = torch.sum(Phi**2, 0)
        PhiPhi_T[PhiPhi_T==0] = 1
        x1 =  v + At(Phi, y - A(Phi, v)) / PhiPhi_T
        x1 = denoiser(shift_back(x1, self.step))
        v1 = shift(x1, self.step)

        return x1, v1
    
    def iter_acc(self, v, y0, y, Phi, denoiser):
        """iter one step"""
        PhiPhi_T = torch.sum(Phi**2, 0)
        PhiPhi_T[PhiPhi_T==0] = 1
        x1 =  v + At(Phi, y0 - A(Phi, v)) / PhiPhi_T
        y1 = y0 + (y - A(Phi, v))
        x1 = denoiser(shift_back(x1, self.step))
        v1 = shift(x1, self.step)

        return x1, v1, y1
    
    def solve(self, x0, y, Phi):
        """solving inverse problem"""

        # initialization
        x = x0 if x0 is not None else At(Phi, y)
        v = x
        y0 = y if self.accelerate is True else None

        # solving
        if self.accelerate is True:
            for i in tqdm(range(self.iter_num)):
                x, v, y0 = self.iter_acc(v, y0, y, Phi, self.denoiser)
                print(psnr(x, self.gt, data_range=1.0))
        else:
            for i in tqdm(range(self.iter_num)):
                x, v = self.iter(v, y, Phi, self.denoiser)
                print(psnr(x, self.gt, data_range=1.0))

        return x
    
    def forward(self, x0, y, Phi):
        return self.solve(x0, y, Phi)


class ADMM(nn.Module):
    """ADMM solver"""
    def __init__(self, args, denoiser, gt=None) -> None:
        super().__init__()
        self.step = args.step
        self.iter_num = args.iter_num
        self.rho = args.rho
        self.denoiser = denoiser
        self.denoiser_name = denoiser._get_name()
        self.gt = gt if gt is not None else None

    def iter(self, v, u, y, Phi, denoiser):
        """iter one step"""
        PhiPhi_T = torch.sum(Phi**2, 0)
        x1 = v - u / self.rho + At(Phi, (y-A(Phi, v-u/self.rho))/(PhiPhi_T+self.rho))
        if "dip" in self.denoiser_name.lower():
            x = denoiser(shift_back(x1 + u / self.rho, self.step), Phi, y)
        else:
            x = denoiser(shift_back(x1 + u / self.rho, self.step))
        v1 = shift(x, self.step)
        u1 = u + self.rho * (x1 - v1)

        return x, v1, u1
    
    def solve(self, x0, y, Phi):
        """solving inverse problem"""

        # initialization
        x = x0 if x0 is not None else At(Phi, y)
        v = x
        u = 0

        # solving
        for i in tqdm(range(self.iter_num)):
            x, v, u = self.iter(v, u, y, Phi, self.denoiser)
            if self.gt is not None:
                print(psnr(x, self.gt, data_range=1.0))

        return shift_back(v, self.step)
    
    def forward(self, x0, y, Phi):
        return self.solve(x0, y, Phi)