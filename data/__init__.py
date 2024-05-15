import numpy as np
import scipy.io as sio
import torch
from scipy.interpolate import interp1d

import sys
sys.path.append('..')
from utils import A, shift

def load_data_simu(path_args, step, device):
    gt_path, mask_path, x0_path = path_args.gt_path, path_args.mask_path, path_args.x0_path
    gt = torch.from_numpy(sio.loadmat(gt_path)['img']).permute(2, 0, 1)
    ch , _, _ = gt.shape
    mask = torch.from_numpy(sio.loadmat(mask_path)['mask'])
    x0 = shift(torch.from_numpy(sio.loadmat(x0_path)['img']).permute(2, 0, 1), step) if x0_path is not None else None

    Phi = shift(mask.repeat(ch, 1, 1), step)
    mea = A(Phi, shift(gt, step))

    mea, Phi, gt = mea.to(device), Phi.to(device), gt.to(device)
    x0 = x0.to(device) if x0 is not None else None

    return mea, Phi, gt, x0