import time
import torch
import numpy as np
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from scipy import io as sio

from config import path_args, solver_args, denoiser_args, device
from data import load_data_simu
from solver import get_solver
from denoisers import get_denoiser

def main():

    print(path_args)
    print(solver_args)
    print(denoiser_args)
    print(device)

    mea, Phi, gt, x0 = load_data_simu(path_args, solver_args.step, device)
    denoiser = get_denoiser(denoiser_args)
    solver = get_solver(args=solver_args, denoiser=denoiser, gt=gt)

    t1 = time.time()
    x = solver(x0, mea, Phi)
    t2 = time.time()
    print("------Done------")

    sio.savemat("result/kaist_scene09.mat", {"img":x.permute(1,2,0).cpu().numpy()})

    print(f"{solver_args.name.upper()}-{denoiser_args.name.upper()}:")
    print(f"psnr: {psnr(x, gt, data_range=1.0).item():.4f}")
    print(f"ssim: {ssim(x.unsqueeze(0), gt.unsqueeze(0), data_range=1.0).item():.4f}")
    print(f"time: {(t2-t1):.4f} s")

if __name__ == "__main__":
    main()