from .tv import TV_denoiser
from .hsi_decnn import hsicnn_denoiser
from .dip_hsi import DIP_HSI

def get_denoiser(args):
    if args.name.lower() == "tv":
        return TV_denoiser(args)
    if args.name.lower() == "hsicnn":
        return hsicnn_denoiser(args)
    if args.name.lower() == "dip-hsi":
        return DIP_HSI(args)
    else:
        ValueError(f"{args.name} not implemented yet")