import yaml
from argparse import Namespace


with open('./config/config.yaml', 'r', encoding='utf-8') as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)

path_args = Namespace(**args["path"])
solver_args = Namespace(**args["solver"])
denoiser_args = Namespace(**args["denoiser"])
device = args["device"]

with open('./config/config_solver.yaml', 'r', encoding='utf-8') as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)

if solver_args.name.lower() == "gap":
    solver_args.__dict__.update(args["gap_args"])
elif solver_args.name.lower() == "admm":
    solver_args.__dict__.update(args["admm_args"])

with open('./config/config_denoiser.yaml', 'r', encoding='utf-8') as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)

if denoiser_args.name.lower() == "tv":
    denoiser_args.__dict__.update(args["tv_args"])
elif denoiser_args.name.lower() == "hsicnn":
    denoiser_args.__dict__.update(args["hsicnn_args"])
elif denoiser_args.name.lower() == "dip-hsi":
    denoiser_args.__dict__.update(args["dip_hsi_args"])
    denoiser_args.step = solver_args.step
