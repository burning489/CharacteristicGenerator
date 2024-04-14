import argparse
import logging
import math

import torch
from torchvision.utils import save_image

from sde import Follmer
from misc import add_dict_to_argparser, create_network
from defaults import data_defaults, model_defaults, eval_defaults
from eval import ode_sampler, characteristic_sampler, _compute_x_mean

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.
torch._logging.set_logs(all=logging.ERROR)

def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(
        device="cuda:0",
        seed=42,
        mode="denoiser",
        nsample=64,
        data_name="cifar",
        checkpoint="",
    )
    defaults.update(data_defaults(defaults["data_name"]))
    defaults.update(model_defaults(defaults["data_name"]))
    defaults.update(eval_defaults(defaults["data_name"]))
    add_dict_to_argparser(parser, defaults)
    return parser

@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = create_network(args, traj_mode=True if args.mode.lower() == 'characteristic' else False).eval().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['ema'])
    sde = Follmer(args)
    x_mean = _compute_x_mean(args).to(device)
    rng=None
    latents = torch.randn([args.nsample, args.in_channels, args.img_resolution, args.img_resolution], device=device)
    labels = None
    if args.label_dim:
        labels = torch.eye(args.label_dim, device=device)[torch.randint(args.label_dim, size=[args.nsample], device=device)]
    if args.class_idx is not None:
        labels[:, :] = 0
        labels[:, args.class_idx] = 1
    if args.mode == 'denoiser':
        images = ode_sampler(sde, model, latents, x_mean, labels, args.ode_solver, rng, args.num_steps, args.rho, args.eval_eps0, 1-args.eval_eps1, args.churn, args.s_min, args.s_max, args.gamma)
    elif args.mode == 'characteristic':
        images = characteristic_sampler(sde, model, latents, x_mean, labels, rng, args.num_steps, args.rho, args.eval_eps0, 1-args.eval_eps1, args.churn, args.s_min, args.s_max, args.gamma)
    else:
        raise ValueError(f"unsupported eval mode {args.mode}")
    images = (images+1.)/2 if args.data_name.lower() != 'mnist' else images
    save_image(images, f'./asset/{args.data_name}_{args.mode}_nfe{args.num_steps}.png', nrow=int(math.sqrt(args.nsample)))


if __name__ == '__main__':
    args = create_parser().parse_args()
    main(args)