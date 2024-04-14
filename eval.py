import os
import argparse
import logging

import torch
import torch.distributed as dist
from torchvision.utils import save_image

from sde import Follmer
from misc import add_dict_to_argparser, create_network, create_karras_grid, create_dataset, create_dataloader
from defaults import data_defaults, model_defaults, eval_defaults
from dist_util import setup_dist, get_device

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.
torch._logging.set_logs(all=logging.ERROR)

def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(
        data_name="mnist",
        start_seed=0,
        end_seed=9999,
        mode="denoiser",
        workdir="tmp/",
        checkpoint="",
    )
    defaults.update(data_defaults(defaults["data_name"]))
    defaults.update(model_defaults(defaults["data_name"]))
    defaults.update(eval_defaults(defaults["data_name"]))
    add_dict_to_argparser(parser, defaults)
    return parser

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

@torch.no_grad()  
def ode_sampler(sde, model, latents, x_mean=None, label=None, solver='euler', rng=None, num_steps=18, rho=7, t_min=1e-4, t_max=0.9999, churn=False, s_min=0.05, s_max=0.6, gamma=0.4):
    """Enhanced ODE sampler, solves from latents at 1 to 0.
    (1, t_max) is solved with x_mean(exact velocity at 1 is data mean, we use empirical mean here).
    (t_max, t_min) is solved using `solver`.
    Inject noises over interval (s_min, s_max), with noise level (1+gamma)*t if churn=True.
    (t_min, 0) is solved with one euler step if using heun solver as it faces singularity at zero.
    (t_max, t_min) is discretized by Karras grid."""
    bsz = latents.shape[0]
    device = latents.device
    xt = latents - (1-t_max) * x_mean if x_mean is not None else latents
    step_fn = sde._get_step_fn(solver)
    if num_steps == 1:
        t_cur = torch.ones([bsz, ], device=device)*t_max
        dt = t_min-t_max
        xt = sde.deis_step(model, xt, t_cur, label, dt) 
    else:
        grid = create_karras_grid(t_min, t_max, num_steps, rho).to(device)
        for i in range(num_steps-1):
            t_cur = grid[i]
            if churn and (s_min <= t_cur <= s_max):
                t_churn = t_cur*(1+gamma)
                xt = sde.churn_noise(xt, t_cur, t_churn, rng)
                t_cur = t_churn
            dt = grid[i+1] - t_cur
            t_cur = torch.ones([bsz, ], device=device)*t_cur
            xt = step_fn(model, xt, t_cur, label, dt)
        t_cur = torch.ones([bsz, ], device=device)*t_min
        dt = -t_min
        xt = sde.deis_step(model, xt, t_cur, label, dt) if solver.lower() == 'deis' else sde.euler_step(model, xt, t_cur, label, dt)
    return xt

@torch.no_grad()  
def characteristic_sampler(sde, model, latents, x_mean, label=None, rng=None, num_steps=18, rho=7, t_min=1e-4, t_max=0.9999, churn=False, s_min=0.05, s_max=0.6, gamma=0.4):
    """Enhanced ODE sampler, solves from latents at t_max to t_min.
    (t_max, t_min) is discretized by Karras grid.
    """
    bsz = latents.shape[0]
    device = latents.device
    xt = latents - (1-t_max) * x_mean
    if num_steps == 1:
        t_cur = torch.ones(bsz, device=device)*t_max
        t_next = torch.zeros(bsz, device=device)
        _, xt = sde.get_denoiser_and_traj(model, xt, t_cur, t_next, label)
    else:
        grid = create_karras_grid(t_min, t_max, num_steps, rho).to(device)
        for i in range(num_steps-1):
            t_cur = grid[i]
            if churn and (s_min <= t_cur <= s_max):
                t_churn = t_cur*(1+gamma)
                xt = sde.churn_noise(xt, t_cur, t_churn, rng)
                t_cur = t_churn
            t_cur = torch.ones(bsz, device=device)*t_cur
            t_next = torch.ones(bsz, device=device)*grid[i+1]
            _, xt = sde.get_denoiser_and_traj(model, xt, t_cur, t_next, label)
        t_cur = torch.ones(bsz, device=device)*t_min
        t_next = torch.zeros(bsz, device=device)
        _, xt = sde.get_denoiser_and_traj(model, xt, t_cur, t_next, label)
    return xt

@torch.no_grad()
def main(args):
    setup_dist()
    os.makedirs(args.workdir, exist_ok=True)
    device = get_device()
    seeds = range(args.start_seed, args.end_seed+1)
    num_batches = ((len(seeds) - 1) // (args.eval_bsz * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    model = create_network(args, traj_mode=True if args.mode.lower() == 'characteristic' else False).eval().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['ema'])
    sde = Follmer(args)
    x_mean = _compute_x_mean(args).to(device)

    for batch_seeds in rank_batches:
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        rng = StackedRandomGenerator(device, batch_seeds)
        latents = rng.randn([batch_size, args.in_channels, args.img_resolution, args.img_resolution], device=device)
        labels = None
        if args.label_dim:
            labels = torch.eye(args.label_dim, device=device)[rng.randint(args.label_dim, size=[batch_size], device=device)]
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
        for seed, image in zip(batch_seeds, images):
            save_image(image, f'{args.workdir}/{seed:06d}.png')

def _compute_x_mean(args):
    dataset = create_dataset(args)
    # compute target mean
    loader = create_dataloader(dataset, args.bsz, args.num_workers, infinite=False)
    img_shape = [args.in_channels, args.img_resolution, args.img_resolution]
    x_mean = torch.zeros(img_shape)
    n_batch = 0
    for batch, _ in loader:
        x_mean += batch.mean(dim=0)
        n_batch += 1
    x_mean /= n_batch
    x_mean = x_mean*2.-1 if args.data_name.lower() != 'mnist' else x_mean
    return x_mean  

if __name__ == '__main__':
    args = create_parser().parse_args()
    main(args)