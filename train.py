import os
import argparse
import math
from copy import deepcopy
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from misc import add_dict_to_argparser, create_dataset, create_network, create_dataloader
from sde import Follmer
from dist_util import setup_dist, get_device, sync_params
from defaults import data_defaults, model_defaults, train_defaults, characteristic_defaults

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(
        data_name="mnist",
        seed=42,
        mode="characteristic",
        workdir="logs/cifar-characteristic",
    )
    defaults.update(data_defaults(defaults["data_name"]))
    defaults.update(model_defaults(defaults["data_name"]))
    if defaults['mode'] == 'denoiser':
        defaults.update(train_defaults(defaults["data_name"]))
    elif defaults['mode'] == 'characteristic':
        defaults.update(characteristic_defaults(defaults["data_name"]))
    add_dict_to_argparser(parser, defaults)
    return parser


def denoiser_matching(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    setup_dist()
    assert args.global_bsz % (args.bsz * dist.get_world_size()) == 0
    os.makedirs(args.workdir, exist_ok=True)
    device = get_device()

    dataset = create_dataset(args)
    # compute target mean
    loader = create_dataloader(dataset, args.bsz, args.num_workers, infinite=False)
    img_shape = [args.in_channels, args.img_resolution, args.img_resolution]
    x_mean = torch.zeros(img_shape)
    n_batch = 0
    for batch, label in loader:
        x_mean += batch.mean(dim=0)
        n_batch += 1
    x_mean /= n_batch
    x_mean = x_mean*2.-1 if args.data_name.lower() != 'mnist' else x_mean
    x_mean = x_mean.to(device)

    loader = create_dataloader(dataset, args.bsz, args.num_workers)
    sde = Follmer(args)
    model = create_network(args).to(device)
    ddp_model = DDP(
        model, device_ids=[dist.get_rank()], broadcast_buffers=False, bucket_cap_mb=128
    )
    ema = deepcopy(model).eval().requires_grad_(False)
    optim = torch.optim.RAdam(ddp_model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"loading checkpoint {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            optim.load_state_dict(checkpoint["optim"])
            start_step = checkpoint['step']
            del checkpoint

    sync_params(model.parameters())
    sync_params(model.buffers())

    eval_x1 = sde.sampling_prior(shape=[args.nsampling, *img_shape], device=device)
    ode_grid = torch.linspace(1 - args.eval_eps1, args.eval_eps0, args.num_steps, device=device)
    sde_grid = torch.linspace(1 - args.eval_eps1, 0, args.num_steps, device=device)
    eval_label = None
    if args.label_dim:
        eval_label = torch.randint(0, args.label_dim, (args.nsampling,), device=device)
        eval_label = F.one_hot(eval_label, args.label_dim).float()

    logger = SummaryWriter(args.workdir) if dist.get_rank() == 0 else None
    for step in range(start_step+1, args.train_steps+1):
        n_rounds = args.global_bsz // (args.bsz * dist.get_world_size())
        optim.zero_grad()
        for i in range(n_rounds):
            last_round = (i == n_rounds-1)
            batch, label = next(loader)
            batch = batch.to(device)
            label = label.to(device)
            batch = batch*2.-1 if args.data_name.lower() != 'mnist' else batch
            sync_context = nullcontext if last_round else ddp_model.no_sync
            with sync_context():
                loss = sde.compute_dsm_loss(ddp_model, batch, label)
                loss.backward()

        for g in optim.param_groups:
            g["lr"] = args.lr * min(step / args.lr_rampup_steps, 1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optim.step()

        ema_decay = min(args.ema_decay, (1 + step) / (10 + step))
        for p_ema, p_net in zip(ema.parameters(), model.parameters()):
            if p_net.requires_grad:
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_decay))

        if step % args.log_freq == 0 and dist.get_rank() == 0:
            logger.add_scalar("DSM", loss.item(), step)

        if step % args.sampling_freq == 0 and dist.get_rank() == 0:
            with torch.no_grad():
                eval_ode = sde.ode_sampler(ddp_model, eval_x1, x_mean, ode_grid, args.ode_solver, eval_label)
                eval_ode = (eval_ode+1.)/2 if args.data_name.lower() != 'mnist' else eval_ode
                grid = make_grid(eval_ode, int(math.sqrt(args.nsampling)))
                logger.add_image(f"ODE", grid, step)
                eval_sde = sde.sde_sampler(ddp_model, eval_x1, sde_grid, args.sde_solver, eval_label)
                eval_sde = (eval_sde+1.)/2 if args.data_name.lower() != 'mnist' else eval_sde
                grid = make_grid(eval_sde, int(math.sqrt(args.nsampling)))
                logger.add_image(f"SDE", grid, step)
        if (
            step % args.dump_freq == 0
            or step == args.train_steps
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(
                    model=ddp_model.module.state_dict(),
                    ema=ema.state_dict(),
                    optim=optim.state_dict(),
                    step=step,
                ),
                f"{args.workdir}/{args.architecture}-{step}.pth",
            )
    if dist.get_rank() == 0:
        logger.close()


def characteristic_matching(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    setup_dist()
    assert args.global_bsz % (args.bsz * dist.get_world_size()) == 0
    os.makedirs(args.workdir, exist_ok=True)
    device = get_device()

    dataset = create_dataset(args)
    loader = create_dataloader(dataset, args.bsz, args.num_workers)
    sde = Follmer(args)
    teacher_model = create_network(args).eval().to(device)
    teacher_model.load_state_dict(torch.load(args.teacher, device)['ema'])
    model = create_network(args, traj_mode=True).eval().to(device)
    if dist.get_rank() != 0:
        dist.barrier()
    if dist.get_rank() == 0:
        model.load_state_dict(teacher_model.state_dict(), strict=False)
        dist.barrier()
    ddp_model = DDP(
        model, device_ids=[dist.get_rank()], broadcast_buffers=False, bucket_cap_mb=128
    )
    ema = deepcopy(model).eval().to(device).requires_grad_(False)
    optim = torch.optim.RAdam(ddp_model.parameters(), lr=args.lr)
    if args.loss_norm == 'l2':
        feature_extractor = lambda pred, target: (pred-target)**2
    elif args.loss_norm == 'lpips':
        from lpips import LPIPS
        feature_extractor = LPIPS(replace_pooling=True, reduction="none")
        feature_extractor.model.to(device)
    else:
        raise ValueError(f"unsupported loss norm {args.loss_norm}")

    start_step = 0
    sync_params(model.parameters())
    sync_params(model.buffers())

    img_shape = [args.in_channels, args.img_resolution, args.img_resolution]
    eval_x1 = sde.sampling_prior(shape=[args.nsampling, *img_shape], device=device)
    ones = torch.ones([args.nsampling, ], device=device)*(1-args.train_eps1)
    zeros = torch.ones([args.nsampling, ], device=device)*args.train_eps0
    eval_label = None
    if args.label_dim:
        eval_label = torch.randint(0, args.label_dim, (args.nsampling,), device=device)
        eval_label = F.one_hot(eval_label, args.label_dim).float()

    logger = SummaryWriter(args.workdir) if dist.get_rank() == 0 else None
    for step in range(start_step+1, args.train_steps+1):
        n_rounds = args.global_bsz // (args.bsz * dist.get_world_size())
        optim.zero_grad()
        for i in range(n_rounds):
            last_round = (i == n_rounds-1)
            batch, label = next(loader)
            batch = batch.to(device)
            label = label.to(device)
            batch = batch*2.-1 if args.data_name.lower() != 'mnist' else batch
            sync_context = nullcontext if last_round else ddp_model.no_sync
            with sync_context():
                dsm_loss, traj_loss = sde.compute_characteristic_loss(ddp_model, ema, teacher_model, feature_extractor, batch, label)
                loss = dsm_loss + traj_loss
                loss.backward()
        for g in optim.param_groups:
            g["lr"] = args.lr * min(step / args.lr_rampup_steps, 1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optim.step()

        ema_decay = min(args.ema_decay, (1 + step) / (10 + step))
        for p_ema, p_net in zip(ema.parameters(), model.parameters()):
            if p_net.requires_grad:
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_decay))

        if step % args.log_freq == 0 and dist.get_rank() == 0:
            logger.add_scalar("denoiser", dsm_loss.item(), step)
            logger.add_scalar("characteristic", traj_loss.item(), step)

        if step % args.sampling_freq == 0 and dist.get_rank() == 0:
            with torch.no_grad():
                _, eval_traj = sde.get_denoiser_and_traj(ema, eval_x1, ones, zeros, eval_label)
                eval_traj = (eval_traj+1.)/2 if args.data_name.lower() != 'mnist' else eval_traj
                grid = make_grid(eval_traj, int(math.sqrt(args.nsampling)))
                logger.add_image(f"Characteristic", grid, step)
        if (
            step % args.dump_freq == 0
            or step == args.train_steps
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(
                    model=ddp_model.module.state_dict(),
                    ema=ema.state_dict(),
                    optim=optim.state_dict(),
                    step=step,
                ),
                f"{args.workdir}/{args.architecture}-{step}.pth",
            )
    if dist.get_rank() == 0:
        logger.close()


if __name__ == "__main__":
    args = create_parser().parse_args()
    if args.mode == "denoiser":
        denoiser_matching(args)
    elif args.mode == "characteristic":
        characteristic_matching(args)
    else:
        raise ValueError(f"unsupported training mode {args.mode}")
