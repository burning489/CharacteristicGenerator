import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision.transforms import ToTensor, Compose

from network import UNet


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    if isinstance(x, int):
        return x
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def create_karras_grid(t_min, t_max, num_steps, rho=7):
    grid = (
        t_max ** (1 / rho)
        + torch.arange(num_steps)
        / (num_steps - 1)
        * (t_min ** (1 / rho) - t_max ** (1 / rho))
    ) ** rho
    return grid


def create_dataloader(dataset, bsz, num_workers=1, infinite=True):
    if infinite:
        seed = 0
        while True:
            sampler = DistributedSampler(
                dataset, dist.get_world_size(), dist.get_rank(), seed=seed
            )
            loader = DataLoader(
                dataset, batch_size=bsz, sampler=sampler, num_workers=num_workers
            )
            yield from loader
            seed += 1
    else:
        loader = DataLoader(
                dataset, batch_size=bsz, num_workers=num_workers
            )
        yield from loader
        
def create_dataset(args):
    if args.data_name.lower() == "mnist":
        target_transform = Compose(
            [
                lambda x: torch.LongTensor([x]),
                lambda x: F.one_hot(x, 10),
            ]
        )
        dataset = MNIST(
            root="./data/",
            transform=ToTensor(),
            target_transform=target_transform,
            train=True,
        )
    elif args.data_name.lower() == 'cifar':
        target_transform = Compose(
            [
                lambda x: torch.LongTensor([x]),
                lambda x: F.one_hot(x, 10),
            ]
        )
        dataset = CIFAR10(
            root="./data/",
            transform=ToTensor(),
            target_transform=target_transform,
            train=True,
        )
    elif args.data_name.lower() in ["afhq", "afhq-cat"]:
        dataset = ImageFolder('data/afhq-64x64', transform=ToTensor())
    else:
        raise ValueError(
            f'unsupport dataset {args.data_name}, please choose from ["MNIST", "CIFAR"]'
        )
    return dataset


def create_network(args, traj_mode=False):
    if args.architecture.lower() == "ddpmpp":
        model = UNet(
            img_resolution=args.img_resolution,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            label_dim=args.label_dim,
            model_channels=args.model_channels,
            channel_mult=args.channel_mult,
            channel_mult_emb=args.channel_mult_emb,
            num_blocks=args.num_blocks,
            attn_resolutions=args.attn_resolutions,
            dropout=args.dropout,
            label_dropout=args.label_dropout,
            channel_mult_noise=args.channel_mult_noise,
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            resample_filter=[1, 1],
            traj_mode=traj_mode,
        )
    elif args.architecture.lower() == "ncsnpp":
        model = UNet(
            img_resolution=args.img_resolution,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            label_dim=args.label_dim,
            model_channels=args.model_channels,
            channel_mult=args.channel_mult,
            channel_mult_emb=args.channel_mult_emb,
            num_blocks=args.num_blocks,
            attn_resolutions=args.attn_resolutions,
            dropout=args.dropout,
            label_dropout=args.label_dropout,
            channel_mult_noise=args.channel_mult_noise,
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
            resample_filter=[1, 3, 3, 1],
            traj_mode=traj_mode,
        )
    else:
        raise ValueError(f"unsupport network architecture {args.architecture}")
    return model
