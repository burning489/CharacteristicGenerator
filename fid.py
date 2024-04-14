import argparse

import numpy as np
from scipy import linalg
import torch
import torch.distributed as dist
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from dist_util import setup_dist, get_device
from misc import add_dict_to_argparser, create_dataset
from inception import InceptionV3


def create_parser():
    parser = argparse.ArgumentParser()
    defaults = dict(
        data_name="mnist",
        mode="ref",
        bsz=64,
        num_workers=4,
        dest='',
        img_folder="",
        ref_path='',
        stat_path=''
    )
    add_dict_to_argparser(parser, defaults)
    return parser

def create_dataloader(dataset, bsz, num_workers=1):
    seed = 0
    sampler = DistributedSampler(
        dataset, dist.get_world_size(), dist.get_rank(), seed=seed
    )
    loader = DataLoader(
        dataset, batch_size=bsz, sampler=sampler, num_workers=num_workers
    )
    yield from loader


def calc_stats(args):
    setup_dist()
    device = get_device()

    feature_dim = 2048 if args.data_name.lower() != 'mnist' else 784
    if args.data_name.lower() != 'mnist':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[feature_dim]
        inception_v3 = InceptionV3([block_idx]).to(device)
    if args.mode == 'ref':
        dataset = create_dataset(args)
    elif args.mode == 'gen':
        dataset = ImageFolder(args.img_folder, transform=ToTensor())
    loader = create_dataloader(dataset, args.bsz, args.num_workers)
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _ in loader:
        if images.shape[0] == 0:
            continue
        # if images.shape[1] == 1:
        #     images = images.repeat([1, 3, 1, 1])
        if args.data_name.lower() != 'mnist':
            features = inception_v3(images.to(device))[0].to(torch.float64)
            features = features.squeeze(3).squeeze(2)
        else:
            features = images[:, 0:1, :, :].to(device).to(torch.float64).reshape(images.shape[0], -1)
        mu += features.sum(0)
        sigma += features.T @ features

    dist.barrier()
    dist.all_reduce(mu)
    dist.all_reduce(sigma)
    mu /= len(dataset)
    sigma -= mu.ger(mu) * len(dataset)
    sigma /= len(dataset) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

def calculate_fid_from_inception_stats(mu1, mu2, sigma1, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


if __name__ == '__main__':
    args = create_parser().parse_args()
    if args.mode in ['ref', 'gen']:
        mu, sigma = calc_stats(args)
        np.savez(args.dest, mu=mu, sigma=sigma)
    elif args.mode == 'fid':
        ref = np.load(args.ref_path)
        stat = np.load(args.stat_path)
        fid = calculate_fid_from_inception_stats(ref['mu'], stat['mu'], ref['sigma'], stat['sigma'])
        print(f"{fid:.4f}")
