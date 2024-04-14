# %%
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# csv_file = './stats/stats.csv'
# df = pd.read_csv(csv_file).rename(columns={'mode': 'model', 'nfe': 'NFE'})
# df['model'] = df['model'].replace({'denoiser': 'ODE'})
# df['dataset'] = df['dataset'].replace({'cifar': 'CIFAR10', 'mnist': 'MNIST'})

# g = sns.relplot(
#     data=df,
#     x="NFE", y="FID",
#     hue="model", style="model", col="dataset",
#     height=3, aspect=.7,
#     kind="line", facet_kws={"sharey": False}
# )
# (g.set_axis_labels("NFE", "FID")
#   .set_titles("Dataset: {col_name}")
#   .tight_layout(w_pad=0))
# for ax in g.axes.flatten():
#     ax.grid(linestyle=':')
#     ax.set_xticks([1, 2, 5, 10, 20])

# plt.savefig('./asset/mnist_cifar_nfe.png', dpi=300)

# %%
import argparse

import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from defaults import data_defaults, model_defaults, eval_defaults
from misc import add_dict_to_argparser, create_network
from eval import ode_sampler, characteristic_sampler, _compute_x_mean
from sde import Follmer

parser = argparse.ArgumentParser()
defaults = dict(
    device="cuda:0",
    seed=128,
    nsample=12,
    # data_name="cifar",
    # denoiser_checkpoint="logs/cifar-denoiser/DDPMpp-100000.pth",
    # characteristic_checkpoint="logs/cifar-characteristic/DDPMpp-2000.pth",
    data_name="mnist",
    denoiser_checkpoint="logs/mnist-denoiser/DDPMpp-10000.pth",
    characteristic_checkpoint="logs/mnist-characteristic/DDPMpp-2000.pth",

)
defaults.update(data_defaults(defaults["data_name"]))
defaults.update(model_defaults(defaults["data_name"]))
defaults.update(eval_defaults(defaults["data_name"]))
add_dict_to_argparser(parser, defaults)
args = parser.parse_args(['--ode_solver', 'deis'])

device = torch.device(args.device)
torch.manual_seed(args.seed)

denoiser_model = create_network(args).eval().to(device)
characteristic_model = create_network(args, True).eval().to(device)
denoiser_model.load_state_dict(torch.load(args.denoiser_checkpoint, map_location=device)['ema'])
characteristic_model.load_state_dict(torch.load(args.characteristic_checkpoint, map_location=device)['ema'])
sde = Follmer(args)
x_mean = _compute_x_mean(args).to(device)

rng=None
nfes = (1, 2, 5, 10, 20)
latents = torch.randn([args.nsample, args.in_channels, args.img_resolution, args.img_resolution], device=device)
labels = None
if args.label_dim:
    labels = torch.eye(args.label_dim, device=device)[torch.randint(args.label_dim, size=[args.nsample], device=device)]
if args.class_idx is not None:
    labels[:, :] = 0
    labels[:, args.class_idx] = 1

ode_images = torch.empty([len(nfes), args.nsample, args.in_channels, args.img_resolution, args.img_resolution], device=device)
characteristic_images = torch.empty([len(nfes), args.nsample, args.in_channels, args.img_resolution, args.img_resolution], device=device)
for i, nfe in enumerate(nfes):
    ode_images[i, ...] = ode_sampler(sde, denoiser_model, latents, x_mean, labels, args.ode_solver, rng, nfe, args.rho, args.eval_eps0, 1-args.eval_eps1, args.churn, args.s_min, args.s_max, args.gamma)
    characteristic_images[i, ...] = characteristic_sampler(sde, characteristic_model, latents, x_mean, labels, rng, nfe, args.rho, args.eval_eps0, 1-args.eval_eps1, args.churn, args.s_min, args.s_max, args.gamma)

ode_images = ode_images.permute(1, 0, 2, 3, 4)
ode_images = (ode_images+1.)/2 if args.data_name.lower() != 'mnist' else ode_images
ode_images = ode_images.reshape(-1, args.in_channels, args.img_resolution, args.img_resolution)

characteristic_images = characteristic_images.permute(1, 0, 2, 3, 4)
characteristic_images = (characteristic_images+1.)/2 if args.data_name.lower() != 'mnist' else characteristic_images
characteristic_images = characteristic_images.reshape(-1, args.in_channels, args.img_resolution, args.img_resolution)
ode_images = make_grid(ode_images, nrow=len(nfes))
characteristic_images = make_grid(characteristic_images, nrow=len(nfes))

fig, axes = plt.subplots(1, 2, gridspec_kw={"wspace":0})

axes[0].imshow(ode_images.permute(1, 2, 0).cpu().numpy())
axes[0].set_title('ODE')
x_lim = axes[0].get_xlim()
width_per_icon = (x_lim[1] - x_lim[0]) / len(nfes)
axes[0].set_xticks([(i+1/2)*width_per_icon for i in range(len(nfes))], nfes)
axes[0].set_yticks([])
axes[0].set_xlabel('NFE')
axes[0].tick_params(length=0)

axes[1].imshow(characteristic_images.permute(1, 2, 0).cpu().numpy())
axes[1].set_title('Characteristics')
x_lim = axes[1].get_xlim()
width_per_icon = (x_lim[1] - x_lim[0]) / len(nfes)
axes[1].set_xticks([(i+1/2)*width_per_icon for i in range(len(nfes))], nfes)
axes[1].set_yticks([])
axes[1].set_xlabel('NFE')
axes[1].tick_params(length=0)

plt.savefig(f'{args.data_name}-cmp.png', dpi=300)

# save_image(ode_images, f'ode.png', nrow=len(nfes))
# save_image(characteristic_images, f'characteristic.png', nrow=len(nfes))


# %%
