def data_defaults(data_name):
    return dict(
        img_resolution=28 if data_name.lower() == "mnist" else 32 if data_name.lower() == 'cifar' else 64,
        in_channels=1 if data_name.lower() == "mnist" else 3,
        out_channels=1 if data_name.lower() == "mnist" else 3,
        label_dim=0,
        bsz=256,
        global_bsz=1024,
        num_workers=4,
    )


def model_defaults(data_name):
    return dict(
        architecture="DDPMpp",
        model_channels=32 if data_name.lower() == "mnist" else 128,
        channel_mult=[1, 2, 2] if data_name.lower() == "mnist" else [1, 2, 2, 2] if data_name.lower() == 'cifar' else [1, 1, 2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[14] if data_name.lower() == "mnist" else [16],
        dropout=0.1,
        label_dropout=0.0,
        channel_mult_noise=1,
    )


def train_defaults(data_name):
    return dict(
        sigma_data=1,
        M=999,
        train_steps=int(1e4) if data_name.lower() == 'mnist' else int(4e5),
        lr=1e-3 if data_name.lower() == "mnist" else 2e-4,
        lr_rampup_steps=500,
        ema_decay=0.999 if data_name.lower() == 'mnist' else 0.9999,
        sampling_freq=int(1e3),
        dump_freq=int(5e3) if data_name.lower() == 'mnist' else int(5e4),
        log_freq=200,
        ode_solver="deis",
        sde_solver="euler-maruyama",
        nsampling=64 if data_name.lower() == 'mnist' else 25,
        train_eps0=1e-5,
        eval_eps0=1e-3,
        eval_eps1=1e-3,
        num_steps=50,
        resume="",
    )

def eval_defaults(data_name):
    return dict(
        sigma_data=1,
        M=999,
        ode_solver="euler",
        eval_eps0=1e-3,
        eval_eps1=1e-3,
        rho=1,
        num_steps=18,
        eval_bsz=64,
        class_idx=None,
        churn=False,
        s_min=0.05,
        s_max=0.6,
        gamma=0.1,
    )

def characteristic_defaults(data_name):
    return dict(
        sigma_data=1,
        M=999,
        train_steps=int(2e3),
        lr=1e-3 if data_name.lower() == "mnist" else 1e-4,
        lr_rampup_steps=200,
        ema_decay=0.999 if data_name.lower() == "mnist" else 0.9999,
        sampling_freq=int(50),
        dump_freq=int(1e3),
        log_freq=5,
        teacher_solver="deis",
        teacher_steps_strategy='weighted',
        nsampling=64,
        train_eps0=1e-3,
        train_eps1=1e-3,
        num_steps=20,
        teacher_steps=19,
        rho=1,
        adaptive_weight=True,
        loss_norm='lpips',
        resume="",
        teacher=""
    )
