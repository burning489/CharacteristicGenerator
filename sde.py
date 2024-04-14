import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist


from misc import append_dims, create_karras_grid

class Follmer:
    def __init__(self, args):
        self.args = args

    def get_alpha(self, t):
        return 1-t
    
    def get_beta(self, t):
        return (t*(2-t)).sqrt()
        
    def get_beta2(self, t):
        return t*(2-t)

    def sampling_prior(self, shape, device):
        return torch.randn(shape, device=device)
    
    def get_cnoise(self, t):
        return self.args.M*t
    
    def get_cin(self, t):
        alpha = self.get_alpha(t)
        beta2 = self.get_beta2(t)
        return 1/((alpha*self.args.sigma_data)**2+beta2).sqrt()
    
    def get_cout(self, t, s=None):
        alpha = self.get_alpha(t)
        beta = self.get_beta(t)
        beta2 = self.get_beta2(t)
        if s is not None:
            return self.get_alpha(s) - self.get_alpha(t) * self.get_beta(s) / self.get_beta(t)
            # return (t-s)/beta2
        else:
            return beta*self.args.sigma_data / ((alpha*self.args.sigma_data)**2+beta2).sqrt()
    
    def get_cskip(self, t, s=None):
        alpha = self.get_alpha(t)
        beta2 = self.get_beta2(t)
        if s is not None:
            return self.get_beta(s) / self.get_beta(t)
            # return 1 + alpha*(s-t)/beta2
        else:
            return alpha*self.args.sigma_data**2 / ((alpha*self.args.sigma_data)**2+beta2)
    
    def get_weightning(self, t):
        alpha = self.get_alpha(t)
        beta2 = self.get_beta2(t)
        return ((alpha*self.args.sigma_data)**2+beta2) / beta2*self.args.sigma_data**2
    
    def get_denoiser(self, model, xt, t, s=None, label=None):
        """D = cskip*x + cout*f(cin*x, cnoise_t, cnoise_s, label).
        s is inactive in classical denoiser mode and active in characteristic mode."""
        ndim = xt.ndim
        cnoise_t = self.get_cnoise(t)
        cnoise_s = self.get_cnoise(s) if s is not None else None
        # alpha = self.get_alpha(t)
        # beta = self.get_beta(t)
        # return (xt - append_dims(beta, ndim)*model(xt, cnoise_t, cnoise_s, label)) / append_dims(alpha, ndim)
        cin = self.get_cin(t)
        cout = self.get_cout(t)
        cskip = self.get_cskip(t)
        return append_dims(cskip, ndim)*xt + append_dims(cout, ndim)*model(append_dims(cin, ndim)*xt, cnoise_t, cnoise_s, label)
    
    def get_denoiser_and_traj(self, model, x, t, s, label=None, stop_grad=False):
        """G = cskip*x + cout*D(with second temporal input)."""
        ndim = x.ndim
        if stop_grad:
            with torch.no_grad():
                denoised = self.get_denoiser(model, x, t, s, label)
        else:
            denoised = self.get_denoiser(model, x, t, s, label)
        cskip = append_dims(self.get_cskip(t, s), ndim)
        cout = append_dims(self.get_cout(t, s), ndim)
        traj = cskip*x + cout*denoised
        return denoised, traj
    
    def get_velocity(self, model, xt, t, label):
        """V = (alpha*x - D)/beta^2"""
        ndim = xt.ndim
        # alpha = append_dims(self.get_alpha(t), ndim)
        # beta = append_dims(self.get_beta(t), ndim)
        # cnoise_t = self.get_cnoise(t)
        # return (model(xt, cnoise_t, label=label)/beta - xt)/alpha
        alpha = self.get_alpha(t)
        beta2 = self.get_beta2(t)
        return (append_dims(alpha, ndim)*xt - self.get_denoiser(model, xt, t, label=label)) / append_dims(beta2, ndim)
    
    def get_score(self, model, xt, t, label=None):
        """S = (alpha*D - x)/beta^2"""
        ndim = xt.ndim
        # cnoise_t = self.get_cnoise(t)
        # beta = self.get_beta(t)
        # return -model(xt, cnoise_t, label=label) / append_dims(beta, ndim)
        alpha = self.get_alpha(t)
        beta2 = self.get_beta2(t)
        return (append_dims(alpha, ndim)*self.get_denoiser(model, xt, t, label=label) - xt) / append_dims(beta2, ndim)
        
    def get_drift(self, x, t):
        """dx = f(t, x) dt + g(t)dw, f(t) = x/t-1, g(t) = sqrt{2/1-t}."""
        ndim = x.ndim
        alpha = self.get_alpha(t)
        return x / append_dims(-alpha, ndim)
    
    def get_diffusion(self, t):
        """dx = f(t, x) dt + g(t)dw, f(t) = x/t-1, g(t) = sqrt{2/1-t}."""
        return (2/(1-t)).sqrt()
    
    def get_reverse_drift(self, model, x, t, label):
        """dx = F(t, x) dt + G(t)dw, f(t) = f(t, x) - g(t)^2*S(t, x), G(t) = g(t)."""
        ndim = x.ndim
        drift = self.get_drift(x, t)
        diffusion = append_dims(self.get_diffusion(t), ndim)
        return (drift - diffusion**2*self.get_score(model, x, t, label=label))
    
    def get_reverse_diffusion(self, t):
        """dx = F(t, x) dt + G(t)dw, f(t) = f(t, x) - g(t)^2*S(t, x), G(t) = g(t)."""
        return self.get_diffusion(t)

    def _get_t_from_idx(self, idx):
        """Get continous time from indices in Karras grid."""
        t_max = 1 - self.args.train_eps1
        t_min = self.args.train_eps0
        karras_grid = create_karras_grid(t_min, t_max, self.args.num_steps, self.args.rho).to(idx.device)
        # karras_grid = torch.cat([karras_grid, torch.zeros_like(karras_grid[:1], device=idx.device)]) 
        return karras_grid[idx]

    def compute_dsm_loss(self, model, batch, label=None):
        """||D(t, xt) - x0||^2, xt = alpha*x0 + beta*z"""
        bsz = batch.shape[0]
        ndim = batch.ndim
        device = batch.device
        t = torch.rand((bsz, ), device=device)*(1 - self.args.train_eps0) + self.args.train_eps0
        alpha = append_dims(self.get_alpha(t), ndim)
        beta = append_dims(self.get_beta(t), ndim)
        weightnings = append_dims(self.get_weightning(t), ndim)
        noise = torch.randn_like(batch, device=device)
        noised = alpha*batch + beta*noise
        # cnoise_t = self.get_cnoise(t)
        # loss = torch.mean(torch.square(model(noised, cnoise_t, label=label) - noise))
        denoised = self.get_denoiser(model, noised, t, label=label)
        loss = weightnings*(denoised - batch)**2
        return torch.mean(loss)

    def compute_characteristic_loss(self, model, target_model, teacher_model, feature_extractor, batch, label=None):
        """Returns local denoiser matching loss and characteristic matching loss(at time zero)."""
        bsz = batch.shape[0]
        ndim = batch.ndim
        device = batch.device

        # denoiser matching
        t = torch.rand((bsz, ), device=device)*(1 - self.args.train_eps0) + self.args.train_eps0
        alpha = append_dims(self.get_alpha(t), ndim)
        beta = append_dims(self.get_beta(t), ndim)
        weightnings = append_dims(self.get_weightning(t), ndim)
        noised = alpha*batch + beta*torch.randn_like(batch, device=device)
        denoised, _ = self.get_denoiser_and_traj(model, noised, t, t, label)
        dsm_loss = torch.mean(weightnings*(denoised - batch)**2)

        # characteristic matching
        # n ~ weighted randint [1, teacher_steps] (working teacher_steps in this batch)
        teacher_steps = [self._get_teacher_steps()]
        dist.broadcast_object_list(teacher_steps, 0)
        teacher_steps = teacher_steps[0]
        # t ~ randint [0, N-n-1]
        t_idx = torch.randint(self.args.num_steps-teacher_steps, (bsz, ), device=device)
        t = self._get_t_from_idx(t_idx)
        alpha = append_dims(self.get_alpha(t), ndim)
        beta = append_dims(self.get_beta(t), ndim)
        noised = alpha*batch + beta*torch.randn_like(batch, device=device)
        # u = t + n
        u_idx = t_idx + teacher_steps
        u = self._get_t_from_idx(u_idx)
        # s ~ randint [u, N-1]
        s_idx = torch.from_numpy(np.random.randint(
            low=u_idx.cpu().detach().numpy(),
            high=self.args.num_steps,
            size=(bsz, ),
            dtype=int)).to(device)
        s = self._get_t_from_idx(s_idx)
        zero = torch.zeros((bsz, ), device=device)
        _, x_t_s = self.get_denoiser_and_traj(model, noised, t, s, label)
        _, x_t_s_0 = self.get_denoiser_and_traj(target_model, x_t_s, s, zero, label, False)
        x_t_u = self._teacher_solver(teacher_model, noised, t_idx, teacher_steps, self.args.teacher_solver, label)
        _, x_t_u_s = self.get_denoiser_and_traj(model, x_t_u, u, s, label, True)
        _, x_t_u_s_0 = self.get_denoiser_and_traj(target_model, x_t_u_s, s, zero, label, True)
        traj_loss = torch.mean(self._extract_feature(feature_extractor, x_t_s_0, x_t_u_s_0))
        if self.args.adaptive_weight:
            if self.args.data_name.lower() == 'mnist':
                balance_weight = self._calculate_adaptive_weight(traj_loss, dsm_loss, last_layer=model.module.dec['28x28_aux_conv'].weight)
            elif self.args.data_name.lower() == 'cifar':
                balance_weight = self._calculate_adaptive_weight(traj_loss, dsm_loss, last_layer=model.module.dec['32x32_aux_conv'].weight)
        else:
            balance_weight = 1.
        dsm_loss = balance_weight*dsm_loss
        return dsm_loss, traj_loss

    def ode_sampler(self, model, x, x_mean, grid, solver, label=None):
        """ODE sampler from x at 1 to 0, the grid starts from 1-eps1 to eps0. 
        (1, 1-eps1) is solved with empirical data mean(exact velocity at 1 is data mean);
        (eps0, 0) is solved with one euler step if using heun solver as it faces singularity at zero."""
        bsz = x.shape[0]
        device = x.device
        # exact velocity at noise end
        xt = x - (1-grid[0]) * x_mean if x_mean is not None else x
        steps = len(grid)
        step_fn = self._get_step_fn(solver)
        for i in range(steps-1):
            t = torch.ones(bsz, device=device)*grid[i]
            dt = grid[i+1] - grid[i]
            xt = step_fn(model, xt, t, label, dt)
        # denoise at data end
        t = torch.ones(bsz, device=device)*grid[-1]
        xt = self.deis_step(model, xt, t, label, -grid[-1]) if solver == 'heun' else self.euler_step(model, xt, t, label, -grid[-1])
        return xt
    
    def sde_sampler(self, model, x, grid, solver, label=None):
        """SDE sampler from x at grid[0] to grid[-1]."""
        bsz = x.shape[0]
        device = x.device
        xt = x
        steps = len(grid)
        for i in range(steps-1):
            t = torch.ones(bsz, device=device)*grid[i]
            dt = grid[i+1] - grid[i]
            if solver.lower() == "euler-maruyama":
                xt = self.euler_maruyama_step(model, xt, t, label, dt)
            else:
                raise ValueError(f"unsupported solver {self.args.sde_solver}")
        return xt
    
    def euler_maruyama_step(self, model, x, t, label, dt):
        """Euler-Maruyama step."""
        ndim = x.ndim
        device = x.device
        dt_sqrt = math.sqrt(math.abs(dt)) if isinstance(dt, int) else dt.abs().sqrt()
        return x + self.get_reverse_drift(model, x, t, label)*dt + append_dims(self.get_reverse_diffusion(t), ndim)*dt_sqrt*torch.randn_like(x, device=device)

    def euler_step(self, model, x, t, label, dt):
        """Euler solver."""
        return x + self.get_velocity(model, x, t, label)*append_dims(dt, x.ndim)

    def heun_step(self, model, x, t, label, dt):
        """Second order heun solver."""
        v = self.get_velocity(model, x, t, label)
        v_phi = self.get_velocity(model, x + v*append_dims(dt, x.ndim), t+dt, label)
        return x + (v+v_phi) / 2 * dt
    
    def deis_step(self, model, x, t, label, dt):
        """Exponential integrator."""
        ndim = x.ndim
        t_next = t+dt
        alpha = self.get_alpha(t)
        alpha_next = self.get_alpha(t_next)
        beta = self.get_beta(t)
        beta_next = self.get_beta(t_next)
        denoised = self.get_denoiser(model, x, t, label=label)
        beta_ratio = beta_next / beta
        lin = append_dims(beta_ratio, ndim)
        nonlin = append_dims(alpha_next - beta_ratio*alpha, ndim)
        return lin * x + nonlin * denoised
    
    def churn_noise(self, x, t_cur, t_next, rng=None):
        """Churning noise from x at t_cur to t_next (t_next > t_cur)."""
        ratio = (1-t_next) / (1-t_cur)
        z = rng.randn_like(x) if rng is not None else torch.randn_like(x, device=x.device)
        x = ratio * x + z*(1 - ratio**2).sqrt() 
        return x

    def _get_step_fn(self, solver):
        """Get ODE step function."""
        if solver.lower() == "euler":
            step_fn = self.euler_step
        elif solver.lower() == "heun":
            step_fn = self.heun_step
        elif solver.lower() == "deis":
            step_fn = self.deis_step
        else:
            raise ValueError(f"unsupported solver {solver}")
        return step_fn
    
    def _get_teacher_steps(self):
        """Pick teacher steps from range [1, teacher_steps].
        Use weighted strategy to prefer larger steps."""
        if self.args.teacher_steps_strategy == 'uniform':
            num_heun_steps = np.random.randint(1,1+self.args.teacher_steps)
        elif self.args.teacher_steps_strategy == 'weighted':
            p = np.array([i for i in range(1,1+self.args.teacher_steps)])
            p = p / sum(p)
            num_heun_steps = np.random.choice([i+1 for i in range(len(p))], size=1, p=p)[0]
        else:
            raise ValueError(f'unsupported heun step random strategy{self.args.teacher_steps_strategy}')
        return num_heun_steps
    
    @torch.no_grad()
    def _teacher_solver(self, model, x, idx, num_steps, solver='deis', label=None):
        """Alike ode_sampler, solves from x at t(computed by idx) for num_steps.
        ode_sampler solves x synchronously;
        _teacher_slover solves x starting at different times, but with same #steps."""
        step_fn = self._get_step_fn(solver)
        for i in range(num_steps):
            t_cur = self._get_t_from_idx(idx+i)
            t_next = self._get_t_from_idx(idx+i+1)
            dt = t_next - t_cur
            x = step_fn(model, x, t_cur, label, dt)
        return x
    
    def _calculate_adaptive_weight(self, loss1, loss2, last_layer=None):
        loss1_grad = torch.autograd.grad(loss1, last_layer, retain_graph=True)[0]
        loss2_grad = torch.autograd.grad(loss2, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(loss1_grad) / (torch.norm(loss2_grad) + 1e-4)
        if dist.get_rank() == 0:
            print(f'grad1={torch.norm(loss1_grad)}, grad2={torch.norm(loss2_grad)}')
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    
    def _extract_feature(self, feature_extractor, pred, target):
        if self.args.loss_norm == 'l2':
            loss = feature_extractor(pred, target)
        elif self.args.loss_norm == 'lpips': 
            if pred.shape[-2] < 256:
                pred = F.interpolate(pred, size=224, mode="bilinear")
                target = F.interpolate(target, size=224, mode="bilinear")
            if pred.shape[1] == 1:
                pred = pred.repeat([1, 3, 1, 1])
                target = target.repeat([1, 3, 1, 1])
            pred = (pred+1)/2. if self.args.data_name.lower() != 'mnist' else pred
            target = (target+1)/2. if self.args.data_name.lower() != 'mnist' else target
            loss = feature_extractor(pred, target)
        return loss
    