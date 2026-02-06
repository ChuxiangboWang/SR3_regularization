import os
import math
import torch
import torchvision.utils as vutils
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from regularization import *
from datetime import datetime


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        tv1_weight=None,
        tv2_weight=None,
        tvf_weight=None,
        tvf_alpha=1.6,
        wavelet_l1_weight = None,
        wavelet_type = "haar",
        vgg_opt=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)
        self.tv1_weight = tv1_weight
        self.tv2_weight = tv2_weight
        self.tvf_weight = tvf_weight
        self.tvf_alpha = tvf_alpha
        self.wavelet_type = wavelet_type
        self.wavelet_l1_weight = wavelet_l1_weight


        if vgg_opt is None:
            vgg_opt = {}
         
        # weight for VGG loss
        self.vgg_weight = float(vgg_opt.get("weight", 0.0))
        self.vgg_start_step = int(vgg_opt.get("start_step", 0))
        self.vgg_vis_freq = int(vgg_opt.get("vis_freq", 1000))
        self.vgg_vis_num_channels = int(vgg_opt.get("vis_num_channels", 36))

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.vgg_vis_dir = os.path.join("VGG feature", timestamp)
        self.global_step = 0
        
        
        
        
        
    
        
        
        
        
        
        # --------- VGG19 ---------
        import torchvision.models as models
        import torch.nn as nn

        vgg = models.vgg19(pretrained=True).features[:16]
        self.vgg = nn.Sequential(*vgg)

        # Freeze VGG parameters
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.vgg.eval()    # inference mode




    def vgg_features(self, x):
        
        
        if x.dim() == 4 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)   # [B,1,H,W] -> [B,3,H,W]
        
        # normalized to ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)

        x_norm = (x - mean) / std
        return self.vgg(x_norm)
    
    def save_feature_map(self, fmap, filename, num_channels=36):

        b, c, h, w = fmap.shape
        # Take batch 0, first num_channels channels
        n = min(num_channels, c)
        x = fmap[0, :n, :, :].clone()   # [n, H, W]

        # Normalize each channel to [0, 1]
        for i in range(n):
            fm = x[i]
            fm_min = fm.min()
            fm_max = fm.max()
            x[i] = (fm - fm_min) / (fm_max - fm_min + 1e-8)
            
        x = x.unsqueeze(1)
        nrow = int(math.sqrt(n)) if int(math.sqrt(n))**2 == n else int(math.sqrt(n)) + 1
        vutils.save_image(x, filename, nrow=nrow)


    def save_rgb_image(self, img, filename):

        x = img[0].detach() 

     
        x = (x + 1) / 2.0
        x = x.clamp(0.0, 1.0)

        vutils.save_image(x, filename)





    def set_loss(self, device):

        self.vgg.to(device)

        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None,generator = None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) if t > 0 else torch.zeros_like(x)

        # noise = torch.randn_like(x,generator=generator) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False,noise=None, generator=None):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            if noise is None:
                img = torch.randn(shape, device=device,generator=generator)
            else:
                img = noise
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i,generator=generator)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            if noise is None:
                img = torch.randn(shape, device=device,generator=generator)
            else:
                img = noise
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x,generator=generator)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False,generator=None):
        return self.p_sample_loop(x_in, continous,generator=generator)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        # print('sqrt_alphas_cumprod_prev', self.sqrt_alphas_cumprod_prev.shape)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        y_recon = self.predict_start_from_noise(x_noisy, t-1, x_recon)


        # Save φ(y), φ(y_recon), and the difference (every 1000 steps)
        if self.vgg_weight > 0 and self.global_step >= self.vgg_start_step and (self.global_step % self.vgg_vis_freq == 0):
            y = x_in['HR']
            with torch.no_grad():
                phi_y = self.vgg_features(y)
                phi_recon = self.vgg_features(y_recon)
                phi_diff = phi_y - phi_recon

            # ensure folder exists
            os.makedirs(self.vgg_vis_dir, exist_ok=True)

            # Save original HR and reconstructed HR
            self.save_rgb_image(
                y,
                f"{self.vgg_vis_dir}/hr_y_step{self.global_step}.png"
            )
            self.save_rgb_image(
                y_recon,
                f"{self.vgg_vis_dir}/hr_y_recon_step{self.global_step}.png"
            )

            # Save feature maps
            self.save_feature_map(
                phi_y,
                f"{self.vgg_vis_dir}/vgg_phi_y_step{self.global_step}.png", num_channels=self.vgg_vis_num_channels,
            )
            self.save_feature_map(
                phi_recon,
                f"{self.vgg_vis_dir}/vgg_phi_recon_step{self.global_step}.png", num_channels=self.vgg_vis_num_channels,
            )
            self.save_feature_map(
                phi_diff,
                f"{self.vgg_vis_dir}/vgg_phi_diff_step{self.global_step}.png", num_channels=self.vgg_vis_num_channels,
            )



        # --------- VGG Loss ---------
        if self.vgg_weight > 0 and self.global_step >= self.vgg_start_step:
            # Ground-truth HR image
            y = x_in['HR']

            # φ(y) and φ(y_recon)
            with torch.no_grad():
                phi_y = self.vgg_features(y)   # GT feature

            phi_recon = self.vgg_features(y_recon)  # reconstructed feature

            # L1 loss between VGG features
            loss_vgg = self.vgg_weight * F.l1_loss(phi_recon, phi_y)

        else:
            loss_vgg = 0.0

        loss_noise = self.loss_func(noise, x_recon)
        loss_TV1 = self.tv1_weight*TV1(y_recon)
        loss_TV2 = self.tv2_weight*TV2(y_recon)
        loss_TVF = self.tvf_weight*FTV(y_recon, alpha=self.tvf_alpha)
        loss_wave = self.wavelet_l1_weight*waveL1(y_recon, wname=self.wavelet_type)
        l_total = loss_noise + loss_TV1+ loss_TV2 + loss_TVF+ loss_wave + loss_vgg


        self.global_step += 1

        return  {
            "total": l_total,
            "loss_noise": loss_noise,
            "loss_TV1": loss_TV1,
            "loss_TV2": loss_TV2,
            "loss_TVF": loss_TVF,
            "loss_wave_l1":loss_wave,
            "loss_vgg": loss_vgg
        }


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

