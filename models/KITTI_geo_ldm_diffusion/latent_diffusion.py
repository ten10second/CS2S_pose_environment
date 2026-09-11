import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from inspect import isfunction

from utils.util import instantiate_from_config
import torch.nn.functional as F
def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

class DDPM(pl.LightningModule):
    def __init__(self,
                unet_config,
                control_grd=None,
                timesteps=200,
                linear_start=1e-4,
                linear_end=2e-2,
                cosine_s=8e-3,
                parameterization="eps",
                v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                logvar_init = 0.,
                ):
        super().__init__()
        #lr
        self.log_every_t = timesteps/3
        self.parameterization = parameterization
        self.v_posterior = v_posterior
        self.clip_denoised = False
        self.timesteps = timesteps
        

        self.denoise_model = instantiate_from_config(unet_config)
        self.control_grd = instantiate_from_config(control_grd) if control_grd else None
        self.last_loss_metrics = {}
        self.register_schedule(timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))

    def register_schedule(self, beta_schedule="linear", timesteps=200,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                    cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def t_losses(self, x_start, cond_init_grd=None, cond_sat=None, cond_txt = None, lidar_context=None, lidar_evidence=None, noise=None, left_camera_k=None, gt_shift_x=None, gt_shift_y=None, theta=None, range_img=None, range_mask=None, camera_to_lidar=None, loss_mask=None, loss_mask_weight=0.0, x0_loss_weight=0.0, extra_loss_mask=None, extra_loss_mask_weight=0.0, extra_x0_loss_weight=0.0, foreground_loss_mask=None, foreground_loss_weight=0.0, foreground_x0_loss_weight=0.0, foreground_image_loss_weight=0.0, foreground_lpips_loss_weight=0.0, foreground_image_loss_mask=None, foreground_lpips_padding=8, foreground_lpips_size=96, image_x0_loss_weight=0.0, crop_image_x0_loss_weight=0.0, point_image_x0_loss_weight=0.0, object_lpips_loss_weight=0.0, x0_image_target=None, image_loss_mask=None, crop_image_loss_mask=None, point_image_loss_mask=None, object_boxes=None, object_box_valid=None, object_lpips_model=None, object_lpips_padding=4, object_lpips_size=64, object_lpips_max_boxes=4, image_decoder=None, latent_scale_factor=1.0, static_teacher_loss_mask=None, static_teacher_consistency_weight=0.0, lidar_depth_target=None, lidar_depth_mask=None, lidar_depth_loss_weight=0.0, lidar_depth_output_scale=1.0, lidar_depth_bottleneck_scale=1.0, lidar_depth_log_eps=1e-3, return_outputs=False):
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
        return self.p_losses(x_start, t, cond_init_grd = cond_init_grd, cond_sat = cond_sat, cond_txt = cond_txt, lidar_context=lidar_context, lidar_evidence=lidar_evidence,  left_camera_k = left_camera_k, gt_shift_x = gt_shift_x, gt_shift_y = gt_shift_y, theta = theta, range_img=range_img, range_mask=range_mask, camera_to_lidar=camera_to_lidar, loss_mask=loss_mask, loss_mask_weight=loss_mask_weight, x0_loss_weight=x0_loss_weight, extra_loss_mask=extra_loss_mask, extra_loss_mask_weight=extra_loss_mask_weight, extra_x0_loss_weight=extra_x0_loss_weight, foreground_loss_mask=foreground_loss_mask, foreground_loss_weight=foreground_loss_weight, foreground_x0_loss_weight=foreground_x0_loss_weight, foreground_image_loss_weight=foreground_image_loss_weight, foreground_lpips_loss_weight=foreground_lpips_loss_weight, foreground_image_loss_mask=foreground_image_loss_mask, foreground_lpips_padding=foreground_lpips_padding, foreground_lpips_size=foreground_lpips_size, image_x0_loss_weight=image_x0_loss_weight, crop_image_x0_loss_weight=crop_image_x0_loss_weight, point_image_x0_loss_weight=point_image_x0_loss_weight, object_lpips_loss_weight=object_lpips_loss_weight, x0_image_target=x0_image_target, image_loss_mask=image_loss_mask, crop_image_loss_mask=crop_image_loss_mask, point_image_loss_mask=point_image_loss_mask, object_boxes=object_boxes, object_box_valid=object_box_valid, object_lpips_model=object_lpips_model, object_lpips_padding=object_lpips_padding, object_lpips_size=object_lpips_size, object_lpips_max_boxes=object_lpips_max_boxes, image_decoder=image_decoder, latent_scale_factor=latent_scale_factor, static_teacher_loss_mask=static_teacher_loss_mask, static_teacher_consistency_weight=static_teacher_consistency_weight, lidar_depth_target=lidar_depth_target, lidar_depth_mask=lidar_depth_mask, lidar_depth_loss_weight=lidar_depth_loss_weight, lidar_depth_output_scale=lidar_depth_output_scale, lidar_depth_bottleneck_scale=lidar_depth_bottleneck_scale, lidar_depth_log_eps=lidar_depth_log_eps, return_outputs=return_outputs)

    def _prepare_loss_mask(self, loss_mask, loss_raw):
        if loss_mask is None:
            return None
        if loss_mask.ndim == 3:
            loss_mask = loss_mask[:, None]
        if loss_mask.shape[-2:] != loss_raw.shape[-2:]:
            loss_mask = F.interpolate(loss_mask.float(), size=loss_raw.shape[-2:], mode="area")
        return loss_mask.to(device=loss_raw.device, dtype=loss_raw.dtype).clamp(0.0, 1.0)

    def _masked_loss_mean(self, loss_raw, loss_mask):
        loss_mask = self._prepare_loss_mask(loss_mask, loss_raw)
        if loss_mask is None:
            return loss_raw.mean(dim=[1, 2, 3]).mean()
        masked = loss_raw * loss_mask
        denom = loss_mask.mean(dim=[1, 2, 3]).clamp_min(1e-6)
        return (masked.mean(dim=[1, 2, 3]) / denom).mean()

    def _object_lpips_loss(self, pred_image, target_image, object_boxes, object_box_valid, lpips_model, padding, crop_size, max_boxes):
        if lpips_model is None or object_boxes is None or object_box_valid is None:
            return pred_image.new_tensor(0.0)
        losses = []
        _, _, h, w = pred_image.shape
        boxes = object_boxes.to(device=pred_image.device)
        valid = object_box_valid.to(device=pred_image.device)
        for batch_idx in range(pred_image.shape[0]):
            valid_indices = torch.nonzero(valid[batch_idx] > 0.5, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            if max_boxes > 0:
                valid_indices = valid_indices[:max_boxes]
            for box_idx in valid_indices:
                box = boxes[batch_idx, box_idx]
                x0 = max(0, int(torch.floor(box[0]).detach().cpu()) - int(padding))
                y0 = max(0, int(torch.floor(box[1]).detach().cpu()) - int(padding))
                x1 = min(w - 1, int(torch.ceil(box[2]).detach().cpu()) + int(padding))
                y1 = min(h - 1, int(torch.ceil(box[3]).detach().cpu()) + int(padding))
                if x1 <= x0 or y1 <= y0:
                    continue
                pred_crop = pred_image[batch_idx : batch_idx + 1, :, y0 : y1 + 1, x0 : x1 + 1]
                target_crop = target_image[batch_idx : batch_idx + 1, :, y0 : y1 + 1, x0 : x1 + 1]
                if crop_size > 0:
                    pred_crop = F.interpolate(pred_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                    target_crop = F.interpolate(target_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                losses.append(lpips_model(target_crop, pred_crop).mean())
        if not losses:
            return pred_image.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _foreground_lpips_loss(self, pred_image, target_image, foreground_mask, lpips_model, padding, crop_size):
        if lpips_model is None or foreground_mask is None:
            return pred_image.new_tensor(0.0)
        mask = self._prepare_loss_mask(foreground_mask, pred_image)
        if mask is None:
            return pred_image.new_tensor(0.0)
        losses = []
        _, _, h, w = pred_image.shape
        for batch_idx in range(pred_image.shape[0]):
            coords = torch.nonzero(mask[batch_idx, 0] > 0.5, as_tuple=False)
            if coords.numel() == 0:
                continue
            y0 = max(0, int(coords[:, 0].min().detach().cpu()) - int(padding))
            y1 = min(h - 1, int(coords[:, 0].max().detach().cpu()) + int(padding))
            x0 = max(0, int(coords[:, 1].min().detach().cpu()) - int(padding))
            x1 = min(w - 1, int(coords[:, 1].max().detach().cpu()) + int(padding))
            if x1 <= x0 or y1 <= y0:
                continue
            pred_crop = pred_image[batch_idx : batch_idx + 1, :, y0 : y1 + 1, x0 : x1 + 1]
            target_crop = target_image[batch_idx : batch_idx + 1, :, y0 : y1 + 1, x0 : x1 + 1]
            mask_crop = mask[batch_idx : batch_idx + 1, :, y0 : y1 + 1, x0 : x1 + 1]
            pred_crop = pred_crop * mask_crop + target_crop * (1.0 - mask_crop)
            if crop_size > 0:
                pred_crop = F.interpolate(pred_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                target_crop = F.interpolate(target_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
            losses.append(lpips_model(target_crop, pred_crop).mean())
        if not losses:
            return pred_image.new_tensor(0.0)
        return torch.stack(losses).mean()

    def p_losses(self, x_start, t, cond_init_grd = None, cond_sat = None, cond_txt = None, lidar_context=None, lidar_evidence=None, noise=None,  left_camera_k=None, gt_shift_x=None, gt_shift_y=None, theta=None, range_img=None, range_mask=None, camera_to_lidar=None, loss_mask=None, loss_mask_weight=0.0, x0_loss_weight=0.0, extra_loss_mask=None, extra_loss_mask_weight=0.0, extra_x0_loss_weight=0.0, foreground_loss_mask=None, foreground_loss_weight=0.0, foreground_x0_loss_weight=0.0, foreground_image_loss_weight=0.0, foreground_lpips_loss_weight=0.0, foreground_image_loss_mask=None, foreground_lpips_padding=8, foreground_lpips_size=96, image_x0_loss_weight=0.0, crop_image_x0_loss_weight=0.0, point_image_x0_loss_weight=0.0, object_lpips_loss_weight=0.0, x0_image_target=None, image_loss_mask=None, crop_image_loss_mask=None, point_image_loss_mask=None, object_boxes=None, object_box_valid=None, object_lpips_model=None, object_lpips_padding=4, object_lpips_size=64, object_lpips_max_boxes=4, image_decoder=None, latent_scale_factor=1.0, static_teacher_loss_mask=None, static_teacher_consistency_weight=0.0, lidar_depth_target=None, lidar_depth_mask=None, lidar_depth_loss_weight=0.0, lidar_depth_output_scale=1.0, lidar_depth_bottleneck_scale=1.0, lidar_depth_log_eps=1e-3, return_outputs=False):
        loss_metrics = {}

        def record_loss(name, value):
            if torch.is_tensor(value):
                loss_metrics[name] = float(value.detach().cpu())
            else:
                loss_metrics[name] = float(value)

        noise = default(noise, lambda: torch.randn_like(x_start)) 
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        control_grd_para = None
        if cond_init_grd is not None and self.control_grd is not None:
            control_grd_para = self.control_grd(
                x_noisy,
                t,
                cond_init_grd=cond_init_grd,
                cond_sat=cond_sat,
                cond_txt=cond_txt,
                range_img=range_img,
                range_mask=range_mask,
                camera_to_lidar=camera_to_lidar,
                camera_k=left_camera_k,
                image_size=tuple(cond_init_grd.shape[-2:]),
        )
        ray_mask_mode = str(getattr(self, "ray_evidence_mask_mode", "foreground") or "foreground")
        if ray_mask_mode == "none":
            lidar_geometry_mask = None
        elif ray_mask_mode == "lidar_hit" and lidar_evidence is not None and lidar_evidence.shape[1] > 1:
            lidar_geometry_mask = lidar_evidence[:, 1:2]
        else:
            lidar_geometry_mask = foreground_loss_mask
        model_out = self.denoise_model(
            x_noisy,
            t,
            context=cond_txt,
            lidar_context=lidar_context,
            lidar_evidence=lidar_evidence,
            lidar_geometry_mask=lidar_geometry_mask,
            control_grd=control_grd_para,
            left_camera_k=left_camera_k,
            gt_shift_x=gt_shift_x,
            gt_shift_y=gt_shift_y,
            theta=theta,
        )
        lidar_depth_pred = getattr(self.denoise_model, "last_lidar_depth_pred", None)
        lidar_bottleneck_depth_pred = getattr(self.denoise_model, "last_lidar_bottleneck_depth_pred", None)

        target = noise
        # loss = (target - model_out).abs().mean()
        loss_raw = F.mse_loss(target, model_out, reduction='none')
        if loss_mask is not None and loss_mask_weight > 0.0:
            prepared_mask = self._prepare_loss_mask(loss_mask, loss_raw)
            weight = 1.0 + float(loss_mask_weight) * prepared_mask
            loss_per_sample = (loss_raw * weight).mean(dim=[1, 2, 3]) / weight.mean(dim=[1, 2, 3]).clamp_min(1e-6)
            loss = loss_per_sample.mean()
        else:
            loss = loss_raw.mean(dim=[1, 2, 3]).mean()
        record_loss("loss_eps_base", loss)
        if extra_loss_mask is not None and extra_loss_mask_weight > 0.0:
            extra_eps_loss = self._masked_loss_mean(loss_raw, extra_loss_mask)
            loss = loss + float(extra_loss_mask_weight) * extra_eps_loss
            record_loss("loss_lidar_hit_eps", extra_eps_loss)
            record_loss("loss_lidar_hit_eps_contrib", float(extra_loss_mask_weight) * extra_eps_loss)
        if foreground_loss_mask is not None and foreground_loss_weight > 0.0:
            foreground_eps_loss = self._masked_loss_mean(loss_raw, foreground_loss_mask)
            loss = loss + float(foreground_loss_weight) * foreground_eps_loss
            record_loss("loss_foreground_eps", foreground_eps_loss)
            record_loss("loss_foreground_eps_contrib", float(foreground_loss_weight) * foreground_eps_loss)
        if (
            static_teacher_loss_mask is not None
            and static_teacher_consistency_weight > 0.0
        ):
            with torch.no_grad():
                teacher_out = self.denoise_model(
                    x_noisy,
                    t,
                    context=cond_txt,
                    lidar_context=None,
                    control_grd=None,
                    left_camera_k=left_camera_k,
                    gt_shift_x=gt_shift_x,
                    gt_shift_y=gt_shift_y,
                    theta=theta,
                )
            teacher_loss_raw = F.mse_loss(model_out, teacher_out, reduction="none")
            static_teacher_loss = self._masked_loss_mean(teacher_loss_raw, static_teacher_loss_mask)
            loss = loss + float(static_teacher_consistency_weight) * static_teacher_loss
            record_loss("loss_static_teacher_eps", static_teacher_loss)
            record_loss(
                "loss_static_teacher_eps_contrib",
                float(static_teacher_consistency_weight) * static_teacher_loss,
            )
        needs_pred_x0 = (
            (loss_mask is not None and x0_loss_weight > 0.0)
            or (extra_loss_mask is not None and extra_x0_loss_weight > 0.0)
            or (foreground_loss_mask is not None and foreground_x0_loss_weight > 0.0)
            or (image_x0_loss_weight > 0.0 and x0_image_target is not None and image_loss_mask is not None and image_decoder is not None)
            or (crop_image_x0_loss_weight > 0.0 and x0_image_target is not None and crop_image_loss_mask is not None and image_decoder is not None)
            or (point_image_x0_loss_weight > 0.0 and x0_image_target is not None and point_image_loss_mask is not None and image_decoder is not None)
            or (foreground_image_loss_weight > 0.0 and x0_image_target is not None and foreground_image_loss_mask is not None and image_decoder is not None)
            or (foreground_lpips_loss_weight > 0.0 and x0_image_target is not None and foreground_image_loss_mask is not None and object_lpips_model is not None and image_decoder is not None)
            or (object_lpips_loss_weight > 0.0 and x0_image_target is not None and object_boxes is not None and object_box_valid is not None and object_lpips_model is not None and image_decoder is not None)
            or bool(return_outputs)
        )
        pred_x0 = self.predict_start_from_noise(x_noisy, t=t, noise=model_out) if needs_pred_x0 else None
        if loss_mask is not None and x0_loss_weight > 0.0:
            x0_loss_raw = F.mse_loss(x_start, pred_x0, reduction="none")
            x0_loss = self._masked_loss_mean(x0_loss_raw, loss_mask)
            loss = loss + float(x0_loss_weight) * x0_loss
            record_loss("loss_dynamic_x0", x0_loss)
            record_loss("loss_dynamic_x0_contrib", float(x0_loss_weight) * x0_loss)
        if extra_loss_mask is not None and extra_x0_loss_weight > 0.0:
            x0_loss_raw = F.mse_loss(x_start, pred_x0, reduction="none")
            extra_x0_loss = self._masked_loss_mean(x0_loss_raw, extra_loss_mask)
            loss = loss + float(extra_x0_loss_weight) * extra_x0_loss
            record_loss("loss_lidar_hit_x0", extra_x0_loss)
            record_loss("loss_lidar_hit_x0_contrib", float(extra_x0_loss_weight) * extra_x0_loss)
        if foreground_loss_mask is not None and foreground_x0_loss_weight > 0.0:
            x0_loss_raw = F.mse_loss(x_start, pred_x0, reduction="none")
            foreground_x0_loss = self._masked_loss_mean(x0_loss_raw, foreground_loss_mask)
            loss = loss + float(foreground_x0_loss_weight) * foreground_x0_loss
            record_loss("loss_foreground_x0", foreground_x0_loss)
            record_loss("loss_foreground_x0_contrib", float(foreground_x0_loss_weight) * foreground_x0_loss)
        def add_lidar_depth_loss(depth_pred, prefix, weight_scale=1.0):
            depth_pred = depth_pred.float().clamp(1e-6, 1.0)
            depth_target = lidar_depth_target.float()
            if depth_target.ndim == 3:
                depth_target = depth_target[:, None]
            if depth_target.shape[-2:] != depth_pred.shape[-2:]:
                depth_target = F.interpolate(depth_target, size=depth_pred.shape[-2:], mode="nearest")
            depth_target = depth_target.to(device=depth_pred.device, dtype=depth_pred.dtype).clamp(1e-6, 1.0)
            depth_mask = self._prepare_loss_mask(lidar_depth_mask, depth_pred)
            eps = max(float(lidar_depth_log_eps), 1e-6)
            depth_loss_raw = (
                torch.log(depth_pred.clamp_min(eps)) - torch.log(depth_target.clamp_min(eps))
            ).abs()
            lidar_depth_loss = self._masked_loss_mean(depth_loss_raw, depth_mask)
            denom = depth_mask.sum().clamp_min(1e-6)
            loss_name = f"loss_lidar_{prefix}_depth_log_l1" if prefix else "loss_lidar_depth_log_l1"
            contrib_name = (
                f"loss_lidar_{prefix}_depth_log_l1_contrib"
                if prefix
                else "loss_lidar_depth_log_l1_contrib"
            )
            coverage_name = f"lidar_{prefix}_depth_mask_coverage" if prefix else "lidar_depth_mask_coverage"
            pred_mean_name = f"lidar_{prefix}_depth_pred_mean" if prefix else "lidar_depth_pred_mean"
            target_mean_name = f"lidar_{prefix}_depth_target_mean" if prefix else "lidar_depth_target_mean"
            depth_weight = float(lidar_depth_loss_weight) * float(weight_scale)
            record_loss(loss_name, lidar_depth_loss)
            record_loss(contrib_name, depth_weight * lidar_depth_loss)
            record_loss(coverage_name, depth_mask.mean())
            record_loss(pred_mean_name, (depth_pred * depth_mask).sum() / denom)
            record_loss(target_mean_name, (depth_target * depth_mask).sum() / denom)
            return depth_weight * lidar_depth_loss

        if (
            lidar_depth_target is not None
            and lidar_depth_mask is not None
            and float(lidar_depth_loss_weight) > 0.0
        ):
            if lidar_depth_pred is not None and float(lidar_depth_output_scale) > 0.0:
                loss = loss + add_lidar_depth_loss(lidar_depth_pred, "", lidar_depth_output_scale)
            if lidar_bottleneck_depth_pred is not None and float(lidar_depth_bottleneck_scale) > 0.0:
                loss = loss + add_lidar_depth_loss(
                    lidar_bottleneck_depth_pred,
                    "bottleneck",
                    lidar_depth_bottleneck_scale,
                )
        needs_image_x0 = x0_image_target is not None and image_decoder is not None and (
            (image_x0_loss_weight > 0.0 and image_loss_mask is not None)
            or (crop_image_x0_loss_weight > 0.0 and crop_image_loss_mask is not None)
            or (point_image_x0_loss_weight > 0.0 and point_image_loss_mask is not None)
            or (foreground_image_loss_weight > 0.0 and foreground_image_loss_mask is not None)
            or (foreground_lpips_loss_weight > 0.0 and foreground_image_loss_mask is not None and object_lpips_model is not None)
            or (object_lpips_loss_weight > 0.0 and object_boxes is not None and object_box_valid is not None and object_lpips_model is not None)
            or bool(return_outputs)
        )
        pred_image = None
        if needs_image_x0:
            pred_image = image_decoder.decode(pred_x0 * (1.0 / float(latent_scale_factor)))
            image_loss_raw = F.l1_loss(pred_image, x0_image_target, reduction="none")
        if image_x0_loss_weight > 0.0 and x0_image_target is not None and image_loss_mask is not None and image_decoder is not None:
            image_x0_loss = self._masked_loss_mean(image_loss_raw, image_loss_mask)
            loss = loss + float(image_x0_loss_weight) * image_x0_loss
            record_loss("loss_dynamic_image_l1", image_x0_loss)
            record_loss("loss_dynamic_image_l1_contrib", float(image_x0_loss_weight) * image_x0_loss)
        if crop_image_x0_loss_weight > 0.0 and x0_image_target is not None and crop_image_loss_mask is not None and image_decoder is not None:
            crop_image_x0_loss = self._masked_loss_mean(image_loss_raw, crop_image_loss_mask)
            loss = loss + float(crop_image_x0_loss_weight) * crop_image_x0_loss
            record_loss("loss_dynamic_crop_image_l1", crop_image_x0_loss)
            record_loss("loss_dynamic_crop_image_l1_contrib", float(crop_image_x0_loss_weight) * crop_image_x0_loss)
        if point_image_x0_loss_weight > 0.0 and x0_image_target is not None and point_image_loss_mask is not None and image_decoder is not None:
            point_image_x0_loss = self._masked_loss_mean(image_loss_raw, point_image_loss_mask)
            loss = loss + float(point_image_x0_loss_weight) * point_image_x0_loss
            record_loss("loss_lidar_hit_image_l1", point_image_x0_loss)
            record_loss("loss_lidar_hit_image_l1_contrib", float(point_image_x0_loss_weight) * point_image_x0_loss)
        if foreground_image_loss_weight > 0.0 and x0_image_target is not None and foreground_image_loss_mask is not None and image_decoder is not None:
            foreground_image_loss = self._masked_loss_mean(image_loss_raw, foreground_image_loss_mask)
            loss = loss + float(foreground_image_loss_weight) * foreground_image_loss
            record_loss("loss_foreground_image_l1", foreground_image_loss)
            record_loss("loss_foreground_image_l1_contrib", float(foreground_image_loss_weight) * foreground_image_loss)
        if object_lpips_loss_weight > 0.0 and x0_image_target is not None and object_boxes is not None and object_box_valid is not None and object_lpips_model is not None and image_decoder is not None:
            object_lpips = self._object_lpips_loss(
                pred_image,
                x0_image_target,
                object_boxes,
                object_box_valid,
                object_lpips_model,
                object_lpips_padding,
                object_lpips_size,
                object_lpips_max_boxes,
            )
            loss = loss + float(object_lpips_loss_weight) * object_lpips
            record_loss("loss_object_lpips", object_lpips)
            record_loss("loss_object_lpips_contrib", float(object_lpips_loss_weight) * object_lpips)
        if foreground_lpips_loss_weight > 0.0 and x0_image_target is not None and foreground_image_loss_mask is not None and object_lpips_model is not None and image_decoder is not None:
            foreground_lpips = self._foreground_lpips_loss(
                pred_image,
                x0_image_target,
                foreground_image_loss_mask,
                object_lpips_model,
                foreground_lpips_padding,
                foreground_lpips_size,
            )
            loss = loss + float(foreground_lpips_loss_weight) * foreground_lpips
            record_loss("loss_foreground_lpips", foreground_lpips)
            record_loss("loss_foreground_lpips_contrib", float(foreground_lpips_loss_weight) * foreground_lpips)
        # loss = F.mse_loss(target, model_out)
        record_loss("loss_total", loss)
        self.last_loss_metrics = loss_metrics
        if return_outputs:
            return {
                "loss": loss,
                "model_out": model_out,
                "loss_raw": loss_raw,
                "pred_x0": pred_x0,
                "pred_image": pred_image,
                "lidar_depth_pred": lidar_depth_pred,
                "loss_metrics": loss_metrics,
            }
        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond_init_grd=None, cond_sat=None):
        model_out = self.denoise_model(x, t, cond_init_grd=cond_init_grd, cond_sat=cond_sat)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, cond_init_grd=None, cond_sat=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond_init_grd=cond_init_grd, cond_sat=cond_sat)
        noise = torch.randn_like(x, device = x.device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def sample(self, x_start, cond_init_grd=None, cond_sat=None):
        B,C,H,W = x_start.size()
        img = torch.randn_like(x_start, device = x_start.device)
        intermediates = [img]
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((B,), i, device=x_start.device, dtype=torch.long),
                                clip_denoised=self.clip_denoised, cond_init_grd=cond_init_grd, cond_sat=cond_sat)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        return intermediates



    
