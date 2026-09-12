import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from models.eval.evaluate import Evaluate_indic
from models.eval.dynamic_metrics import dynamic_masked_metrics
from utils.util import instantiate_from_config
from torchvision import transforms
from collections import OrderedDict

# from models.CVUSA_geo_ldm_diffusion.openaimodel import AttentionBlock
# from ldm.modules.CVUSA_attention import CrossAttention

from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler
import random
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def save_img(img, save_path):
    img = transforms.functional.to_pil_image(img, mode='RGB')
    img.save(save_path)

class Boost_Sat2Den_ddpm(pl.LightningModule):
    def __init__(self,
                # Sat2Den_config,
                # Sat2Den_ckpt_path,
                AE_config,
                AE_ckpt_path,
                DDPM_config,
                Condition_config_grd,
                Condition_config_sat,
                scale_factor,
                pre_sat2grd_model_path,
                Condition_config_txt,
                # control_txt,
                pre_ldm_model_path,
                use_lidar_cond=False,
                lidar_condition_key="lidar_cond",
                freeze_for_lidar_control=False,
                dynamic_loss_weight=0.0,
                dynamic_x0_loss_weight=0.0,
                dynamic_image_loss_weight=0.0,
                dynamic_crop_image_loss_weight=0.0,
                dynamic_crop_padding=8,
                dynamic_point_loss_weight=0.0,
                dynamic_point_x0_loss_weight=0.0,
                dynamic_point_image_loss_weight=0.0,
                dynamic_point_dilation=4,
                dynamic_object_lpips_weight=0.0,
                dynamic_object_lpips_padding=6,
                dynamic_object_lpips_size=64,
                dynamic_object_lpips_max_boxes=4,
                dynamic_mask_key="dynamic_mask",
                foreground_mask_key="foreground_mask",
                foreground_loss_weight=0.0,
                foreground_x0_loss_weight=0.0,
                foreground_image_loss_weight=0.0,
                foreground_lpips_loss_weight=0.0,
                foreground_lpips_padding=8,
                foreground_lpips_size=96,
                foreground_lidar_intersection=False,
                lidar_counterfactual_weight=0.0,
                lidar_counterfactual_margin=0.02,
                lidar_counterfactual_probes="zero",
                lidar_counterfactual_stop_negative=True,
                lidar_counterfactual_separation_weight=0.0,
                lidar_counterfactual_point_fallback=True,
                lidar_counterfactual_exist_weight=1.0,
                lidar_depth_loss_weight=0.0,
                lidar_depth_output_scale=1.0,
                lidar_depth_bottleneck_scale=1.0,
                lidar_depth_log_eps=1e-3,
                lidar_depth_resample_mode="masked_area",
                lidar_semantic_alignment_weight=0.0,
                lidar_semantic_alignment_key="image_semantic_feat",
                lidar_semantic_alignment_mask_mode="all",
                lidar_semantic_contrast_weight=0.0,
                lidar_semantic_contrast_margin=0.1,
                lidar_zero_reconstruction_loss_weight=0.0,
                lidar_zero_reconstruction_mask_mode="all",
                static_teacher_consistency_weight=0.0,
                static_teacher_gate_channel=-1,
                static_teacher_mask_mode="no_lidar",
                ray_evidence_mask_mode="foreground",
                dynamic_class_token_weight=0.0,
                dynamic_class_hist_key="dynamic_class_hist",
                dynamic_class_token_count=8,
                dynamic_class_token_dim=768,
                satellite_condition_dropout_prob=0.0,
                lidar_train_sat_condition=False,
                lidar_sat_lr_scale=1.0,
                lidar_unfreeze_denoise_all=False,
                lidar_unfreeze_output_blocks=0,
                lidar_unfreeze_out=False,
                lidar_unfreeze_transformers="none",
                lidar_unet_lr_scale=0.25,
                lidar_unet_new_lr_scale=1.0,
                lidar_geom_mode="raw",
                Lidar_context_config=None,
                use_lidar_control_residual=False,
                lidar_context_lr_scale=1.0,
                # lossconfig
                 ):
        super().__init__()

        self.pre_AE_model = self.init_AE(AE_config, AE_ckpt_path)
        self.DDPM = instantiate_from_config(DDPM_config)
        if pre_ldm_model_path:
            pre_ldm_model = torch.load(pre_ldm_model_path, map_location="cpu")
            self.load_pre_ldm_model(pre_ldm_model['state_dict'])

        self.condition_model_sat = instantiate_from_config(Condition_config_sat)
        self.lidar_context_model = instantiate_from_config(Lidar_context_config) if Lidar_context_config else None
        self.scale_factor = scale_factor
        self.use_lidar_cond = use_lidar_cond
        self.lidar_condition_key = lidar_condition_key
        self.use_lidar_control_residual = False
        self.freeze_for_lidar_control = freeze_for_lidar_control
        self.dynamic_loss_weight = dynamic_loss_weight
        self.dynamic_x0_loss_weight = dynamic_x0_loss_weight
        self.dynamic_image_loss_weight = dynamic_image_loss_weight
        self.dynamic_crop_image_loss_weight = dynamic_crop_image_loss_weight
        self.dynamic_crop_padding = dynamic_crop_padding
        self.dynamic_point_loss_weight = dynamic_point_loss_weight
        self.dynamic_point_x0_loss_weight = dynamic_point_x0_loss_weight
        self.dynamic_point_image_loss_weight = dynamic_point_image_loss_weight
        self.dynamic_point_dilation = dynamic_point_dilation
        self.dynamic_object_lpips_weight = dynamic_object_lpips_weight
        self.dynamic_object_lpips_padding = dynamic_object_lpips_padding
        self.dynamic_object_lpips_size = dynamic_object_lpips_size
        self.dynamic_object_lpips_max_boxes = dynamic_object_lpips_max_boxes
        self.dynamic_mask_key = dynamic_mask_key
        self.foreground_mask_key = foreground_mask_key
        self.foreground_loss_weight = float(foreground_loss_weight)
        self.foreground_x0_loss_weight = float(foreground_x0_loss_weight)
        self.foreground_image_loss_weight = float(foreground_image_loss_weight)
        self.foreground_lpips_loss_weight = float(foreground_lpips_loss_weight)
        self.foreground_lpips_padding = int(foreground_lpips_padding)
        self.foreground_lpips_size = int(foreground_lpips_size)
        self.foreground_lidar_intersection = bool(foreground_lidar_intersection)
        self.lidar_counterfactual_weight = float(lidar_counterfactual_weight)
        self.lidar_counterfactual_margin = float(lidar_counterfactual_margin)
        self.lidar_counterfactual_probes = str(lidar_counterfactual_probes or "")
        self.lidar_counterfactual_stop_negative = bool(lidar_counterfactual_stop_negative)
        self.lidar_counterfactual_separation_weight = float(lidar_counterfactual_separation_weight)
        self.lidar_counterfactual_point_fallback = bool(lidar_counterfactual_point_fallback)
        self.lidar_counterfactual_exist_weight = float(lidar_counterfactual_exist_weight)
        self.last_lidar_counterfactual_metrics = {}
        self.lidar_depth_loss_weight = float(lidar_depth_loss_weight)
        self.lidar_depth_output_scale = float(lidar_depth_output_scale)
        self.lidar_depth_bottleneck_scale = float(lidar_depth_bottleneck_scale)
        self.lidar_depth_log_eps = float(lidar_depth_log_eps)
        self.lidar_depth_resample_mode = str(lidar_depth_resample_mode or "masked_area")
        if self.lidar_depth_resample_mode not in {"legacy_nearest", "masked_area"}:
            raise ValueError(
                "lidar_depth_resample_mode must be either "
                f"'legacy_nearest' or 'masked_area', got {self.lidar_depth_resample_mode!r}"
            )
        self.lidar_semantic_alignment_weight = float(lidar_semantic_alignment_weight)
        self.lidar_semantic_alignment_key = str(lidar_semantic_alignment_key or "image_semantic_feat")
        self.lidar_semantic_alignment_mask_mode = str(lidar_semantic_alignment_mask_mode or "all")
        if self.lidar_semantic_alignment_mask_mode not in {"all", "lidar_hit", "foreground", "foreground_lidar"}:
            raise ValueError(
                "lidar_semantic_alignment_mask_mode must be one of "
                "'all', 'lidar_hit', 'foreground', 'foreground_lidar', "
                f"got {self.lidar_semantic_alignment_mask_mode!r}"
            )
        self.lidar_semantic_contrast_weight = float(lidar_semantic_contrast_weight)
        self.lidar_semantic_contrast_margin = float(lidar_semantic_contrast_margin)
        self.last_lidar_semantic_alignment_metrics = {}
        self.lidar_zero_reconstruction_loss_weight = float(lidar_zero_reconstruction_loss_weight)
        self.lidar_zero_reconstruction_mask_mode = str(lidar_zero_reconstruction_mask_mode or "all")
        if self.lidar_zero_reconstruction_mask_mode not in {"all", "background"}:
            raise ValueError(
                "lidar_zero_reconstruction_mask_mode must be 'all' or 'background', "
                f"got {self.lidar_zero_reconstruction_mask_mode!r}"
            )
        self.satellite_condition_dropout_prob = float(satellite_condition_dropout_prob)
        self.last_satellite_condition_dropout_metrics = {}
        self.static_teacher_consistency_weight = float(static_teacher_consistency_weight)
        self.static_teacher_gate_channel = int(static_teacher_gate_channel)
        self.static_teacher_mask_mode = str(static_teacher_mask_mode or "no_lidar")
        if self.static_teacher_mask_mode not in {"no_lidar", "background"}:
            raise ValueError(
                "static_teacher_mask_mode must be 'no_lidar' or 'background', "
                f"got {self.static_teacher_mask_mode!r}"
            )
        self.ray_evidence_mask_mode = str(ray_evidence_mask_mode or "foreground")
        if self.ray_evidence_mask_mode not in {"none", "foreground", "lidar_hit"}:
            raise ValueError(
                "ray_evidence_mask_mode must be 'none', 'foreground', or 'lidar_hit', "
                f"got {self.ray_evidence_mask_mode!r}"
            )
        self.DDPM.ray_evidence_mask_mode = self.ray_evidence_mask_mode
        self.dynamic_class_token_weight = dynamic_class_token_weight
        self.dynamic_class_hist_key = dynamic_class_hist_key
        if dynamic_class_token_weight > 0.0:
            self.dynamic_class_tokens = nn.Parameter(torch.zeros(dynamic_class_token_count, dynamic_class_token_dim))
        else:
            self.dynamic_class_tokens = None
        self.lidar_train_sat_condition = bool(lidar_train_sat_condition)
        self.lidar_sat_lr_scale = float(lidar_sat_lr_scale)
        self.lidar_unfreeze_denoise_all = bool(lidar_unfreeze_denoise_all)
        self.lidar_unfreeze_output_blocks = int(lidar_unfreeze_output_blocks)
        self.lidar_unfreeze_out = bool(lidar_unfreeze_out)
        self.lidar_unfreeze_transformers = str(lidar_unfreeze_transformers or "none")
        self.lidar_unet_lr_scale = float(lidar_unet_lr_scale)
        self.lidar_unet_new_lr_scale = float(lidar_unet_new_lr_scale)
        self.lidar_context_lr_scale = float(lidar_context_lr_scale)
        self.lidar_geom_mode = str(lidar_geom_mode or "raw")
        if self.lidar_geom_mode not in {"raw", "ray_depth", "ray_depth_inv"}:
            raise ValueError(f"unsupported lidar_geom_mode: {self.lidar_geom_mode}")
        if pre_sat2grd_model_path:
            pre_sat2grd_model = torch.load(pre_sat2grd_model_path, map_location="cpu")
            self.load_pre_sat2grd_model(pre_sat2grd_model['state_dict'])
            
        self.evaluate = Evaluate_indic()
        for param in self.evaluate.loss_fn_alex.parameters():
            param.requires_grad = False
        self.evaluate.loss_fn_alex.eval()
        self.RMSE = []
        self.SSIM = []
        self.PSNR = []
        self.SD = []
        self.P_alex = []
        self.P_squeeze = []

        self.res_RMSE = []
        self.res_SSIM = []
        self.res_PSNR = []
        self.res_SD = []
        self.res_P_alex = []
        self.res_P_squeeze = []

        self.gt_res_RMSE = []
        self.gt_res_SSIM = []
        self.gt_res_PSNR = []
        self.gt_res_SD = []
        self.gt_res_P_alex = []
        self.gt_res_P_squeeze = []
        self.dynamic_PSNR = []
        self.dynamic_SSIM = []
        self.dynamic_LPIPS = []

    def latent_dynamic_mask(self, dynamic_mask, latent_shape):
        if dynamic_mask is None:
            return None
        mask = dynamic_mask.float()
        if mask.ndim == 3:
            mask = mask[:, None]
        target_h, target_w = latent_shape[-2:]
        if mask.shape[-2] % target_h == 0 and mask.shape[-1] % target_w == 0:
            kernel = (mask.shape[-2] // target_h, mask.shape[-1] // target_w)
            return F.max_pool2d(mask, kernel_size=kernel, stride=kernel).clamp(0.0, 1.0)
        return F.interpolate(mask, size=(target_h, target_w), mode="area").clamp(0.0, 1.0)

    def dynamic_crop_mask(self, dynamic_mask, image_shape):
        if dynamic_mask is None:
            return None
        mask = dynamic_mask.float()
        if mask.ndim == 3:
            mask = mask[:, None]
        target_h, target_w = image_shape[-2:]
        if mask.shape[-2:] != (target_h, target_w):
            mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
        padding = int(self.dynamic_crop_padding)
        if padding > 0:
            kernel = 2 * padding + 1
            mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=padding)
        return mask.clamp(0.0, 1.0)

    def dynamic_point_mask(self, lidar_cond, latent_shape):
        if lidar_cond is None or lidar_cond.shape[1] < 2:
            return None
        point_channel = 7 if lidar_cond.shape[1] >= 8 else 1
        mask = (lidar_cond[:, point_channel : point_channel + 1].float() > 0.0).float()
        dilation = int(self.dynamic_point_dilation)
        if dilation > 0:
            kernel = 2 * dilation + 1
            mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
        return self.latent_dynamic_mask(mask, latent_shape)

    def dynamic_point_image_mask(self, lidar_cond, image_shape):
        if lidar_cond is None or lidar_cond.shape[1] < 2:
            return None
        point_channel = 7 if lidar_cond.shape[1] >= 8 else 1
        mask = (lidar_cond[:, point_channel : point_channel + 1].float() > 0.0).float()
        target_h, target_w = image_shape[-2:]
        if mask.shape[-2:] != (target_h, target_w):
            mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
        dilation = int(self.dynamic_point_dilation)
        if dilation > 0:
            kernel = 2 * dilation + 1
            mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
        return mask.clamp(0.0, 1.0)

    def lidar_depth_target_mask(self, lidar_cond, latent_shape):
        if lidar_cond is None or lidar_cond.shape[1] < 3:
            return None, None
        hit = lidar_cond[:, 1:2].float().clamp(0.0, 1.0)
        depth = lidar_cond[:, 2:3].float().clamp(0.0, 1.0) * hit
        target_h, target_w = latent_shape[-2:]
        if hit.shape[-2] % target_h == 0 and hit.shape[-1] % target_w == 0:
            kernel = (hit.shape[-2] // target_h, hit.shape[-1] // target_w)
            hit_avg = F.avg_pool2d(hit, kernel_size=kernel, stride=kernel)
            depth_avg = F.avg_pool2d(depth, kernel_size=kernel, stride=kernel)
        else:
            hit_avg = F.interpolate(hit, size=(target_h, target_w), mode="area")
            depth_avg = F.interpolate(depth, size=(target_h, target_w), mode="area")
        mask = (hit_avg > 0.0).to(depth_avg.dtype)
        target = (depth_avg / hit_avg.clamp_min(1e-6)).clamp(0.0, 1.0)
        return target * mask, mask

    def static_teacher_loss_mask(self, lidar_cond, latent_shape, foreground_loss_mask=None):
        if self.static_teacher_consistency_weight <= 0.0:
            return None
        if self.static_teacher_mask_mode == "background" and foreground_loss_mask is not None:
            if foreground_loss_mask.shape[-2:] != latent_shape[-2:]:
                foreground_loss_mask = F.interpolate(
                    foreground_loss_mask.float(),
                    size=latent_shape[-2:],
                    mode="area",
                )
            return (1.0 - foreground_loss_mask.float()).clamp(0.0, 1.0)
        if (
            lidar_cond is None
            or self.static_teacher_gate_channel < 0
            or self.static_teacher_gate_channel >= lidar_cond.shape[1]
        ):
            return None
        gate = lidar_cond[:, self.static_teacher_gate_channel : self.static_teacher_gate_channel + 1].float()
        gate = gate.clamp(0.0, 1.0)
        if gate.shape[-2:] != latent_shape[-2:]:
            gate = F.interpolate(gate, size=latent_shape[-2:], mode="area")
        return (1.0 - gate).clamp(0.0, 1.0)

    def foreground_image_mask(self, batch, lidar_cond, image_shape):
        if self.foreground_mask_key not in batch:
            return None
        mask = batch[self.foreground_mask_key].float()
        if mask.ndim == 3:
            mask = mask[:, None]
        target_h, target_w = image_shape[-2:]
        if mask.shape[-2:] != (target_h, target_w):
            mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
        mask = mask.clamp(0.0, 1.0)
        if self.foreground_lidar_intersection:
            lidar_mask = self.dynamic_point_image_mask(lidar_cond, image_shape)
            if lidar_mask is not None:
                mask = mask * lidar_mask.to(device=mask.device, dtype=mask.dtype)
        return mask.clamp(0.0, 1.0)

    def append_dynamic_class_token(self, cond_label, batch=None):
        if self.dynamic_class_tokens is None or self.dynamic_class_token_weight <= 0.0:
            return cond_label
        if batch is None or self.dynamic_class_hist_key not in batch:
            return cond_label
        hist = batch[self.dynamic_class_hist_key].to(cond_label.device).float()
        if hist.ndim == 1:
            hist = hist[None]
        token_count = self.dynamic_class_tokens.shape[0]
        if hist.shape[-1] < token_count:
            hist = F.pad(hist, (0, token_count - hist.shape[-1]))
        hist = hist[:, :token_count]
        if float(hist.sum().detach().cpu()) <= 0.0:
            return cond_label
        weights = (hist / hist.sum(dim=-1, keepdim=True).clamp_min(1.0)).to(cond_label.dtype)
        tokens = self.dynamic_class_tokens.to(device=cond_label.device, dtype=cond_label.dtype)
        class_token = weights @ tokens
        class_token = class_token[:, None, :] * float(self.dynamic_class_token_weight)
        return cond_label + class_token.type_as(cond_label)

    def make_condition(self, inputs, batch=None):
        cond_label = self.condition_model_sat(inputs)
        cond_label = cond_label[:, 1:, :]
        return self.append_dynamic_class_token(cond_label, batch)

    def apply_satellite_condition_dropout(self, cond_label):
        metrics = {
            "satellite_condition_dropout_prob": float(self.satellite_condition_dropout_prob),
            "satellite_condition_dropout_applied": 0.0,
            "satellite_condition_dropout_fraction": 0.0,
        }
        prob = max(0.0, min(1.0, float(self.satellite_condition_dropout_prob)))
        if (not self.training) or prob <= 0.0 or cond_label is None:
            self.last_satellite_condition_dropout_metrics = metrics
            return cond_label
        drop_mask = (torch.rand(cond_label.shape[0], 1, 1, device=cond_label.device) < prob).to(
            dtype=cond_label.dtype
        )
        dropped = cond_label * (1.0 - drop_mask)
        metrics.update(
            {
                "satellite_condition_dropout_applied": 1.0,
                "satellite_condition_dropout_fraction": float(drop_mask.detach().float().mean().cpu()),
            }
        )
        self.last_satellite_condition_dropout_metrics = metrics
        return dropped

    def build_lidar_geom_cond(self, lidar_cond):
        if lidar_cond is None or self.lidar_geom_mode == "raw":
            return lidar_cond
        cond = lidar_cond.float()
        b, _, h, w = cond.shape
        zero = cond.new_zeros((b, 1, h, w))
        hit = cond[:, 1:2].clamp(0.0, 1.0) if cond.shape[1] > 1 else zero
        depth = cond[:, 2:3].clamp(0.0, 1.0) * hit if cond.shape[1] > 2 else zero
        if cond.shape[1] >= 7:
            xyz = cond[:, 4:7].float()
            z = xyz[:, 2:3].abs().clamp_min(1e-4)
            ray_x = (xyz[:, 0:1] / z).clamp(-2.0, 2.0) * 0.5
            ray_y = (xyz[:, 1:2] / z).clamp(-2.0, 2.0) * 0.5
        else:
            ray_x = zero
            ray_y = zero
        ray_x = ray_x.clamp(-1.0, 1.0) * hit
        ray_y = ray_y.clamp(-1.0, 1.0) * hit
        log_depth = (torch.log1p(79.0 * depth.clamp_min(0.0)) / np.log(80.0)).clamp(0.0, 1.0) * hit
        if self.lidar_geom_mode == "ray_depth":
            return torch.cat([hit, ray_x, ray_y, log_depth], dim=1).to(dtype=lidar_cond.dtype)
        inv_depth = (1.0 - depth).clamp(0.0, 1.0) * hit
        return torch.cat([hit, ray_x, ray_y, log_depth, inv_depth], dim=1).to(dtype=lidar_cond.dtype)

    def make_lidar_context(
        self,
        lidar_cond,
        range_img=None,
        range_mask=None,
        camera_to_lidar=None,
        left_camera_k=None,
        lidar_points=None,
        lidar_points_mask=None,
        lidar_point_features=None,
        lidar_point_features_mask=None,
        lidar_ray_features=None,
        lidar_ray_features_mask=None,
        lidar_pixel_features=None,
        lidar_pixel_features_mask=None,
        lidar_pixel_features_available=None,
    ):
        if self.lidar_context_model is None or lidar_cond is None:
            return None
        lidar_geom_cond = self.build_lidar_geom_cond(lidar_cond)
        context_kwargs = {
            "raw_lidar_cond": lidar_cond,
            "range_img": range_img,
            "range_mask": range_mask,
            "camera_k": left_camera_k,
            "camera_to_lidar": camera_to_lidar,
            "lidar_points": lidar_points,
            "lidar_points_mask": lidar_points_mask,
            "lidar_point_features": lidar_point_features,
            "lidar_point_features_mask": lidar_point_features_mask,
            "lidar_ray_features": lidar_ray_features,
            "lidar_ray_features_mask": lidar_ray_features_mask,
        }
        uses_pixel_features = bool(getattr(self.lidar_context_model, "uses_pixel_features", False))
        if self.lidar_context_model.__class__.__name__ == "LidarPixelConditionEncoder":
            uses_pixel_features = True
        if uses_pixel_features:
            context_kwargs.update(
                {
                    "lidar_pixel_features": lidar_pixel_features,
                    "lidar_pixel_features_mask": lidar_pixel_features_mask,
                    "lidar_pixel_features_available": lidar_pixel_features_available,
                }
            )
        return self.lidar_context_model(lidar_geom_cond, **context_kwargs)

    def get_optional_lidar_batch_tensor(self, batch, key, device):
        if not self.use_lidar_cond or key not in batch:
            return None
        return batch[key].to(device).float()

    def lidar_semantic_alignment_loss(
        self,
        semantic_pred_tokens,
        batch,
        lidar_cond,
        foreground_image_loss_mask,
    ):
        metrics = {
            "lidar_semantic_alignment_applied": 0,
            "lidar_semantic_alignment_loss": 0.0,
            "lidar_semantic_alignment_contrib": 0.0,
            "lidar_semantic_alignment_mask_coverage": 0.0,
            "lidar_semantic_alignment_target_available": 0.0,
            "lidar_semantic_contrast_loss": 0.0,
            "lidar_semantic_contrast_contrib": 0.0,
        }
        if self.lidar_semantic_alignment_weight <= 0.0:
            self.last_lidar_semantic_alignment_metrics = metrics
            return None
        if semantic_pred_tokens is None or self.lidar_semantic_alignment_key not in batch:
            self.last_lidar_semantic_alignment_metrics = metrics
            return None

        pred = semantic_pred_tokens
        target = batch[self.lidar_semantic_alignment_key].to(device=pred.device, dtype=pred.dtype)
        if target.ndim == 4:
            if target.shape[1] == pred.shape[-1]:
                target_map = target
            elif target.shape[-1] == pred.shape[-1]:
                target_map = target.permute(0, 3, 1, 2).contiguous()
            else:
                self.last_lidar_semantic_alignment_metrics = metrics
                return None
            grid_h, grid_w = self.lidar_context_model.token_grid
            if target_map.shape[-2:] != (grid_h, grid_w):
                target_map = F.interpolate(target_map, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
            target_tokens = target_map.flatten(2).transpose(1, 2)
        elif target.ndim == 3:
            target_tokens = target
            if target_tokens.shape[1] != pred.shape[1] and target_tokens.shape[2] == pred.shape[1]:
                target_tokens = target_tokens.transpose(1, 2)
            if target_tokens.shape[1] != pred.shape[1] or target_tokens.shape[2] != pred.shape[2]:
                self.last_lidar_semantic_alignment_metrics = metrics
                return None
        else:
            self.last_lidar_semantic_alignment_metrics = metrics
            return None

        mask = pred.new_ones((pred.shape[0], pred.shape[1], 1))
        if "image_semantic_available" in batch:
            available = batch["image_semantic_available"].to(device=pred.device, dtype=pred.dtype).view(pred.shape[0], -1)
            mask = mask * available[:, :1].unsqueeze(1)
            metrics["lidar_semantic_alignment_target_available"] = float(available[:, 0].detach().mean().cpu())
        if "image_semantic_mask" in batch:
            sem_mask = batch["image_semantic_mask"].to(device=pred.device, dtype=pred.dtype)
            if sem_mask.ndim == 3:
                sem_mask = sem_mask.unsqueeze(1)
            grid_h, grid_w = self.lidar_context_model.token_grid
            sem_mask = F.interpolate(sem_mask.float(), size=(grid_h, grid_w), mode="nearest").flatten(2).transpose(1, 2)
            mask = mask * sem_mask.clamp(0.0, 1.0)

        hit_tokens = None
        if lidar_cond is not None:
            hit = lidar_cond[:, 1:2].float().to(device=pred.device, dtype=pred.dtype)
            grid_h, grid_w = self.lidar_context_model.token_grid
            hit_tokens = F.adaptive_max_pool2d(hit, (grid_h, grid_w)).flatten(2).transpose(1, 2).clamp(0.0, 1.0)
        fg_tokens = None
        if foreground_image_loss_mask is not None:
            grid_h, grid_w = self.lidar_context_model.token_grid
            fg_tokens = F.interpolate(
                foreground_image_loss_mask.to(device=pred.device, dtype=pred.dtype),
                size=(grid_h, grid_w),
                mode="nearest",
            ).flatten(2).transpose(1, 2).clamp(0.0, 1.0)

        if self.lidar_semantic_alignment_mask_mode == "lidar_hit" and hit_tokens is not None:
            mask = mask * hit_tokens
        elif self.lidar_semantic_alignment_mask_mode == "foreground" and fg_tokens is not None:
            mask = mask * fg_tokens
        elif self.lidar_semantic_alignment_mask_mode == "foreground_lidar":
            if fg_tokens is not None:
                mask = mask * fg_tokens
            if hit_tokens is not None:
                mask = mask * hit_tokens

        mask_sum = mask.sum().clamp_min(1.0)
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target_tokens.detach(), dim=-1)
        align_raw = 1.0 - (pred_norm * target_norm).sum(dim=-1, keepdim=True)
        align_loss = (align_raw * mask).sum() / mask_sum
        contrast_loss = pred.new_zeros(())
        if self.lidar_semantic_contrast_weight > 0.0 and fg_tokens is not None:
            fg_mask = (mask * fg_tokens).clamp(0.0, 1.0)
            bg_mask = (mask * (1.0 - fg_tokens)).clamp(0.0, 1.0)
            if float(fg_mask.detach().sum().cpu()) > 1.0 and float(bg_mask.detach().sum().cpu()) > 1.0:
                fg_target = F.normalize((target_norm * fg_mask).sum(dim=1) / fg_mask.sum(dim=1).clamp_min(1.0), dim=-1)
                bg_target = F.normalize((target_norm * bg_mask).sum(dim=1) / bg_mask.sum(dim=1).clamp_min(1.0), dim=-1)
                fg_sim = (pred_norm * fg_target.unsqueeze(1)).sum(dim=-1, keepdim=True)
                fg_bg_sim = (pred_norm * bg_target.unsqueeze(1)).sum(dim=-1, keepdim=True)
                contrast_raw = F.relu(float(self.lidar_semantic_contrast_margin) + fg_bg_sim - fg_sim)
                contrast_loss = (contrast_raw * fg_mask).sum() / fg_mask.sum().clamp_min(1.0)
        total = float(self.lidar_semantic_alignment_weight) * align_loss
        total = total + float(self.lidar_semantic_contrast_weight) * contrast_loss
        metrics.update(
            {
                "lidar_semantic_alignment_applied": 1,
                "lidar_semantic_alignment_loss": float(align_loss.detach().cpu()),
                "lidar_semantic_alignment_contrib": float(
                    (float(self.lidar_semantic_alignment_weight) * align_loss).detach().cpu()
                ),
                "lidar_semantic_alignment_mask_coverage": float(mask.detach().float().mean().cpu()),
                "lidar_semantic_contrast_loss": float(contrast_loss.detach().cpu()),
                "lidar_semantic_contrast_contrib": float(
                    (float(self.lidar_semantic_contrast_weight) * contrast_loss).detach().cpu()
                ),
            }
        )
        self.last_lidar_semantic_alignment_metrics = metrics
        return total

    def make_lidar_evidence(self, lidar_cond):
        if self.lidar_context_model is None or lidar_cond is None:
            return None
        if hasattr(self.lidar_context_model, "make_evidence_maps"):
            return self.lidar_context_model.make_evidence_maps(lidar_cond)
        return None

    def make_lidar_geometry_mask(self, lidar_evidence, foreground_loss_mask=None):
        mode = str(getattr(self.DDPM, "ray_evidence_mask_mode", "lidar_hit") or "lidar_hit")
        if mode == "none":
            return None
        if mode == "lidar_hit" and lidar_evidence is not None and lidar_evidence.shape[1] > 1:
            return lidar_evidence[:, 1:2]
        if mode == "foreground":
            return foreground_loss_mask
        raise ValueError(f"Unsupported or unavailable LiDAR geometry mask mode: {mode}")

    def lidar_counterfactual_probe_names(self):
        probes = [item.strip() for item in self.lidar_counterfactual_probes.split(",") if item.strip()]
        unsupported = [probe for probe in probes if probe != "zero"]
        if unsupported:
            raise ValueError(
                "LiDAR counterfactual training now supports only the zero probe; "
                f"remove unsupported probes: {', '.join(unsupported)}"
            )
        return probes

    def apply_lidar_counterfactual_probe(self, tensor, probe):
        if tensor is None:
            return None
        if probe == "zero":
            return torch.zeros_like(tensor)
        raise ValueError(f"Unsupported LiDAR counterfactual probe: {probe}")

    def lidar_counterfactual_mask(self, foreground_image_loss_mask, point_image_loss_mask):
        mask = None
        if foreground_image_loss_mask is not None and float(foreground_image_loss_mask.detach().sum().cpu()) > 0.0:
            mask = foreground_image_loss_mask
        elif (
            self.lidar_counterfactual_point_fallback
            and point_image_loss_mask is not None
            and float(point_image_loss_mask.detach().sum().cpu()) > 0.0
        ):
            mask = point_image_loss_mask
        return mask

    def masked_image_l1(self, pred_image, target_image, mask):
        loss_raw = F.l1_loss(pred_image, target_image, reduction="none")
        return self.DDPM._masked_loss_mean(loss_raw, mask)

    def masked_depth_log_l1(self, pred_depth, target_depth, mask):
        if pred_depth is None or target_depth is None or mask is None:
            return None
        pred = pred_depth.float().clamp(1e-6, 1.0)
        target = target_depth.float()
        if target.ndim == 3:
            target = target[:, None]
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
        target = target.to(device=pred.device, dtype=pred.dtype).clamp(1e-6, 1.0)
        prepared_mask = self.DDPM._prepare_loss_mask(mask, pred)
        eps = max(float(self.lidar_depth_log_eps), 1e-6)
        loss_raw = (torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps))).abs()
        return self.DDPM._masked_loss_mean(loss_raw, prepared_mask)

    def snapshot_lidar_monitor_state(self):
        state = {"context": {}, "attention": []}
        context_model = getattr(self, "lidar_context_model", None)
        if context_model is not None:
            for name in [
                "last_valid_sample_ratio",
                "last_hit_coverage",
                "last_empty_coverage",
                "last_pointmap_coverage",
                "last_token_mean_norm",
                "last_token_centered_norm",
                "last_token_centered_to_mean_ratio",
                "last_token_var_mean",
                "last_token_proj_bias_norm",
                "last_token_output_mean_norm",
                "last_token_output_centered_norm",
                "last_token_output_centered_to_mean_ratio",
                "last_token_output_var_mean",
                "last_point_valid_ratio",
                "last_point_count_mean",
                "last_ray_token_coverage",
            ]:
                value = getattr(context_model, name, None)
                if torch.is_tensor(value):
                    state["context"][name] = value.detach().clone()
        for module in self.DDPM.denoise_model.modules():
            if not hasattr(module, "last_attn_entropy_norm") and not hasattr(module, "last_evidence_sat_weight"):
                continue
            module_state = {}
            for name in [
                "last_sim_std_mean",
                "last_sim_range_mean",
                "last_attn_entropy_norm",
                "last_attn_max_mean",
                "last_attn_std_mean",
                "last_attn_token_count",
                "last_attn_query_count",
                "last_evidence_sat_weight",
                "last_evidence_lidar_weight",
                "last_evidence_null_weight",
                "last_evidence_entropy_norm",
                "last_evidence_lidar_mask_mean",
                "last_evidence_lidar_weight_masked",
                "last_evidence_lidar_weight_background",
            ]:
                if not hasattr(module, name):
                    continue
                value = getattr(module, name)
                module_state[name] = value.detach().clone() if torch.is_tensor(value) else value
            state["attention"].append((module, module_state))
        return state

    def restore_lidar_monitor_state(self, state):
        context_model = getattr(self, "lidar_context_model", None)
        if context_model is not None:
            for name, value in state.get("context", {}).items():
                target = getattr(context_model, name, None)
                if torch.is_tensor(target):
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
        for module, module_state in state.get("attention", []):
            for name, value in module_state.items():
                target = getattr(module, name, None)
                if torch.is_tensor(target) and torch.is_tensor(value):
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
                else:
                    setattr(module, name, value)

    def lidar_counterfactual_loss(
        self,
        normal_outputs,
        pre_residual_laten,
        t,
        noise,
        outputs,
        cond_label,
        lidar_cond,
        range_img,
        range_mask,
        camera_to_lidar,
        left_camera_k,
        lidar_points,
        lidar_points_mask,
        lidar_point_features,
        lidar_point_features_mask,
        lidar_ray_features,
        lidar_ray_features_mask,
        lidar_pixel_features,
        lidar_pixel_features_mask,
        lidar_pixel_features_available,
        gt_shift_x,
        gt_shift_y,
        theta,
        foreground_image_loss_mask,
        point_image_loss_mask,
        lidar_depth_target,
        lidar_depth_mask,
    ):
        metrics = {
            "lidar_counterfactual_applied": 0,
            "lidar_counterfactual_mask_coverage": 0.0,
            "lidar_counterfactual_normal_l1": 0.0,
            "lidar_counterfactual_zero_l1": 0.0,
            "lidar_counterfactual_normal_depth_log_l1": 0.0,
            "lidar_counterfactual_zero_depth_log_l1": 0.0,
            "lidar_counterfactual_zero_minus_normal_depth_log_l1": 0.0,
            "lidar_counterfactual_exist_loss": 0.0,
            "lidar_counterfactual_rank_loss": 0.0,
            "lidar_counterfactual_separation_loss": 0.0,
            "lidar_counterfactual_total_contrib": 0.0,
            "lidar_zero_reconstruction_loss": 0.0,
            "lidar_zero_reconstruction_loss_contrib": 0.0,
            "lidar_zero_reconstruction_mask_coverage": 0.0,
        }
        probes = self.lidar_counterfactual_probe_names()
        needs_rgb_counterfactual = self.lidar_counterfactual_weight > 0.0 and "zero" in probes
        needs_zero_reconstruction = self.lidar_zero_reconstruction_loss_weight > 0.0 and "zero" in probes
        if not needs_rgb_counterfactual and not needs_zero_reconstruction:
            self.last_lidar_counterfactual_metrics = metrics
            return pre_residual_laten.new_tensor(0.0)
        if lidar_cond is None:
            self.last_lidar_counterfactual_metrics = metrics
            return pre_residual_laten.new_tensor(0.0)
        mask = self.lidar_counterfactual_mask(foreground_image_loss_mask, point_image_loss_mask)
        if mask is None:
            self.last_lidar_counterfactual_metrics = metrics
            return pre_residual_laten.new_tensor(0.0)

        normal_pred = normal_outputs["pred_image"]
        normal_l1 = (
            self.masked_image_l1(normal_pred, outputs, mask)
            if needs_rgb_counterfactual and normal_pred is not None
            else pre_residual_laten.new_tensor(0.0)
        )
        normal_monitor_state = self.snapshot_lidar_monitor_state()

        zero_lidar_cond = self.apply_lidar_counterfactual_probe(lidar_cond, "zero")
        zero_range_img = self.apply_lidar_counterfactual_probe(range_img, "zero")
        zero_range_mask = self.apply_lidar_counterfactual_probe(range_mask, "zero")
        zero_lidar_points = self.apply_lidar_counterfactual_probe(lidar_points, "zero")
        zero_lidar_points_mask = self.apply_lidar_counterfactual_probe(lidar_points_mask, "zero")
        zero_lidar_point_features = self.apply_lidar_counterfactual_probe(lidar_point_features, "zero")
        zero_lidar_point_features_mask = self.apply_lidar_counterfactual_probe(lidar_point_features_mask, "zero")
        zero_lidar_ray_features = self.apply_lidar_counterfactual_probe(lidar_ray_features, "zero")
        zero_lidar_ray_features_mask = self.apply_lidar_counterfactual_probe(lidar_ray_features_mask, "zero")
        zero_lidar_pixel_features = self.apply_lidar_counterfactual_probe(lidar_pixel_features, "zero")
        zero_lidar_pixel_features_mask = self.apply_lidar_counterfactual_probe(lidar_pixel_features_mask, "zero")
        zero_lidar_pixel_features_available = self.apply_lidar_counterfactual_probe(
            lidar_pixel_features_available, "zero"
        )
        stop_negative = (
            self.lidar_counterfactual_stop_negative
            and not needs_zero_reconstruction
        )
        neg_ctx = torch.no_grad() if stop_negative else torch.enable_grad()
        with neg_ctx:
            zero_lidar_context = self.make_lidar_context(
                zero_lidar_cond,
                range_img=zero_range_img,
                range_mask=zero_range_mask,
                camera_to_lidar=camera_to_lidar,
                left_camera_k=left_camera_k,
                lidar_points=zero_lidar_points,
                lidar_points_mask=zero_lidar_points_mask,
                lidar_point_features=zero_lidar_point_features,
                lidar_point_features_mask=zero_lidar_point_features_mask,
                lidar_ray_features=zero_lidar_ray_features,
                lidar_ray_features_mask=zero_lidar_ray_features_mask,
                lidar_pixel_features=zero_lidar_pixel_features,
                lidar_pixel_features_mask=zero_lidar_pixel_features_mask,
                lidar_pixel_features_available=zero_lidar_pixel_features_available,
            )
            zero_lidar_evidence = self.make_lidar_evidence(zero_lidar_cond)
            zero_outputs = self.DDPM.p_losses(
                pre_residual_laten,
                t,
                cond_init_grd=None,
                cond_sat=None,
                cond_txt=cond_label,
                lidar_context=zero_lidar_context,
                lidar_evidence=zero_lidar_evidence,
                noise=noise,
                left_camera_k=left_camera_k,
                gt_shift_x=gt_shift_x,
                gt_shift_y=gt_shift_y,
                theta=theta,
                range_img=zero_range_img,
                range_mask=zero_range_mask,
                camera_to_lidar=camera_to_lidar,
                x0_image_target=outputs,
                image_decoder=self.pre_AE_model,
                latent_scale_factor=self.scale_factor,
                return_outputs=True,
            )

        zero_pred = zero_outputs["pred_image"]
        zero_l1 = (
            self.masked_image_l1(zero_pred, outputs, mask)
            if needs_rgb_counterfactual and zero_pred is not None
            else pre_residual_laten.new_tensor(0.0)
        )
        normal_depth_l1 = self.masked_depth_log_l1(
            normal_outputs.get("lidar_depth_pred"),
            lidar_depth_target,
            lidar_depth_mask,
        )
        zero_depth_l1 = self.masked_depth_log_l1(
            zero_outputs.get("lidar_depth_pred"),
            lidar_depth_target,
            lidar_depth_mask,
        )
        zero_l1_for_rank = zero_l1.detach() if self.lidar_counterfactual_stop_negative else zero_l1
        if needs_rgb_counterfactual and self.lidar_counterfactual_exist_weight > 0.0:
            exist_loss = F.relu(normal_l1 - zero_l1_for_rank + float(self.lidar_counterfactual_margin))
        else:
            exist_loss = pre_residual_laten.new_tensor(0.0)

        if needs_rgb_counterfactual and self.lidar_counterfactual_separation_weight > 0.0:
            sep_target = zero_pred.detach() if self.lidar_counterfactual_stop_negative else zero_pred
            sep_l1 = self.masked_image_l1(normal_pred, sep_target, mask)
            separation_loss = F.relu(float(self.lidar_counterfactual_margin) - sep_l1)
        else:
            separation_loss = pre_residual_laten.new_tensor(0.0)
        zero_reconstruction_mask_coverage = 0.0
        if needs_zero_reconstruction:
            if self.lidar_zero_reconstruction_mask_mode == "background":
                zero_loss_raw = zero_outputs.get("loss_raw")
                if zero_loss_raw is None:
                    zero_reconstruction = zero_outputs.get("loss", pre_residual_laten.new_tensor(0.0))
                    zero_reconstruction_mask_coverage = 1.0
                else:
                    foreground_mask = self.DDPM._prepare_loss_mask(mask, zero_loss_raw)
                    reconstruction_mask = (1.0 - foreground_mask).clamp(0.0, 1.0)
                    zero_reconstruction = self.DDPM._masked_loss_mean(zero_loss_raw, reconstruction_mask)
                    zero_reconstruction_mask_coverage = float(
                        reconstruction_mask.detach().float().mean().cpu()
                    )
            else:
                zero_reconstruction = zero_outputs.get("loss", pre_residual_laten.new_tensor(0.0))
                zero_reconstruction_mask_coverage = 1.0
        else:
            zero_reconstruction = pre_residual_laten.new_tensor(0.0)
        total = float(self.lidar_counterfactual_weight) * (
            float(self.lidar_counterfactual_exist_weight) * exist_loss
            + float(self.lidar_counterfactual_separation_weight) * separation_loss
        )
        total = total + float(self.lidar_zero_reconstruction_loss_weight) * zero_reconstruction
        metrics.update(
            {
                "lidar_counterfactual_applied": 1,
                "lidar_counterfactual_mask_coverage": float(mask.detach().float().mean().cpu()),
                "lidar_counterfactual_normal_l1": float(normal_l1.detach().cpu()),
                "lidar_counterfactual_zero_l1": float(zero_l1.detach().cpu()),
                "lidar_counterfactual_normal_depth_log_l1": float(normal_depth_l1.detach().cpu())
                if normal_depth_l1 is not None
                else 0.0,
                "lidar_counterfactual_zero_depth_log_l1": float(zero_depth_l1.detach().cpu())
                if zero_depth_l1 is not None
                else 0.0,
                "lidar_counterfactual_zero_minus_normal_depth_log_l1": float(
                    (zero_depth_l1 - normal_depth_l1).detach().cpu()
                )
                if normal_depth_l1 is not None and zero_depth_l1 is not None
                else 0.0,
                "lidar_counterfactual_exist_loss": float(exist_loss.detach().cpu()),
                "lidar_counterfactual_rank_loss": float(exist_loss.detach().cpu()),
                "lidar_counterfactual_separation_loss": float(separation_loss.detach().cpu()),
                "lidar_counterfactual_total_contrib": float(total.detach().cpu()),
                "lidar_zero_reconstruction_loss": float(zero_reconstruction.detach().cpu()),
                "lidar_zero_reconstruction_loss_contrib": float(
                    (
                        float(self.lidar_zero_reconstruction_loss_weight) * zero_reconstruction
                    ).detach().cpu()
                ),
                "lidar_zero_reconstruction_mask_coverage": zero_reconstruction_mask_coverage,
            }
        )
        self.last_lidar_counterfactual_metrics = metrics
        if "loss_metrics" in normal_outputs:
            self.DDPM.last_loss_metrics = normal_outputs["loss_metrics"]
        self.restore_lidar_monitor_state(normal_monitor_state)
        return total

    def load_pre_sat2grd_model(self, pre_sat2grd_model):
        # self.load_pth_rematch(pre_sat2grd_model, self.Sat2Den, 'Sat2Den.', None)
        self.load_pth_rematch(pre_sat2grd_model, self.pre_AE_model, 'pre_AE_model.', None)
        self.load_pth_rematch(pre_sat2grd_model, self.DDPM.denoise_model, 'DDPM.denoise_model.', None)
        self.load_pth_rematch(pre_sat2grd_model, self.condition_model_grd, 'condition_model_grd.', None)
        self.load_pth_rematch(pre_sat2grd_model, self.condition_model_sat, 'condition_model_sat.', None)

    # def extract_attention_blocks(self):
    #     """
    #     Extracts and returns a dictionary of all AttentionBlock and CrossAttention modules in the model.
        
    #     Returns:
    #         `dict` of modules: A dictionary containing all AttentionBlock and CrossAttention modules in the model,
    #         indexed by their names.
    #     """
    #     attention_blocks = {}

    #     def fn_recursive_extract(name, module, attention_blocks):
    #         if isinstance(module, (AttentionBlock, CrossAttention)):
    #             attention_blocks[name] = module.parameters()

    #         for sub_name, child in module.named_children():
    #             fn_recursive_extract(f"{name}.{sub_name}", child, attention_blocks)

    #         return attention_blocks

    #     for name, module in self.named_children():
    #         fn_recursive_extract(name, module, attention_blocks)

    #     return attention_blocks


    def load_pre_ldm_model(self, pre_sat2grd_model):
        AE_state_dict = OrderedDict()
        for k, v in pre_sat2grd_model.items():
            if 'first_stage_model' in k:
                new_k = k.replace('first_stage_model.', '')
                AE_state_dict[new_k] = v
            elif 'pre_AE_model' in k:
                new_k = k.replace('pre_AE_model.', '')
                AE_state_dict[new_k] = v
        if AE_state_dict:
            self.pre_AE_model.load_state_dict(AE_state_dict, strict=False)

        # cond_state_dict = OrderedDict()
        # for k, v in pre_sat2grd_model.items():
        #     if 'cond_stage_model' in k:
        #         new_k = k.replace('cond_stage_model.', '')
        #         cond_state_dict[new_k] = v
        # self.condition_model_txt.load_state_dict(cond_state_dict)

        DDPM_state_dict = OrderedDict()
        for k, v in pre_sat2grd_model.items():
            # if '.' not in k:
            #     DDPM_state_dict[k] = v
            if 'diffusion_model' in k:
                new_k = k.replace('model.diffusion_model.', '')
                DDPM_state_dict[new_k] = v
            elif 'DDPM.denoise_model' in k:
                new_k = k.replace('DDPM.denoise_model.', '')
                DDPM_state_dict[new_k] = v
        if DDPM_state_dict:
            self.DDPM.denoise_model.load_state_dict(DDPM_state_dict, strict=False)
        
        
    def load_pth_rematch(self, state_dict, model, orin_key, aim_key):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if orin_key in k:
                new_k = k.replace(orin_key, '')  
                new_state_dict[new_k] = v
        if aim_key:
            eval("model." + aim_key).load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)

    def init_AE(self, AE_config, AE_ckpt_path):
        model = instantiate_from_config(AE_config)
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False

        checkpoint = torch.load(AE_ckpt_path, map_location="cpu")['state_dict']
        model_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if 'first_stage_model' in key:
                new_k = key.replace('first_stage_model.', '') 
                model_state_dict[new_k] = checkpoint[key]
            elif 'pre_AE_model' in key:
                new_k = key.replace('pre_AE_model.', '')
                model_state_dict[new_k] = checkpoint[key]
        # self.load_pth_rematch(checkpoint['state_dict'], model, 'pre_AE_model.', None)
        model.load_state_dict(model_state_dict, strict=False)
        return model
        

    def init_Sat2Den(self, Sat2Den_config, AE_ckpt_path):
        model = instantiate_from_config(Sat2Den_config)
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        checkpoint = torch.load(AE_ckpt_path, map_location="cpu")
        self.load_pth_rematch(checkpoint['netG'], model, 'depth_model.', 'depth_model')
        self.load_pth_rematch(checkpoint['netG'], model, 'denoise_model.', 'denoise_model')
        self.load_pth_rematch(checkpoint['netG'], model, 'style_encode.', 'style_encode')
        self.load_pth_rematch(checkpoint['netG'], model, 'style_model.', 'style_model')
        return model

    def logsnr_schedule_cosine(self, t, *, logsnr_min=-20., logsnr_max=20.):
        b = np.arctan(np.exp(-.5 * logsnr_max))
        a = np.arctan(np.exp(-.5 * logsnr_min)) - b
        
        return -2. * torch.log(torch.tan(a * t + b))

    def q_sample(self, gt, logsnr, noise):
        
        # lambdas = logsnr_schedule_cosine(t)
        
        alpha = logsnr.sigmoid().sqrt().to(gt.device)
        sigma = (-logsnr).sigmoid().sqrt().to(gt.device)
        
        alpha = alpha[:,None, None, None]
        sigma = sigma[:,None, None, None]

        return alpha * gt + sigma * noise
    
    @torch.no_grad()
    def p_mean_variance(self, oriimg, noise, logsnr, logsnr_next, w=2.0):
        
        b = oriimg.shape[0]
        w = w[:, None, None, None]
        
        c = - torch.special.expm1(logsnr - logsnr_next)
        
        squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()
        
        alpha, sigma, alpha_next = map(lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
    
        pred_noise = self.denoise_model(oriimg, noise, logsnr.repeat(b).to(oriimg.device))
        
        pred_noise_final = pred_noise
        
        noise = noise
        
        z_start = (noise - sigma * pred_noise_final) / alpha
        z_start.clamp_(-1., 1.)
        
        model_mean = alpha_next * (noise * (1 - c) / alpha + c * z_start)
        
        posterior_variance = squared_sigma_next * c
        
        return model_mean, posterior_variance
    
    @torch.no_grad()
    def p_sample(self, oriimg, noise, logsnr, logsnr_next, w):
        model_mean, model_variance = self.p_mean_variance( oriimg, noise, logsnr=logsnr, logsnr_next=logsnr_next, w = w)
        
        if logsnr_next==0:
            return model_mean
        
        return model_mean + model_variance.sqrt() * torch.randn_like(oriimg).to(oriimg.device)

 
    @torch.no_grad()
    def sample(self, oriimg, w, timesteps=256):
        res_img = torch.randn_like(oriimg).to(oriimg.device)

        logsnrs = self.logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[:-1])
        logsnr_nexts = self.logsnr_schedule_cosine(torch.linspace(1., 0., timesteps+1)[1:])

        res_imgs = []
        for logsnr, logsnr_next in zip(logsnrs, logsnr_nexts): # [1, ..., 0] = size is 257
            res_img = self.p_sample(oriimg, res_img, logsnr=logsnr, logsnr_next=logsnr_next, w=w)
            res_imgs.append(res_img.detach().cpu())
        return res_imgs


    def forward(self, init_pre):
        pre_residual, posterior = self.pre_AE_model(init_pre)
        return pre_residual, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def gen_cond(self, inputs):
        return self.make_condition(inputs)


    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, "sat_map")
        # style_img = self.get_input(batch, 'sky_histc')
        outputs = self.get_input(batch, "grd_left_imgs")
        lidar_cond = None
        if self.use_lidar_cond and self.lidar_condition_key in batch:
            lidar_cond = self.get_input(batch, self.lidar_condition_key)
        range_img = self.get_input(batch, "range_img") if self.use_lidar_cond and "range_img" in batch else None
        range_mask = self.get_input(batch, "range_mask") if self.use_lidar_cond and "range_mask" in batch else None
        lidar_points = (
            batch["lidar_points"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_points" in batch
            else None
        )
        lidar_points_mask = (
            batch["lidar_points_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_points_mask" in batch
            else None
        )
        lidar_point_features = (
            batch["lidar_point_features"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_point_features" in batch
            else None
        )
        lidar_point_features_mask = (
            batch["lidar_point_features_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_point_features_mask" in batch
            else None
        )
        lidar_ray_features = (
            batch["lidar_ray_features"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_ray_features" in batch
            else None
        )
        lidar_ray_features_mask = (
            batch["lidar_ray_features_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_ray_features_mask" in batch
            else None
        )
        lidar_pixel_features = self.get_optional_lidar_batch_tensor(batch, "lidar_pixel_features", outputs.device)
        lidar_pixel_features_mask = self.get_optional_lidar_batch_tensor(
            batch, "lidar_pixel_features_mask", outputs.device
        )
        lidar_pixel_features_available = self.get_optional_lidar_batch_tensor(
            batch, "lidar_pixel_features_available", outputs.device
        )
        camera_to_lidar = (
            self.get_input(batch, "camera_to_lidar").squeeze(-1)
            if self.use_lidar_cond and "camera_to_lidar" in batch
            else None
        )

        left_camera_k = self.get_input(batch, "left_camera_k").squeeze(-1)
        gt_shift_x = batch['gt_shift_x']
        gt_shift_y = batch['gt_shift_y']
        theta = batch['theta']


        inputs = inputs*2 - 1
        outputs = outputs*2 - 1


        cond_label = self.apply_satellite_condition_dropout(self.make_condition(inputs, batch))
        lidar_context = self.make_lidar_context(
            lidar_cond,
            range_img=range_img,
            range_mask=range_mask,
            camera_to_lidar=camera_to_lidar,
            left_camera_k=left_camera_k,
            lidar_points=lidar_points,
            lidar_points_mask=lidar_points_mask,
            lidar_point_features=lidar_point_features,
            lidar_point_features_mask=lidar_point_features_mask,
            lidar_ray_features=lidar_ray_features,
            lidar_ray_features_mask=lidar_ray_features_mask,
            lidar_pixel_features=lidar_pixel_features,
            lidar_pixel_features_mask=lidar_pixel_features_mask,
            lidar_pixel_features_available=lidar_pixel_features_available,
        )
        semantic_pred_tokens = (
            getattr(self.lidar_context_model, "last_semantic_pred_tokens", None)
            if self.lidar_context_model is not None
            else None
        )
        lidar_evidence = self.make_lidar_evidence(lidar_cond)
        pre_residual_laten = self.pre_AE_model.encode(outputs).sample().detach()

        pre_residual_laten = pre_residual_laten * self.scale_factor
        loss_mask = None
        needs_dynamic_mask = (
            self.dynamic_loss_weight > 0.0
            or self.dynamic_x0_loss_weight > 0.0
            or self.dynamic_image_loss_weight > 0.0
            or self.dynamic_crop_image_loss_weight > 0.0
        )
        if needs_dynamic_mask and self.dynamic_mask_key in batch:
            loss_mask = self.latent_dynamic_mask(batch[self.dynamic_mask_key].to(pre_residual_laten.device), pre_residual_laten.shape)
        image_loss_mask = batch[self.dynamic_mask_key].to(outputs.device) if self.dynamic_image_loss_weight > 0.0 and self.dynamic_mask_key in batch else None
        crop_image_loss_mask = self.dynamic_crop_mask(batch[self.dynamic_mask_key].to(outputs.device), outputs.shape) if self.dynamic_crop_image_loss_weight > 0.0 and self.dynamic_mask_key in batch else None
        point_loss_mask = self.dynamic_point_mask(lidar_cond, pre_residual_laten.shape) if (self.dynamic_point_loss_weight > 0.0 or self.dynamic_point_x0_loss_weight > 0.0) else None
        point_image_loss_mask = self.dynamic_point_image_mask(lidar_cond, outputs.shape) if self.dynamic_point_image_loss_weight > 0.0 else None
        lidar_depth_target, lidar_depth_mask = (
            self.lidar_depth_target_mask(lidar_cond, pre_residual_laten.shape)
            if self.lidar_depth_loss_weight > 0.0
            and (self.lidar_depth_output_scale > 0.0 or self.lidar_depth_bottleneck_scale > 0.0)
            else (None, None)
        )
        needs_foreground_mask = (
            self.foreground_loss_weight > 0.0
            or self.foreground_x0_loss_weight > 0.0
            or self.foreground_image_loss_weight > 0.0
            or self.foreground_lpips_loss_weight > 0.0
        )
        foreground_image_loss_mask = self.foreground_image_mask(batch, lidar_cond, outputs.shape) if needs_foreground_mask else None
        foreground_loss_mask = (
            self.latent_dynamic_mask(foreground_image_loss_mask, pre_residual_laten.shape)
            if foreground_image_loss_mask is not None
            else None
        )
        semantic_alignment_loss = self.lidar_semantic_alignment_loss(
            semantic_pred_tokens,
            batch,
            lidar_cond,
            foreground_image_loss_mask,
        )
        static_teacher_loss_mask = self.static_teacher_loss_mask(
            lidar_cond,
            pre_residual_laten.shape,
            foreground_loss_mask=foreground_loss_mask,
        )
        object_boxes = batch.get("dynamic_boxes")
        object_box_valid = batch.get("dynamic_box_valid")
        if object_boxes is not None:
            object_boxes = object_boxes.to(outputs.device)
        if object_box_valid is not None:
            object_box_valid = object_box_valid.to(outputs.device)
        if self.dynamic_object_lpips_weight > 0.0 or self.foreground_lpips_loss_weight > 0.0:
            self.evaluate.loss_fn_alex.eval()
        loss_kwargs = {
            "cond_init_grd": None,
            "cond_sat": None,
            "cond_txt": cond_label,
            "lidar_context": lidar_context,
            "lidar_evidence": lidar_evidence,
            "left_camera_k": left_camera_k,
            "gt_shift_x": gt_shift_x,
            "gt_shift_y": gt_shift_y,
            "theta": theta,
            "range_img": range_img,
            "range_mask": range_mask,
            "camera_to_lidar": camera_to_lidar,
            "loss_mask": loss_mask,
            "loss_mask_weight": self.dynamic_loss_weight,
            "x0_loss_weight": self.dynamic_x0_loss_weight,
            "extra_loss_mask": point_loss_mask,
            "extra_loss_mask_weight": self.dynamic_point_loss_weight,
            "extra_x0_loss_weight": self.dynamic_point_x0_loss_weight,
            "foreground_loss_mask": foreground_loss_mask,
            "foreground_loss_weight": self.foreground_loss_weight,
            "foreground_x0_loss_weight": self.foreground_x0_loss_weight,
            "foreground_image_loss_weight": self.foreground_image_loss_weight,
            "foreground_lpips_loss_weight": self.foreground_lpips_loss_weight,
            "foreground_image_loss_mask": foreground_image_loss_mask,
            "foreground_lpips_padding": self.foreground_lpips_padding,
            "foreground_lpips_size": self.foreground_lpips_size,
            "image_x0_loss_weight": self.dynamic_image_loss_weight,
            "crop_image_x0_loss_weight": self.dynamic_crop_image_loss_weight,
            "point_image_x0_loss_weight": self.dynamic_point_image_loss_weight,
            "object_lpips_loss_weight": self.dynamic_object_lpips_weight,
            "x0_image_target": outputs,
            "image_loss_mask": image_loss_mask,
            "crop_image_loss_mask": crop_image_loss_mask,
            "point_image_loss_mask": point_image_loss_mask,
            "object_boxes": object_boxes,
            "object_box_valid": object_box_valid,
            "object_lpips_model": self.evaluate.loss_fn_alex,
            "object_lpips_padding": self.dynamic_object_lpips_padding,
            "object_lpips_size": self.dynamic_object_lpips_size,
            "object_lpips_max_boxes": self.dynamic_object_lpips_max_boxes,
            "image_decoder": self.pre_AE_model,
            "latent_scale_factor": self.scale_factor,
            "static_teacher_loss_mask": static_teacher_loss_mask,
            "static_teacher_consistency_weight": self.static_teacher_consistency_weight,
            "lidar_depth_target": lidar_depth_target,
            "lidar_depth_mask": lidar_depth_mask,
            "lidar_depth_loss_weight": self.lidar_depth_loss_weight,
            "lidar_depth_output_scale": self.lidar_depth_output_scale,
            "lidar_depth_bottleneck_scale": self.lidar_depth_bottleneck_scale,
            "lidar_depth_log_eps": self.lidar_depth_log_eps,
            "lidar_depth_resample_mode": self.lidar_depth_resample_mode,
        }
        self.last_lidar_counterfactual_metrics = {}
        if (
            (
                self.lidar_counterfactual_weight > 0.0
                or self.lidar_zero_reconstruction_loss_weight > 0.0
            )
            and self.use_lidar_cond
        ):
            t = torch.randint(0, self.DDPM.num_timesteps, (pre_residual_laten.shape[0],), device=pre_residual_laten.device).long()
            noise = torch.randn_like(pre_residual_laten)
            normal_outputs = self.DDPM.p_losses(
                pre_residual_laten,
                t,
                noise=noise,
                return_outputs=True,
                **loss_kwargs,
            )
            loss = normal_outputs["loss"]
            loss = loss + self.lidar_counterfactual_loss(
                normal_outputs,
                pre_residual_laten,
                t,
                noise,
                outputs,
                cond_label,
                lidar_cond,
                range_img,
                range_mask,
                camera_to_lidar,
                left_camera_k,
                lidar_points,
                lidar_points_mask,
                lidar_point_features,
                lidar_point_features_mask,
                lidar_ray_features,
                lidar_ray_features_mask,
                lidar_pixel_features,
                lidar_pixel_features_mask,
                lidar_pixel_features_available,
                gt_shift_x,
                gt_shift_y,
                theta,
                foreground_image_loss_mask,
                point_image_loss_mask,
                lidar_depth_target,
                lidar_depth_mask,
            )
        else:
            loss = self.DDPM.t_losses(pre_residual_laten, **loss_kwargs)
        if semantic_alignment_loss is not None:
            loss = loss + semantic_alignment_loss
        self.log("L1_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
        

    def test_step(self, batch, batch_idx):
        inputs = self.get_input(batch, "sat_map")
        # style_img = self.get_input(batch, 'sky_histc')
        outputs = self.get_input(batch, "grd_left_imgs")
        lidar_cond = None
        if self.use_lidar_cond and self.lidar_condition_key in batch:
            lidar_cond = self.get_input(batch, self.lidar_condition_key)
        range_img = self.get_input(batch, "range_img") if self.use_lidar_cond and "range_img" in batch else None
        range_mask = self.get_input(batch, "range_mask") if self.use_lidar_cond and "range_mask" in batch else None
        lidar_points = (
            batch["lidar_points"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_points" in batch
            else None
        )
        lidar_points_mask = (
            batch["lidar_points_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_points_mask" in batch
            else None
        )
        lidar_point_features = (
            batch["lidar_point_features"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_point_features" in batch
            else None
        )
        lidar_point_features_mask = (
            batch["lidar_point_features_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_point_features_mask" in batch
            else None
        )
        lidar_ray_features = (
            batch["lidar_ray_features"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_ray_features" in batch
            else None
        )
        lidar_ray_features_mask = (
            batch["lidar_ray_features_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_ray_features_mask" in batch
            else None
        )
        lidar_pixel_features = self.get_optional_lidar_batch_tensor(batch, "lidar_pixel_features", outputs.device)
        lidar_pixel_features_mask = self.get_optional_lidar_batch_tensor(
            batch, "lidar_pixel_features_mask", outputs.device
        )
        lidar_pixel_features_available = self.get_optional_lidar_batch_tensor(
            batch, "lidar_pixel_features_available", outputs.device
        )
        camera_to_lidar = (
            self.get_input(batch, "camera_to_lidar").squeeze(-1)
            if self.use_lidar_cond and "camera_to_lidar" in batch
            else None
        )

        left_camera_k = self.get_input(batch, "left_camera_k").squeeze(-1)
        gt_shift_x = batch['gt_shift_x']
        gt_shift_y = batch['gt_shift_y']
        theta = batch['theta']
        file_name = batch['file_name']

        inputs = inputs*2 - 1
        outputs = outputs*2 - 1


        cond_label = self.make_condition(inputs, batch)
        sat_con = cond_label.detach()
        lidar_context = self.make_lidar_context(
            lidar_cond,
            range_img=range_img,
            range_mask=range_mask,
            camera_to_lidar=camera_to_lidar,
            left_camera_k=left_camera_k,
            lidar_points=lidar_points,
            lidar_points_mask=lidar_points_mask,
            lidar_point_features=lidar_point_features,
            lidar_point_features_mask=lidar_point_features_mask,
            lidar_ray_features=lidar_ray_features,
            lidar_ray_features_mask=lidar_ray_features_mask,
            lidar_pixel_features=lidar_pixel_features,
            lidar_pixel_features_mask=lidar_pixel_features_mask,
            lidar_pixel_features_available=lidar_pixel_features_available,
        )
        lidar_evidence = self.make_lidar_evidence(lidar_cond)
        lidar_geometry_mask = self.make_lidar_geometry_mask(lidar_evidence)
        sampler = KITTI_DDIMSampler(self.DDPM, self.pre_AE_model, self.scale_factor)
        n_samples = outputs.size()[0] 
        shape = [1, 4, 16, 64] 
        x_T = torch.randn(shape, device=inputs.device).repeat(sat_con.size()[0], 1, 1, 1)
        start_code = x_T 
        shape = [4, 16, 64] 
        ddim_steps = 50
        scale = 7.5
        ddim_eta = 1
        temperature = 1
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        cond_sat=None, cond_grd=None,
                                        conditioning=sat_con,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=None,
                                        eta=ddim_eta,
                                        x_T=start_code,
                                        temperature = temperature,
                                        left_camera_k = left_camera_k, gt_shift_x = gt_shift_x, gt_shift_y = gt_shift_y, theta = theta,
                                        range_img=range_img, range_mask=range_mask, camera_to_lidar=camera_to_lidar,
                                        lidar_context=lidar_context,
                                        lidar_evidence=lidar_evidence,
                                        lidar_geometry_mask=lidar_geometry_mask,
                                        cond_init_grd = None)
        
        samples_ddim = samples_ddim * (1 / self.scale_factor)
        pre_residual = self.pre_AE_model.decode(samples_ddim)

        pre_residual = torch.clamp((pre_residual + 1.0) / 2.0, min=0.0, max=1.0)
        inputs = torch.clamp((inputs + 1.0) / 2.0, min=0.0, max=1.0)
        outputs = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)

        toPIL = transforms.ToPILImage()
        for batch_num in range(pre_residual.size(0)):
            name = os.path.dirname(file_name[batch_num])
            os.makedirs(f"result/2024-09-26T12-32-20_KITTI_geo_ldm_txt_control/test2/gt_img/{name}", exist_ok=True)
            val_save = toPIL(outputs[batch_num])
            val_save.save(f"result/2024-09-26T12-32-20_KITTI_geo_ldm_txt_control/test2/gt_img/{file_name[batch_num]}.png")

            name = os.path.dirname(file_name[batch_num])
            os.makedirs(f"result/2024-09-26T12-32-20_KITTI_geo_ldm_txt_control/test2/test_img/{name}", exist_ok=True)
            val_save = toPIL(pre_residual[batch_num])
            val_save.save(f"result/2024-09-26T12-32-20_KITTI_geo_ldm_txt_control/test2/test_img/{file_name[batch_num]}.png")

        log_eval = self.evaluate((pre_residual).clamp(0, 1), outputs, split="test")
        self.RMSE.append(log_eval["RMSE"])
        self.SSIM.append(log_eval["SSIM"])
        self.PSNR.append(log_eval["PSNR"])
        self.SD.append(log_eval["SD"])
        self.P_alex.append(log_eval["P_alex"])
        self.P_squeeze.append(log_eval["P_squeeze"])
        self.log("RMSE", log_eval["RMSE"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("SSIM", log_eval["SSIM"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("P_alex", log_eval["P_alex"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if "dynamic_mask" in batch:
            dynamic_log = dynamic_masked_metrics(pre_residual, outputs, batch["dynamic_mask"].to(pre_residual.device), self.evaluate.loss_fn_alex)
            self.dynamic_PSNR.append(dynamic_log["dynamic_psnr"].detach().cpu())
            self.dynamic_SSIM.append(dynamic_log["dynamic_ssim"].detach().cpu())
            self.dynamic_LPIPS.append(dynamic_log["dynamic_lpips"].detach().cpu())
            self.log("dynamic/PSNR", dynamic_log["dynamic_psnr"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("dynamic/SSIM", dynamic_log["dynamic_ssim"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("dynamic/LPIPS", dynamic_log["dynamic_lpips"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("dynamic/mask_coverage", dynamic_log["dynamic_mask_coverage"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return self.log_dict
    
    def eval_data(self):
        RMSE = torch.tensor(self.RMSE).mean()
        SSIM = torch.tensor(self.SSIM).mean()
        PSNR = torch.tensor(self.PSNR).mean()
        SD = torch.tensor(self.SD).mean()
        P_alex = torch.tensor(self.P_alex).mean()
        P_squeeze = torch.tensor(self.P_squeeze).mean()
        self.log("test/RMSE", RMSE)
        self.log("test/SSIM", SSIM)
        self.log("test/PSNR", PSNR)
        self.log("test/SD", SD)
        self.log("test/P_alex", P_alex)
        self.log("test/P_squeeze", P_squeeze)
        self.RMSE = []
        self.SSIM = []
        self.PSNR = []
        self.SD = []
        self.P_alex = []
        self.P_squeeze = []
        return RMSE, SSIM, PSNR, SD, P_alex, P_squeeze

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.freeze_for_lidar_control:
            for module in [self.pre_AE_model, self.DDPM.denoise_model]:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            train_params = []
            if getattr(self.DDPM, "control_grd", None) is not None:
                self.DDPM.control_grd.eval()
                for param in self.DDPM.control_grd.parameters():
                    param.requires_grad = False
            if self.dynamic_class_tokens is not None:
                train_params.append(self.dynamic_class_tokens)
            for param in train_params:
                param.requires_grad = True
            lidar_context_params = []
            if self.lidar_context_model is not None:
                self.lidar_context_model.train()
                for param in self.lidar_context_model.parameters():
                    param.requires_grad = True
                    lidar_context_params.append(param)
            sat_params = []
            if self.lidar_train_sat_condition:
                self.condition_model_sat.train()
                for param in self.condition_model_sat.parameters():
                    param.requires_grad = True
                    sat_params.append(param)
            else:
                self.condition_model_sat.eval()
                for param in self.condition_model_sat.parameters():
                    param.requires_grad = False
            lidar_unet_params = []
            old_unet_params = []
            unet_param_ids = set()
            def add_unet_params(module, target_params):
                module.train()
                for param in module.parameters():
                    if id(param) in unet_param_ids:
                        continue
                    param.requires_grad = True
                    target_params.append(param)
                    unet_param_ids.add(id(param))

            def add_unet_param(param, target_params):
                if id(param) in unet_param_ids:
                    return
                param.requires_grad = True
                target_params.append(param)
                unet_param_ids.add(id(param))

            for name, param in self.DDPM.denoise_model.named_parameters():
                if (
                    ".attn_lidar." in name
                    or ".norm_lidar." in name
                    or ".ray_evidence_attn." in name
                    or ".ray_posterior_fusion." in name
                    or name.startswith("lidar_spatial_residuals.")
                    or ".evidence_router." in name
                    or ".lidar_depth_head." in name
                    or ".lidar_bottleneck_depth_head." in name
                    or name.startswith("lidar_depth_head.")
                    or name.startswith("lidar_bottleneck_depth_head.")
                    or name.endswith(".lidar_gate")
                ):
                    add_unet_param(param, lidar_unet_params)

            if self.lidar_unfreeze_denoise_all:
                add_unet_params(self.DDPM.denoise_model, old_unet_params)

            transformer_scope = self.lidar_unfreeze_transformers.lower()
            if transformer_scope not in {"", "none"}:
                scopes = []
                if transformer_scope in {"input", "all"}:
                    scopes.append(self.DDPM.denoise_model.input_blocks)
                if transformer_scope in {"middle", "output_middle", "all"}:
                    scopes.append(self.DDPM.denoise_model.middle_block)
                if transformer_scope in {"output", "output_middle", "all"}:
                    scopes.append(self.DDPM.denoise_model.output_blocks)
                for scope in scopes:
                    for module in scope.modules():
                        if module.__class__.__name__ == "SpatialTransformer":
                            add_unet_params(module, old_unet_params)

            if self.lidar_unfreeze_output_blocks > 0:
                blocks = self.DDPM.denoise_model.output_blocks[-self.lidar_unfreeze_output_blocks:]
                for block in blocks:
                    add_unet_params(block, old_unet_params)
            if self.lidar_unfreeze_out:
                add_unet_params(self.DDPM.denoise_model.out, old_unet_params)
            param_groups = []
            if train_params:
                param_groups.append({"params": train_params, "lr": lr})
            if lidar_context_params:
                param_groups.append({"params": lidar_context_params, "lr": lr * self.lidar_context_lr_scale})
            if sat_params:
                param_groups.append({"params": sat_params, "lr": lr * self.lidar_sat_lr_scale})
            if lidar_unet_params:
                param_groups.append({"params": lidar_unet_params, "lr": lr * self.lidar_unet_new_lr_scale})
            if old_unet_params:
                param_groups.append({"params": old_unet_params, "lr": lr * self.lidar_unet_lr_scale})
            if not param_groups:
                raise ValueError("No trainable parameters configured for LiDAR training")
            opt = torch.optim.AdamW(param_groups)
        else:
            opt= torch.optim.AdamW(list(self.DDPM.denoise_model.parameters()) +
                                   list(self.condition_model_sat.parameters()) +
                                   (list(self.lidar_context_model.parameters()) if self.lidar_context_model is not None else []),
                                      lr=lr)
        return [opt]

    def get_last_layer(self):
        return self.pre_AE_model.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        inputs = self.get_input(batch, "sat_map")
        # style_img = self.get_input(batch, 'sky_histc')
        outputs = self.get_input(batch, "grd_left_imgs")
        lidar_cond = None
        if self.use_lidar_cond and self.lidar_condition_key in batch:
            lidar_cond = self.get_input(batch, self.lidar_condition_key)
        range_img = self.get_input(batch, "range_img") if self.use_lidar_cond and "range_img" in batch else None
        range_mask = self.get_input(batch, "range_mask") if self.use_lidar_cond and "range_mask" in batch else None
        lidar_points = (
            batch["lidar_points"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_points" in batch
            else None
        )
        lidar_points_mask = (
            batch["lidar_points_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_points_mask" in batch
            else None
        )
        lidar_point_features = (
            batch["lidar_point_features"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_point_features" in batch
            else None
        )
        lidar_point_features_mask = (
            batch["lidar_point_features_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_point_features_mask" in batch
            else None
        )
        lidar_ray_features = (
            batch["lidar_ray_features"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_ray_features" in batch
            else None
        )
        lidar_ray_features_mask = (
            batch["lidar_ray_features_mask"].to(outputs.device).float()
            if self.use_lidar_cond and "lidar_ray_features_mask" in batch
            else None
        )
        lidar_pixel_features = self.get_optional_lidar_batch_tensor(batch, "lidar_pixel_features", outputs.device)
        lidar_pixel_features_mask = self.get_optional_lidar_batch_tensor(
            batch, "lidar_pixel_features_mask", outputs.device
        )
        lidar_pixel_features_available = self.get_optional_lidar_batch_tensor(
            batch, "lidar_pixel_features_available", outputs.device
        )
        camera_to_lidar = (
            self.get_input(batch, "camera_to_lidar").squeeze(-1)
            if self.use_lidar_cond and "camera_to_lidar" in batch
            else None
        )

        left_camera_k = self.get_input(batch, "left_camera_k").squeeze(-1)
        gt_shift_x = batch['gt_shift_x']
        gt_shift_y = batch['gt_shift_y']
        theta = batch['theta']


        inputs = inputs*2 - 1
        outputs = outputs*2 - 1

        cond_label = self.make_condition(inputs, batch)
        sat_con = cond_label.detach()
        lidar_context = self.make_lidar_context(
            lidar_cond,
            range_img=range_img,
            range_mask=range_mask,
            camera_to_lidar=camera_to_lidar,
            left_camera_k=left_camera_k,
            lidar_points=lidar_points,
            lidar_points_mask=lidar_points_mask,
            lidar_point_features=lidar_point_features,
            lidar_point_features_mask=lidar_point_features_mask,
            lidar_ray_features=lidar_ray_features,
            lidar_ray_features_mask=lidar_ray_features_mask,
            lidar_pixel_features=lidar_pixel_features,
            lidar_pixel_features_mask=lidar_pixel_features_mask,
            lidar_pixel_features_available=lidar_pixel_features_available,
        )
        lidar_evidence = self.make_lidar_evidence(lidar_cond)
        lidar_geometry_mask = self.make_lidar_geometry_mask(lidar_evidence)
        sampler = KITTI_DDIMSampler(self.DDPM, self.pre_AE_model, self.scale_factor)

        n_samples = outputs.size()[0]
        shape = [1, 4, 16, 64] 
        x_T = torch.randn(shape, device=inputs.device).repeat(sat_con.size()[0], 1, 1, 1)
        start_code = x_T 
        shape = [4, 16, 64] 
        ddim_steps = 50
        scale = 7.5
        ddim_eta = 1
        temperature = 1
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        cond_sat=None, cond_grd=None,
                                        conditioning=sat_con,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=None,
                                        eta=ddim_eta,
                                        x_T=start_code,
                                        temperature = temperature,
                                        left_camera_k = left_camera_k, gt_shift_x = gt_shift_x, gt_shift_y = gt_shift_y, theta = theta,
                                        range_img=range_img, range_mask=range_mask, camera_to_lidar=camera_to_lidar,
                                        lidar_context=lidar_context,
                                        lidar_evidence=lidar_evidence,
                                        lidar_geometry_mask=lidar_geometry_mask,
                                        cond_init_grd = None)
        
        samples_ddim = samples_ddim * (1 / self.scale_factor)
        # img_list = self.DDPM.sample(noise, cond_init_grd=cond_init_grd, cond_sat=cond_sat)
        pre_residual = self.pre_AE_model.decode(samples_ddim)
        pre_residual = torch.clamp((pre_residual + 1.0) / 2.0, min=0.0, max=1.0)
        # init_result['pred'] = torch.clamp((init_result['pred'] + 1.0) / 2.0, min=0.0, max=1.0)
        inputs = torch.clamp((inputs + 1.0) / 2.0, min=0.0, max=1.0)
        outputs = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)

        # optical = grd2sat_uv(0, 128, 512, 2, 256, 256, meter_per_pixel).to(inputs.device)
        # out_val, mask = grid_sample(inputs, optical)
        # log["reconstructions"] = outputs
        log["inputs"] = inputs
        # log["style_img"] = (pre_residual).clamp(0, 1)
        log["outputs"] = outputs
        log["pre_residual"] = pre_residual.clamp(0, 1)
        if "dynamic_mask" in batch:
            log["dynamic_mask"] = batch["dynamic_mask"].to(pre_residual.device).float()
        if lidar_cond is not None:
            if lidar_cond.shape[1] >= 10:
                log["lidar_cond"] = torch.cat(
                    [lidar_cond[:, 8:9], lidar_cond[:, 1:2], lidar_cond[:, 2:3]],
                    dim=1,
                ).clamp(0, 1)
            elif lidar_cond.shape[1] >= 8:
                log["lidar_cond"] = torch.cat(
                    [lidar_cond[:, 0:1], lidar_cond[:, 4:5], lidar_cond[:, 7:8]],
                    dim=1,
                ).clamp(0, 1)
            else:
                log["lidar_cond"] = lidar_cond[:, :3].clamp(0, 1)
        # log["end_reconstructions"] = (pre_residual).clamp(0, 1)

        # z = samples_ddim
        # B, C, _, _ = z.size()
        # sample = self.pre_AE_model.decode(torch.randn_like(z.reshape(B, C, 16, 64)))
        # log["sample"] = sample
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
