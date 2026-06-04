import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
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
                static_teacher_consistency_weight=0.0,
                static_teacher_gate_channel=-1,
                dynamic_class_token_weight=0.0,
                dynamic_class_hist_key="dynamic_class_hist",
                dynamic_class_token_count=8,
                dynamic_class_token_dim=768,
                lidar_unfreeze_output_blocks=0,
                lidar_unfreeze_out=False,
                lidar_unfreeze_transformers="none",
                lidar_unet_lr_scale=0.25,
                # lossconfig
                 ):
        super().__init__()

        self.pre_AE_model = self.init_AE(AE_config, AE_ckpt_path)
        self.DDPM = instantiate_from_config(DDPM_config)
        if pre_ldm_model_path is not None:
            pre_ldm_model = torch.load(pre_ldm_model_path)
            self.load_pre_ldm_model(pre_ldm_model['state_dict'])

        self.condition_model_sat = instantiate_from_config(Condition_config_sat)
        self.scale_factor = scale_factor
        self.use_lidar_cond = use_lidar_cond
        self.lidar_condition_key = lidar_condition_key
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
        self.static_teacher_consistency_weight = float(static_teacher_consistency_weight)
        self.static_teacher_gate_channel = int(static_teacher_gate_channel)
        self.dynamic_class_token_weight = dynamic_class_token_weight
        self.dynamic_class_hist_key = dynamic_class_hist_key
        if dynamic_class_token_weight > 0.0:
            self.dynamic_class_tokens = nn.Parameter(torch.zeros(dynamic_class_token_count, dynamic_class_token_dim))
        else:
            self.dynamic_class_tokens = None
        self.lidar_unfreeze_output_blocks = int(lidar_unfreeze_output_blocks)
        self.lidar_unfreeze_out = bool(lidar_unfreeze_out)
        self.lidar_unfreeze_transformers = str(lidar_unfreeze_transformers or "none")
        self.lidar_unet_lr_scale = float(lidar_unet_lr_scale)
        if pre_sat2grd_model_path is not None:
            pre_sat2grd_model = torch.load(pre_sat2grd_model_path)
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

    def static_teacher_loss_mask(self, lidar_cond, latent_shape):
        if (
            lidar_cond is None
            or self.static_teacher_consistency_weight <= 0.0
            or self.static_teacher_gate_channel < 0
            or self.static_teacher_gate_channel >= lidar_cond.shape[1]
        ):
            return None
        gate = lidar_cond[:, self.static_teacher_gate_channel : self.static_teacher_gate_channel + 1].float()
        gate = gate.clamp(0.0, 1.0)
        if gate.shape[-2:] != latent_shape[-2:]:
            gate = F.interpolate(gate, size=latent_shape[-2:], mode="area")
        return (1.0 - gate).clamp(0.0, 1.0)

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
        self.pre_AE_model.load_state_dict(AE_state_dict)

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
        self.DDPM.denoise_model.load_state_dict(DDPM_state_dict, strict=False)  
        
        
    def load_pth_rematch(self, state_dict, model, orin_key, aim_key):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if orin_key in k:
                new_k = k.replace(orin_key, '')  
                new_state_dict[new_k] = v
        if aim_key:
            eval("model." + aim_key).load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)

    def init_AE(self, AE_config, AE_ckpt_path):
        model = instantiate_from_config(AE_config)
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False

        checkpoint = torch.load(AE_ckpt_path)['state_dict']
        model_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if 'first_stage_model' in key:
                new_k = key.replace('first_stage_model.', '') 
                model_state_dict[new_k] = checkpoint[key]
        # self.load_pth_rematch(checkpoint['state_dict'], model, 'pre_AE_model.', None)
        model.load_state_dict(model_state_dict)
        return model
        

    def init_Sat2Den(self, Sat2Den_config, AE_ckpt_path):
        model = instantiate_from_config(Sat2Den_config)
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        checkpoint = torch.load(AE_ckpt_path)
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

        left_camera_k = self.get_input(batch, "left_camera_k").squeeze(-1)
        gt_shift_x = batch['gt_shift_x']
        gt_shift_y = batch['gt_shift_y']
        theta = batch['theta']


        inputs = inputs*2 - 1
        outputs = outputs*2 - 1


        cond_label = self.make_condition(inputs, batch)
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
        static_teacher_loss_mask = self.static_teacher_loss_mask(lidar_cond, pre_residual_laten.shape)
        object_boxes = batch.get("dynamic_boxes")
        object_box_valid = batch.get("dynamic_box_valid")
        if object_boxes is not None:
            object_boxes = object_boxes.to(outputs.device)
        if object_box_valid is not None:
            object_box_valid = object_box_valid.to(outputs.device)
        if self.dynamic_object_lpips_weight > 0.0:
            self.evaluate.loss_fn_alex.eval()
        loss = self.DDPM.t_losses(pre_residual_laten, cond_init_grd=lidar_cond, cond_sat=None, cond_txt = cond_label, left_camera_k = left_camera_k, gt_shift_x = gt_shift_x, gt_shift_y = gt_shift_y, theta = theta, loss_mask=loss_mask, loss_mask_weight=self.dynamic_loss_weight, x0_loss_weight=self.dynamic_x0_loss_weight, extra_loss_mask=point_loss_mask, extra_loss_mask_weight=self.dynamic_point_loss_weight, extra_x0_loss_weight=self.dynamic_point_x0_loss_weight, image_x0_loss_weight=self.dynamic_image_loss_weight, crop_image_x0_loss_weight=self.dynamic_crop_image_loss_weight, point_image_x0_loss_weight=self.dynamic_point_image_loss_weight, object_lpips_loss_weight=self.dynamic_object_lpips_weight, x0_image_target=outputs, image_loss_mask=image_loss_mask, crop_image_loss_mask=crop_image_loss_mask, point_image_loss_mask=point_image_loss_mask, object_boxes=object_boxes, object_box_valid=object_box_valid, object_lpips_model=self.evaluate.loss_fn_alex, object_lpips_padding=self.dynamic_object_lpips_padding, object_lpips_size=self.dynamic_object_lpips_size, object_lpips_max_boxes=self.dynamic_object_lpips_max_boxes, image_decoder=self.pre_AE_model, latent_scale_factor=self.scale_factor, static_teacher_loss_mask=static_teacher_loss_mask, static_teacher_consistency_weight=self.static_teacher_consistency_weight)
        self.log("L1_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
        

    def test_step(self, batch, batch_idx):
        inputs = self.get_input(batch, "sat_map")
        # style_img = self.get_input(batch, 'sky_histc')
        outputs = self.get_input(batch, "grd_left_imgs")
        lidar_cond = None
        if self.use_lidar_cond and self.lidar_condition_key in batch:
            lidar_cond = self.get_input(batch, self.lidar_condition_key)

        left_camera_k = self.get_input(batch, "left_camera_k").squeeze(-1)
        gt_shift_x = batch['gt_shift_x']
        gt_shift_y = batch['gt_shift_y']
        theta = batch['theta']
        file_name = batch['file_name']

        inputs = inputs*2 - 1
        outputs = outputs*2 - 1


        cond_label = self.make_condition(inputs, batch)
        sat_con = cond_label.detach()

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
                                        cond_init_grd = lidar_cond)
        
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
            for module in [self.pre_AE_model, self.condition_model_sat, self.DDPM.denoise_model]:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            train_params = list(self.DDPM.control_grd.parameters())
            if self.dynamic_class_tokens is not None:
                train_params.append(self.dynamic_class_tokens)
            for param in train_params:
                param.requires_grad = True
            unet_params = []
            unet_param_ids = set()
            def add_unet_params(module):
                module.train()
                for param in module.parameters():
                    if id(param) in unet_param_ids:
                        continue
                    param.requires_grad = True
                    unet_params.append(param)
                    unet_param_ids.add(id(param))

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
                            add_unet_params(module)

            if self.lidar_unfreeze_output_blocks > 0:
                blocks = self.DDPM.denoise_model.output_blocks[-self.lidar_unfreeze_output_blocks:]
                for block in blocks:
                    add_unet_params(block)
            if self.lidar_unfreeze_out:
                add_unet_params(self.DDPM.denoise_model.out)
            if unet_params:
                opt = torch.optim.AdamW(
                    [
                        {"params": train_params, "lr": lr},
                        {"params": unet_params, "lr": lr * self.lidar_unet_lr_scale},
                    ]
                )
            else:
                opt = torch.optim.AdamW(train_params, lr=lr)
        else:
            opt= torch.optim.AdamW(list(self.DDPM.denoise_model.parameters()) +
                                   list(self.condition_model_sat.parameters()),
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

        left_camera_k = self.get_input(batch, "left_camera_k").squeeze(-1)
        gt_shift_x = batch['gt_shift_x']
        gt_shift_y = batch['gt_shift_y']
        theta = batch['theta']


        inputs = inputs*2 - 1
        outputs = outputs*2 - 1

        cond_label = self.make_condition(inputs, batch)
        sat_con = cond_label.detach()

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
                                        cond_init_grd = lidar_cond)
        
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
                    [lidar_cond[:, 0:1], lidar_cond[:, 8:9], lidar_cond[:, 9:10]],
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
