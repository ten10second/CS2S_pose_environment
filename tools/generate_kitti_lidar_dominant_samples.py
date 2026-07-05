import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from dataloader.kitti_raw_lidar_utils import lidar_condition_channels, lidar_condition_uses_pointmap  # noqa: E402
from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed KITTI LiDAR-dominant samples from a checkpoint.")
    parser.add_argument("--config", default="results/kitti_xlidar_overfit/lidar_dominant_sdinit_b4_9000step_partial/run_config.yaml")
    parser.add_argument("--ckpt", default="results/kitti_xlidar_overfit/lidar_dominant_sdinit_b4_9000step_partial/checkpoints/last.pt")
    parser.add_argument("--baseline-cs2s-ckpt", default="result/KITTI.ckpt")
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/foreground_test2_hardcases.jsonl")
    parser.add_argument("--out-dir", default="results/kitti_xlidar_overfit/lidar_dominant_sdinit_b4_9000step_partial/sample_vis_last")
    parser.add_argument(
        "--condition-mode",
        default="",
        choices=["", "none", "bbox_dynamic", "dynamic_points", "raw_lidar", "dynamic_full", "raw_lidar_pointmap"],
        help="LiDAR condition mode. Empty uses the mode saved in the config/checkpoint run config.",
    )
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--probes", default="normal,zero", help="Comma-separated probes. Supported probes: normal,zero.")
    parser.add_argument(
        "--shift-fraction",
        type=float,
        default=None,
        help="Deprecated no-op kept for old commands; global LiDAR shift probes are disabled.",
    )
    parser.add_argument(
        "--key-stats-max-tokens",
        type=int,
        default=256,
        help="Maximum LiDAR tokens used for exact pairwise key cosine diagnostics. Use <=0 to disable key diagnostics.",
    )
    return parser.parse_args()


def safe_sample_id(sample_id):
    return str(sample_id).replace("/", "__")


def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    transforms.functional.to_pil_image(tensor.detach().cpu().clamp(0, 1)).save(path)


def sample_to_batch(sample):
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).cuda(non_blocking=True)
        else:
            batch[key] = [value]
    return batch


def apply_probe_tensor(tensor, probe):
    if tensor is None or probe == "normal":
        return tensor
    if probe == "zero":
        return torch.zeros_like(tensor)
    raise ValueError(
        f"Unsupported probe: {probe}. Global LiDAR shift_x was removed because it moves road/background geometry; "
        "use normal,zero."
    )


def load_checkpoint_into_model(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    cs2s_init_ckpt = metadata.get("cs2s_init_ckpt", "")
    if cs2s_init_ckpt and Path(cs2s_init_ckpt).is_file():
        load_cs2s_backbone(model, cs2s_init_ckpt)
    if "control_grd" in payload and getattr(model.DDPM, "control_grd", None) is not None:
        load_state_dict_compatible(model.DDPM.control_grd, payload["control_grd"])
    if "lidar_context_model" in payload and getattr(model, "lidar_context_model", None) is not None:
        load_state_dict_compatible(model.lidar_context_model, payload["lidar_context_model"])
    if "condition_model_sat" in payload:
        load_state_dict_compatible(model.condition_model_sat, payload["condition_model_sat"])
    if "denoise_model_trainable" in payload:
        load_state_dict_compatible(model.DDPM.denoise_model, payload["denoise_model_trainable"])
    del payload
    gc.collect()


def load_state_dict_compatible(module, state_dict):
    current = module.state_dict()
    compatible = {}
    for name, tensor in state_dict.items():
        if name in current and tuple(current[name].shape) == tuple(tensor.shape):
            compatible[name] = tensor
    module.load_state_dict(compatible, strict=False)


def _prefixed_state_dict(state_dict, prefix):
    return {name[len(prefix):]: value for name, value in state_dict.items() if name.startswith(prefix)}


def load_cs2s_backbone(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    denoise_state = _prefixed_state_dict(state_dict, "DDPM.denoise_model.")
    if denoise_state:
        model.DDPM.denoise_model.load_state_dict(denoise_state, strict=False)
    sat_state = _prefixed_state_dict(state_dict, "condition_model_sat.")
    if sat_state:
        model.condition_model_sat.load_state_dict(sat_state, strict=False)
    ae_state = _prefixed_state_dict(state_dict, "pre_AE_model.")
    if ae_state:
        model.pre_AE_model.load_state_dict(ae_state, strict=False)
    del payload
    gc.collect()


def make_lidar_overlay(gt, lidar_cond, dynamic_mask=None):
    base = transforms.functional.to_pil_image(gt.detach().cpu().clamp(0, 1)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cond = lidar_cond.detach().cpu()
    point = cond[1] > 0.5 if cond.shape[0] > 1 else torch.zeros(cond.shape[-2:], dtype=torch.bool)
    depth = cond[2].clamp(0, 1) if cond.shape[0] > 2 else torch.zeros_like(point, dtype=torch.float32)
    dyn = None
    if dynamic_mask is not None:
        dyn = dynamic_mask.detach().cpu()
        if dyn.ndim == 3:
            dyn = dyn[0]
        dyn = dyn > 0.5
    width, height = base.size
    for y in range(height):
        for x in range(width):
            if dyn is not None and bool(dyn[y, x]):
                draw.point((x, y), fill=(255, 32, 32, 65))
            if bool(point[y, x]):
                d = int(255 * float(depth[y, x]))
                draw.point((x, y), fill=(32, 240, max(64, d), 230))
    return Image.alpha_composite(base, overlay).convert("RGB")


def make_condition_rgb(lidar_cond):
    cond = lidar_cond.detach().cpu().clamp(0, 1)
    if cond.shape[0] >= 10:
        return torch.cat([cond[8:9], cond[1:2], cond[2:3]], dim=0)
    if cond.shape[0] < 3:
        cond = torch.nn.functional.pad(cond, (0, 0, 0, 0, 0, 3 - cond.shape[0]))
    return torch.cat([cond[1:2], cond[2:3], cond[0:1]], dim=0)


def resolve_condition_mode(cfg, args):
    if args.condition_mode:
        return args.condition_mode
    try:
        return str(cfg.data.params.test.params.condition_mode)
    except AttributeError:
        return "raw_lidar"


def configure_lidar_condition_mode(cfg, condition_mode):
    cfg.data.params.test.params.condition_mode = condition_mode
    cfg.data.params.test.params.include_range_image = condition_mode != "none"
    if condition_mode == "none":
        return cfg
    if "Lidar_context_config" not in cfg.model.params or cfg.model.params.Lidar_context_config is None:
        return cfg
    params = cfg.model.params.Lidar_context_config.params
    params.front_in_channels = int(lidar_condition_channels(condition_mode))
    params.use_pointmap_pe = bool(lidar_condition_uses_pointmap(condition_mode))
    return cfg


def lidar_attention_stats(model):
    entropy = []
    max_mean = []
    std_mean = []
    sim_std = []
    sim_range = []
    token_count = []
    query_count = []
    route_x_mean = []
    route_y_mean = []
    route_x_std = []
    route_y_std = []
    route_query_x_corr = []
    route_query_y_corr = []
    denoise_model = getattr(getattr(model, "DDPM", None), "denoise_model", None)
    if denoise_model is None:
        return {
            "lidar_attn_modules": 0,
            "lidar_sim_std_mean": 0.0,
            "lidar_sim_range_mean": 0.0,
            "lidar_attn_entropy_norm_mean": 0.0,
            "lidar_attn_max_mean": 0.0,
            "lidar_attn_std_mean": 0.0,
            "lidar_attn_token_count_mean": 0.0,
            "lidar_attn_query_count_mean": 0.0,
        }
    for module in denoise_model.modules():
        if hasattr(module, "last_attn_entropy_norm"):
            entropy_value = module.last_attn_entropy_norm
            max_value = module.last_attn_max_mean
            std_value = module.last_attn_std_mean
            sim_std_value = getattr(module, "last_sim_std_mean", 0.0)
            sim_range_value = getattr(module, "last_sim_range_mean", 0.0)
            if torch.is_tensor(entropy_value):
                entropy_value = float(entropy_value.detach().float().cpu())
            if torch.is_tensor(max_value):
                max_value = float(max_value.detach().float().cpu())
            if torch.is_tensor(std_value):
                std_value = float(std_value.detach().float().cpu())
            if torch.is_tensor(sim_std_value):
                sim_std_value = float(sim_std_value.detach().float().cpu())
            if torch.is_tensor(sim_range_value):
                sim_range_value = float(sim_range_value.detach().float().cpu())
            entropy.append(float(entropy_value))
            max_mean.append(float(max_value))
            std_mean.append(float(std_value))
            sim_std.append(float(sim_std_value))
            sim_range.append(float(sim_range_value))
            token_count.append(float(module.last_attn_token_count))
            query_count.append(float(module.last_attn_query_count))
            for values, attr in [
                (route_x_mean, "last_route_token_x_mean"),
                (route_y_mean, "last_route_token_y_mean"),
                (route_x_std, "last_route_token_x_std"),
                (route_y_std, "last_route_token_y_std"),
                (route_query_x_corr, "last_route_query_x_corr"),
                (route_query_y_corr, "last_route_query_y_corr"),
            ]:
                value = getattr(module, attr, None)
                if value is None:
                    continue
                if torch.is_tensor(value):
                    value = float(value.detach().float().cpu())
                values.append(float(value))
    if not entropy:
        return {
            "lidar_attn_modules": 0,
            "lidar_sim_std_mean": 0.0,
            "lidar_sim_range_mean": 0.0,
            "lidar_attn_entropy_norm_mean": 0.0,
            "lidar_attn_max_mean": 0.0,
            "lidar_attn_std_mean": 0.0,
            "lidar_attn_token_count_mean": 0.0,
            "lidar_attn_query_count_mean": 0.0,
        }
    count = float(len(entropy))
    return {
        "lidar_attn_modules": int(len(entropy)),
        "lidar_sim_std_mean": sum(sim_std) / count,
        "lidar_sim_range_mean": sum(sim_range) / count,
        "lidar_attn_entropy_norm_mean": sum(entropy) / count,
        "lidar_attn_max_mean": sum(max_mean) / count,
        "lidar_attn_std_mean": sum(std_mean) / count,
        "lidar_attn_token_count_mean": sum(token_count) / count,
        "lidar_attn_query_count_mean": sum(query_count) / count,
        "lidar_route_token_x_mean": sum(route_x_mean) / float(len(route_x_mean)) if route_x_mean else 0.0,
        "lidar_route_token_y_mean": sum(route_y_mean) / float(len(route_y_mean)) if route_y_mean else 0.0,
        "lidar_route_token_x_std": sum(route_x_std) / float(len(route_x_std)) if route_x_std else 0.0,
        "lidar_route_token_y_std": sum(route_y_std) / float(len(route_y_std)) if route_y_std else 0.0,
        "lidar_route_query_x_corr": sum(route_query_x_corr) / float(len(route_query_x_corr)) if route_query_x_corr else 0.0,
        "lidar_route_query_y_corr": sum(route_query_y_corr) / float(len(route_query_y_corr)) if route_query_y_corr else 0.0,
    }


def empty_token_structure_stats(prefix):
    return {
        f"{prefix}_token_count": 0,
        f"{prefix}_sampled_token_count": 0,
        f"{prefix}_dim": 0,
        f"{prefix}_var_mean": 0.0,
        f"{prefix}_var_max": 0.0,
        f"{prefix}_norm_mean": 0.0,
        f"{prefix}_norm_std": 0.0,
        f"{prefix}_cos_offdiag_mean": 0.0,
        f"{prefix}_cos_offdiag_std": 0.0,
        f"{prefix}_cos_offdiag_min": 0.0,
        f"{prefix}_cos_offdiag_p05": 0.0,
        f"{prefix}_cos_offdiag_p50": 0.0,
        f"{prefix}_cos_offdiag_p95": 0.0,
        f"{prefix}_cos_offdiag_max": 0.0,
    }


def token_structure_stats(tokens, prefix, max_tokens=256):
    if tokens is None or int(max_tokens) <= 0:
        return empty_token_structure_stats(prefix)
    x = tokens.detach().float()
    if x.ndim != 3:
        return empty_token_structure_stats(prefix)
    token_count = int(x.shape[1])
    if token_count <= 1:
        stats = empty_token_structure_stats(prefix)
        stats[f"{prefix}_token_count"] = token_count
        stats[f"{prefix}_sampled_token_count"] = token_count
        stats[f"{prefix}_dim"] = int(x.shape[-1])
        return stats
    sampled_count = min(token_count, int(max_tokens))
    if sampled_count < token_count:
        indices = torch.linspace(0, token_count - 1, sampled_count, device=x.device).long()
        x = x.index_select(1, indices)
    norm = x.norm(dim=-1)
    var_per_dim = x.var(dim=1, unbiased=False)
    normalized = F.normalize(x, dim=-1, eps=1e-6)
    cosine = torch.matmul(normalized, normalized.transpose(1, 2))
    mask = ~torch.eye(sampled_count, dtype=torch.bool, device=x.device)
    offdiag = cosine[:, mask].reshape(-1)
    quantiles = torch.quantile(offdiag, torch.tensor([0.05, 0.5, 0.95], device=x.device))
    return {
        f"{prefix}_token_count": token_count,
        f"{prefix}_sampled_token_count": sampled_count,
        f"{prefix}_dim": int(x.shape[-1]),
        f"{prefix}_var_mean": float(var_per_dim.mean().detach().cpu()),
        f"{prefix}_var_max": float(var_per_dim.max().detach().cpu()),
        f"{prefix}_norm_mean": float(norm.mean().detach().cpu()),
        f"{prefix}_norm_std": float(norm.std(unbiased=False).detach().cpu()),
        f"{prefix}_cos_offdiag_mean": float(offdiag.mean().detach().cpu()),
        f"{prefix}_cos_offdiag_std": float(offdiag.std(unbiased=False).detach().cpu()),
        f"{prefix}_cos_offdiag_min": float(offdiag.min().detach().cpu()),
        f"{prefix}_cos_offdiag_p05": float(quantiles[0].detach().cpu()),
        f"{prefix}_cos_offdiag_p50": float(quantiles[1].detach().cpu()),
        f"{prefix}_cos_offdiag_p95": float(quantiles[2].detach().cpu()),
        f"{prefix}_cos_offdiag_max": float(offdiag.max().detach().cpu()),
    }


def summarize_structure_stats(module_stats, source_key, output_prefix):
    if not module_stats:
        return {f"{output_prefix}_module_count": 0}
    suffixes = [
        "var_mean",
        "var_max",
        "norm_mean",
        "norm_std",
        "cos_offdiag_mean",
        "cos_offdiag_std",
        "cos_offdiag_p95",
    ]
    summary = {f"{output_prefix}_module_count": len(module_stats)}
    for suffix in suffixes:
        key = f"{source_key}_{suffix}"
        values = [float(item[source_key][key]) for item in module_stats if key in item.get(source_key, {})]
        if values:
            summary[f"{output_prefix}_{suffix}_mean"] = sum(values) / float(len(values))
            summary[f"{output_prefix}_{suffix}_min"] = min(values)
            summary[f"{output_prefix}_{suffix}_max"] = max(values)
    return summary


@torch.no_grad()
def lidar_key_structure_stats(model, lidar_context, max_tokens=256):
    if lidar_context is None or int(max_tokens) <= 0:
        return {
            "enabled": False,
            "raw_lidar_tokens": empty_token_structure_stats("raw_lidar_tokens"),
            "prepared_lidar_tokens": empty_token_structure_stats("prepared_lidar_tokens"),
            "projected_key_summary": {"projected_key_module_count": 0},
            "projected_key_modules": [],
        }
    raw_stats = token_structure_stats(lidar_context, "raw_lidar_tokens", max_tokens=max_tokens)
    prepared_summary = None
    module_stats = []
    denoise_model = getattr(getattr(model, "DDPM", None), "denoise_model", None)
    if denoise_model is not None:
        for module_index, module in enumerate(denoise_model.modules()):
            attn_lidar = getattr(module, "attn_lidar", None)
            if attn_lidar is None or not hasattr(attn_lidar, "to_k"):
                continue
            prepared_context = (
                attn_lidar.prepare_context(lidar_context)
                if hasattr(attn_lidar, "prepare_context")
                else lidar_context
            )
            if prepared_summary is None:
                prepared_summary = token_structure_stats(
                    prepared_context,
                    "prepared_lidar_tokens",
                    max_tokens=max_tokens,
                )
            key = attn_lidar.to_k(prepared_context)
            key = rearrange(key, "b n (h d) -> (b h) n d", h=attn_lidar.heads)
            module_stats.append(
                {
                    "module_index": module_index,
                    "projected_key": token_structure_stats(key, "projected_key", max_tokens=max_tokens),
                }
            )
    return {
        "enabled": True,
        "raw_lidar_tokens": raw_stats,
        "prepared_lidar_tokens": prepared_summary
        or empty_token_structure_stats("prepared_lidar_tokens"),
        "projected_key_summary": summarize_structure_stats(module_stats, "projected_key", "projected_key"),
        "projected_key_modules": module_stats,
    }


def make_panel(out_dir, sample_id, image_paths):
    labels = list(image_paths.keys())
    images = [Image.open(path).convert("RGB") for path in image_paths.values()]
    width, height = images[0].size
    label_h = 22
    panel = Image.new("RGB", (width * len(images), height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    for idx, (label, image) in enumerate(zip(labels, images)):
        panel.paste(image, (idx * width, label_h))
        draw.text((idx * width + 4, 4), label, fill=(0, 0, 0))
    panel_path = out_dir / "panels" / f"{safe_sample_id(sample_id)}.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(panel_path)
    for image in images:
        image.close()
    return panel_path


@torch.no_grad()
def generate_prediction(
    model,
    batch,
    probe,
    ddim_steps,
    seed,
    guidance_scale,
    eta,
    temperature,
    use_lidar=True,
    key_stats_max_tokens=256,
):
    inputs = model.get_input(batch, "sat_map").cuda()
    outputs = model.get_input(batch, "grd_left_imgs").cuda()
    lidar_cond = model.get_input(batch, model.lidar_condition_key).cuda()
    range_img = model.get_input(batch, "range_img").cuda() if "range_img" in batch else None
    range_mask = model.get_input(batch, "range_mask").cuda() if "range_mask" in batch else None
    camera_to_lidar = model.get_input(batch, "camera_to_lidar").squeeze(-1).cuda()
    left_camera_k = model.get_input(batch, "left_camera_k").squeeze(-1).cuda()
    gt_shift_x = batch["gt_shift_x"].cuda()
    gt_shift_y = batch["gt_shift_y"].cuda()
    theta = batch["theta"].cuda()

    lidar_cond = apply_probe_tensor(lidar_cond, probe)
    range_img = apply_probe_tensor(range_img, probe)
    range_mask = apply_probe_tensor(range_mask, probe)

    inputs = inputs * 2 - 1
    outputs = outputs * 2 - 1
    cond_label = model.make_condition(inputs, batch).detach()
    lidar_context = None
    lidar_evidence = None
    if use_lidar:
        lidar_context = model.make_lidar_context(
            lidar_cond,
            range_img=range_img,
            range_mask=range_mask,
            camera_to_lidar=camera_to_lidar,
            left_camera_k=left_camera_k,
        )
        lidar_evidence = model.make_lidar_evidence(lidar_cond) if hasattr(model, "make_lidar_evidence") else None
    key_structure_stats = lidar_key_structure_stats(model, lidar_context, max_tokens=key_stats_max_tokens)
    torch.manual_seed(seed)
    x_t = torch.randn((cond_label.shape[0], 4, 16, 64), device=inputs.device)
    sampler = KITTI_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor)
    samples_ddim, _ = sampler.sample(
        S=ddim_steps,
        cond_sat=None,
        cond_grd=None,
        conditioning=cond_label,
        batch_size=cond_label.shape[0],
        shape=[4, 16, 64],
        verbose=False,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=None,
        eta=eta,
        x_T=x_t,
        temperature=temperature,
        left_camera_k=left_camera_k,
        gt_shift_x=gt_shift_x,
        gt_shift_y=gt_shift_y,
        theta=theta,
        range_img=range_img,
        range_mask=range_mask,
        camera_to_lidar=camera_to_lidar,
        lidar_context=lidar_context,
        lidar_evidence=lidar_evidence,
        cond_init_grd=None,
    )
    pred = model.pre_AE_model.decode(samples_ddim * (1 / model.scale_factor))
    pred = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)
    target = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)
    return pred, target, lidar_attention_stats(model), key_structure_stats


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    condition_mode = resolve_condition_mode(cfg, args)
    configure_lidar_condition_mode(cfg, condition_mode)
    cfg.data.params.test.params.manifest = args.manifest

    dataset = SatLidarRawDataset(
        manifest=args.manifest,
        condition_mode=condition_mode,
        image_height=128,
        image_width=512,
        sat_size=256,
        max_depth=80.0,
        align_satellite_to_camera=True,
        include_range_image=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = [item.strip() for item in args.probes.split(",") if item.strip()]
    unsupported = [probe for probe in probes if probe not in {"normal", "zero"}]
    if unsupported:
        raise ValueError(
            "Unsupported probes: "
            + ", ".join(unsupported)
            + ". Global LiDAR shift_x was removed because it moves road/background geometry; use normal,zero."
        )
    samples = [dataset[idx] for idx in range(min(args.num_samples, len(dataset)))]
    records = []
    image_paths_by_id = {}
    attention_stats_by_id = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        safe_id = safe_sample_id(sample_id)
        gt_path = out_dir / "images" / "gt" / f"{safe_id}.png"
        overlay_path = out_dir / "images" / "lidar_overlay" / f"{safe_id}.png"
        cond_path = out_dir / "images" / "lidar_cond" / f"{safe_id}.png"
        target = sample["grd_left_imgs"].unsqueeze(0).clamp(0.0, 1.0)
        save_tensor_image(target[0], gt_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        make_lidar_overlay(target[0], sample["lidar_cond"], sample.get("dynamic_mask")).save(overlay_path)
        save_tensor_image(make_condition_rgb(sample["lidar_cond"]), cond_path)

        image_paths = {"GT": gt_path, "LiDAR overlay": overlay_path}
        image_paths_by_id[sample_id] = image_paths
        attention_stats_by_id[sample_id] = {}

    if args.baseline_cs2s_ckpt and Path(args.baseline_cs2s_ckpt).is_file():
        model = instantiate_from_config(cfg.model).cuda().eval()
        load_cs2s_backbone(model, args.baseline_cs2s_ckpt)
        for idx, sample in enumerate(samples):
            sample_id = sample["sample_id"]
            safe_id = safe_sample_id(sample_id)
            batch = sample_to_batch(sample)
            pred, _, attention_stats, key_structure_stats = generate_prediction(
                model,
                batch,
                probe="normal",
                ddim_steps=args.ddim_steps,
                seed=args.seed + idx,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
                temperature=args.temperature,
                use_lidar=False,
                key_stats_max_tokens=args.key_stats_max_tokens,
            )
            pred_path = out_dir / "images" / "KITTI_ckpt" / f"{safe_id}.png"
            save_tensor_image(pred[0], pred_path)
            image_paths_by_id[sample_id]["KITTI.ckpt"] = pred_path
            attention_stats_by_id[sample_id]["KITTI.ckpt"] = {
                "attention": attention_stats,
                "key_structure": key_structure_stats,
            }
            del pred, batch
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.ckpt)
    for idx, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        safe_id = safe_sample_id(sample_id)
        batch = sample_to_batch(sample)
        image_paths = image_paths_by_id[sample_id]
        for probe in probes:
            pred, _, attention_stats, key_structure_stats = generate_prediction(
                model,
                batch,
                probe=probe,
                ddim_steps=args.ddim_steps,
                seed=args.seed + idx,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
                temperature=args.temperature,
                use_lidar=True,
                key_stats_max_tokens=args.key_stats_max_tokens,
            )
            pred_path = out_dir / "images" / probe / f"{safe_id}.png"
            save_tensor_image(pred[0], pred_path)
            image_paths[f"trained:{probe}"] = pred_path
            attention_stats_by_id[sample_id][f"trained:{probe}"] = {
                "attention": attention_stats,
                "key_structure": key_structure_stats,
            }
            del pred
            torch.cuda.empty_cache()
        panel_path = make_panel(out_dir, sample_id, image_paths)
        records.append(
            {
                "sample_id": sample_id,
                "panel_path": str(panel_path),
                "lidar_attention_stats": attention_stats_by_id[sample_id],
                **{k: str(v) for k, v in image_paths.items()},
            }
        )
        del batch
    (out_dir / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    print(json.dumps({"out_dir": str(out_dir), "num_samples": len(records), "records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
