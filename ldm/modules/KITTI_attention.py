from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint

from models.geometry.kitti_sat2grd_new import gen_KITTI_sat2grd

def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.,
        use_lidar_ray_posterior=False,
        lidar_posterior_log_depth_sigma=0.35,
        lidar_posterior_strength=2.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.use_lidar_ray_posterior = bool(use_lidar_ray_posterior)
        self.lidar_posterior_log_depth_sigma = float(lidar_posterior_log_depth_sigma)
        self.lidar_posterior_strength = float(lidar_posterior_strength)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        #采样
        self.num_points = 8
        self.sampling_offsets = nn.Linear(query_dim, self.heads * self.num_points * 2)
        self.attention_weights = nn.Linear(query_dim, self.heads * self.num_points)
        self.value_proj = nn.Linear(context_dim, context_dim)
        self.sample_to_out = nn.Linear(context_dim, query_dim)
        
        self.gen_KITTI_sat2grd = gen_KITTI_sat2grd()
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        self.register_buffer("last_ray_posterior_hit_coverage", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_ray_posterior_prior_entropy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_ray_posterior_entropy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_ray_posterior_weight_shift", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_ray_posterior_depth_error", torch.tensor(0.0), persistent=False)

    @staticmethod
    def _nearest_depth_for_query_grid(ray_depth_evidence, query_hw):
        if ray_depth_evidence is None or ray_depth_evidence.ndim != 4 or ray_depth_evidence.shape[1] < 2:
            return None, None
        query_h, query_w = int(query_hw[0]), int(query_hw[1])
        evidence = ray_depth_evidence.float()
        depth = evidence[:, 0:1].clamp(0.0, 1.0)
        hit = evidence[:, 1:2].clamp(0.0, 1.0)
        pooled_hit = F.adaptive_max_pool2d(hit, (query_h, query_w))
        sentinel = torch.full_like(depth, 2.0)
        masked_depth = torch.where(hit > 0.0, depth, sentinel)
        pooled_depth = -F.adaptive_max_pool2d(-masked_depth, (query_h, query_w))
        pooled_depth = torch.where(pooled_hit > 0.0, pooled_depth, torch.zeros_like(pooled_depth))
        return (
            pooled_depth.flatten(2).squeeze(1),
            pooled_hit.flatten(2).squeeze(1),
        )

    def _apply_lidar_ray_posterior(
        self,
        prior_logits,
        candidate_depth,
        candidate_valid,
        ray_depth_evidence,
        query_hw,
    ):
        prior_weights = prior_logits.softmax(dim=-1)
        if (
            not self.use_lidar_ray_posterior
            or candidate_depth is None
            or candidate_valid is None
            or ray_depth_evidence is None
        ):
            self.last_ray_posterior_hit_coverage.zero_()
            self.last_ray_posterior_weight_shift.zero_()
            return prior_weights

        observed_depth_norm, hit = self._nearest_depth_for_query_grid(ray_depth_evidence, query_hw)
        if observed_depth_norm is None or observed_depth_norm.shape[1] != candidate_depth.shape[1]:
            self.last_ray_posterior_hit_coverage.zero_()
            self.last_ray_posterior_weight_shift.zero_()
            return prior_weights

        observed_depth = observed_depth_norm.to(candidate_depth.dtype) * 80.0
        hit = hit.to(candidate_depth.dtype).clamp(0.0, 1.0)
        sigma = max(float(self.lidar_posterior_log_depth_sigma), 1e-3)
        candidate_depth_safe = candidate_depth.to(prior_logits.dtype).clamp_min(1e-3)
        observed_depth_safe = observed_depth.to(prior_logits.dtype).clamp_min(1e-3)
        log_error = torch.log(candidate_depth_safe) - torch.log(observed_depth_safe.unsqueeze(-1))
        log_likelihood = (-0.5 * (log_error / sigma).square()).clamp(min=-12.0, max=0.0)
        log_likelihood = torch.where(
            candidate_valid.to(device=log_likelihood.device),
            log_likelihood,
            torch.full_like(log_likelihood, -12.0),
        )
        posterior_logits = prior_logits + (
            float(self.lidar_posterior_strength)
            * hit[:, None, :, None]
            * log_likelihood[:, None, :, :]
        )
        posterior_weights = posterior_logits.softmax(dim=-1)

        with torch.no_grad():
            eps = 1e-8
            prior_entropy = -(prior_weights.float().clamp_min(eps).log() * prior_weights.float()).sum(dim=-1)
            posterior_entropy = -(
                posterior_weights.float().clamp_min(eps).log() * posterior_weights.float()
            ).sum(dim=-1)
            hit_heads = hit[:, None, :].expand(-1, prior_logits.shape[1], -1)
            denom = hit_heads.sum().clamp_min(1.0)
            selected_depth = (
                posterior_weights.float() * candidate_depth_safe[:, None].float()
            ).sum(dim=-1)
            depth_error = (
                (torch.log(selected_depth.clamp_min(1e-3)) - torch.log(observed_depth_safe[:, None])).abs()
                * hit_heads
            ).sum() / denom
            self.last_ray_posterior_hit_coverage.copy_(hit.float().mean().to(self.last_ray_posterior_hit_coverage.device))
            self.last_ray_posterior_prior_entropy.copy_(
                ((prior_entropy * hit_heads).sum() / denom).to(self.last_ray_posterior_prior_entropy.device)
            )
            self.last_ray_posterior_entropy.copy_(
                ((posterior_entropy * hit_heads).sum() / denom).to(self.last_ray_posterior_entropy.device)
            )
            self.last_ray_posterior_weight_shift.copy_(
                (
                    ((posterior_weights - prior_weights).abs().mean(dim=-1) * hit_heads).sum()
                    / denom
                ).to(self.last_ray_posterior_weight_shift.device)
            )
            self.last_ray_posterior_depth_error.copy_(depth_error.to(self.last_ray_posterior_depth_error.device))
        return posterior_weights

    def forward(
        self,
        x,
        context=None,
        mask=None,
        left_camera_k=None,
        gt_shift_x=None,
        gt_shift_y=None,
        theta=None,
        ray_depth_evidence=None,
    ):
        if context == None:
            h = self.heads
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        else:
            bs, num_query, dim = x.shape
            feat_size = int(context.size(1) ** (1/2))

            query_hw = (int(num_query**(1/2)/2), int(num_query**(1/2)*2))
            hypotheses = self.gen_KITTI_sat2grd.sat2grd_h(
                int(feat_size),
                query_hw[0],
                query_hw[1],
                left_camera_k,
                gt_shift_x,
                gt_shift_y,
                theta,
                return_ray_hypotheses=self.use_lidar_ray_posterior,
            )
            if self.use_lidar_ray_posterior:
                reference_points_rebatch, candidate_depth, candidate_valid = hypotheses
            else:
                reference_points_rebatch = hypotheses
                candidate_depth = None
                candidate_valid = None
            # reference_points_rebatch, indexes = CVUSA_grd2sat_uv_h(int(num_query**(1/2)/2), int(num_query**(1/2)*2), int(feat_size), int(feat_size), meter_per_pixel)
            reference_points_rebatch = reference_points_rebatch.to(x.device)
            #需要return out为b, 1024, 320 #1024 = 16*64
            heads = self.heads
            cond_sat = context
            # cond_sat = rearrange(cond_sat, 'b (h w) c -> b c h w', h = int(cond_sat.size(1) ** (1/2)))
            sampling_grids = reference_points_rebatch
            # max_len = max([len(each) for each in indexes])

            # new_query = torch.zeros((bs, max_len, dim), device=x.device)
            # for i in range(len(indexes)):
            #     new_query[:, :len(indexes[i]), :] = x[:, indexes[i], :]
            grid_len = sampling_grids.shape[1]
            sampling_offsets = self.sampling_offsets(x).view(bs, grid_len, heads, self.num_points, 2)
            offset_normalizer = feat_size #可以处理多个level 暂时还没有用
            sampling_offsets = sampling_offsets / offset_normalizer
            #bs, max_len, heads, self.num_points, 2
            sampling_grids = sampling_grids[:, :, None, :, :] + sampling_offsets #扩散采样点 #bs, max_len, heads, heights, 2
            sampling_grids = 2 * sampling_grids - 1
            sampling_grids = sampling_grids.transpose(1, 3).flatten(0, 1) #bs*heads, num_points, 2

            _, x_len, _ = x.shape
            attention_logits = self.attention_weights(x).view(bs, x_len, heads, self.num_points)
            attention_logits = attention_logits.transpose(1, 2)
            attention_weights = self._apply_lidar_ray_posterior(
                attention_logits,
                candidate_depth,
                candidate_valid,
                ray_depth_evidence,
                query_hw=query_hw,
            )
            attention_weights = attention_weights.reshape(bs, heads, x_len, self.num_points)#bs*heads, num_points
            
            bs, num_value, _ = cond_sat.shape
            value = self.value_proj(cond_sat)
            value = value.view(bs, num_value, heads, -1)
            value = rearrange(value, 'b (u v) h d -> (b h) d u v', u=int(num_value ** (1/2)))#bs*heads, dim, H, W
            v_dim = value.shape[1]
            sampling_value_l_ = F.grid_sample(
                value, # [2, 3, 256, 256]
                sampling_grids, #[2, 8, 31652, 2]
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            # bs_sam_h, dim, _, _= sampling_value_l_.shape
            gen_img = sampling_value_l_
            # for i in range(len(indexes)):
            #     gen_img[i, :, :, indexes[i]] = sampling_value_l_[i, :, :, :len(indexes[i])]
            gen_img = rearrange(gen_img, '(b samh) dim head len -> b head dim len samh', samh=8)
            gen_img = (gen_img * attention_weights[:, :, None, :, :]).sum(-1).reshape(bs, heads*v_dim, x_len)
            return self.sample_to_out(gen_img.transpose(1, 2))


class TokenCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.center_context_tokens = True
        self.norm_context_tokens = True
        self.coord_pos_encoding = True
        self.coord_pos_scale = 0.25
        self.coord_logit_bias_scale = 0.75
        self.context_norm = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            zero_module(nn.Linear(inner_dim, query_dim)),
            nn.Dropout(dropout),
        )

    def prepare_context(self, context):
        if getattr(self, "center_context_tokens", True):
            context = context - context.mean(dim=1, keepdim=True)
        if getattr(self, "norm_context_tokens", True):
            context = self.context_norm(context)
        return context

    @staticmethod
    def infer_kitti_grid(token_count):
        token_count = int(token_count)
        if token_count <= 0:
            return 1, 1
        token_h = max(1, int(round(math.sqrt(float(token_count) / 4.0))))
        while token_h > 1 and token_count % token_h != 0:
            token_h -= 1
        token_w = max(1, token_count // token_h)
        return token_h, token_w

    @staticmethod
    def coord_encoding_2d(height, width, dim, device, dtype):
        height = int(height)
        width = int(width)
        dim = int(dim)
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, width)
        x = x.expand(height, width)
        y = y.expand(height, width)
        features = [x, y, x * y, x.square(), y.square()]
        for freq in (1.0, 2.0, 4.0, 8.0):
            features.extend(
                [
                    torch.sin(math.pi * freq * x),
                    torch.cos(math.pi * freq * x),
                    torch.sin(math.pi * freq * y),
                    torch.cos(math.pi * freq * y),
                ]
            )
        base = torch.stack(features, dim=-1).reshape(1, height * width, -1)
        base = base - base.mean(dim=1, keepdim=True)
        base = base / base.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        repeat = int(math.ceil(float(dim) / float(base.shape[-1])))
        return base.repeat(1, 1, repeat)[..., :dim]

    def add_coord_pos_encoding(self, tensor, spatial_hw=None):
        if not getattr(self, "coord_pos_encoding", True) or tensor is None:
            return tensor
        if spatial_hw is None:
            h, w = self.infer_kitti_grid(tensor.shape[1])
        else:
            h, w = int(spatial_hw[0]), int(spatial_hw[1])
            if h * w != int(tensor.shape[1]):
                return tensor
        pos = self.coord_encoding_2d(
            h,
            w,
            tensor.shape[-1],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return tensor + float(getattr(self, "coord_pos_scale", 0.25)) * pos

    def coordinate_logit_bias(self, query_count, token_count, query_hw, device, dtype):
        if query_hw is None or float(getattr(self, "coord_logit_bias_scale", 0.0)) == 0.0:
            return None
        query_h, query_w = int(query_hw[0]), int(query_hw[1])
        if query_h * query_w != int(query_count):
            return None
        token_h, token_w = self.infer_kitti_grid(token_count)
        if token_h * token_w != int(token_count):
            return None
        qx = torch.linspace(-1.0, 1.0, query_w, device=device, dtype=dtype).view(1, query_w)
        qy = torch.linspace(-1.0, 1.0, query_h, device=device, dtype=dtype).view(query_h, 1)
        qx = qx.expand(query_h, query_w).reshape(query_count, 1)
        qy = qy.expand(query_h, query_w).reshape(query_count, 1)
        tx = torch.linspace(-1.0, 1.0, token_w, device=device, dtype=dtype).view(1, token_w)
        ty = torch.linspace(-1.0, 1.0, token_h, device=device, dtype=dtype).view(token_h, 1)
        tx = tx.expand(token_h, token_w).reshape(1, token_count)
        ty = ty.expand(token_h, token_w).reshape(1, token_count)
        dist2 = (qx - tx).square() + (qy - ty).square()
        return -float(getattr(self, "coord_logit_bias_scale", 0.75)) * dist2

    def forward(self, x, context, mask=None, query_hw=None):
        h = self.heads
        context = self.prepare_context(context)
        context = self.add_coord_pos_encoding(context)
        x_for_q = self.add_coord_pos_encoding(x, spatial_hw=query_hw)
        q = self.to_q(x_for_q)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value(sim))
        coord_bias = self.coordinate_logit_bias(
            query_count=sim.shape[-2],
            token_count=sim.shape[-1],
            query_hw=query_hw,
            device=sim.device,
            dtype=sim.dtype,
        )
        if coord_bias is not None:
            sim = sim + coord_bias.unsqueeze(0)

        attn = sim.softmax(dim=-1)
        route_tensors = self._compute_route_tensors(attn, query_hw=query_hw)
        with torch.no_grad():
            sim_float = sim.detach().float()
            attn_float = attn.detach().float()
            token_count = max(1, int(attn_float.shape[-1]))
            entropy = -(attn_float.clamp_min(1e-12) * attn_float.clamp_min(1e-12).log()).sum(dim=-1)
            entropy_norm = entropy / math.log(max(2, token_count))
            self.last_sim_std_mean = sim_float.std(dim=-1, unbiased=False).mean().detach()
            self.last_sim_range_mean = (sim_float.max(dim=-1).values - sim_float.min(dim=-1).values).mean().detach()
            self.last_attn_entropy_norm = entropy_norm.mean().detach()
            self.last_attn_max_mean = attn_float.max(dim=-1).values.mean().detach()
            self.last_attn_std_mean = attn_float.std(dim=-1, unbiased=False).mean().detach()
            self.last_attn_token_count = float(token_count)
            self.last_attn_query_count = float(attn_float.shape[-2])
            self._record_route_stats(route_tensors, query_hw=query_hw)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

    def _compute_route_tensors(self, attn, query_hw=None):
        token_count = int(attn.shape[-1])
        query_count = int(attn.shape[-2])
        batch_heads = int(attn.shape[0])
        batch = max(1, batch_heads // max(1, int(self.heads)))
        if batch * int(self.heads) != batch_heads or token_count <= 1:
            return None

        # LiDAR tokens are produced from a camera-view grid. Infer the 4:1 KITTI
        # token aspect ratio used by the current config, falling back gracefully.
        token_h, token_w = self.infer_kitti_grid(token_count)
        if token_h * token_w != token_count:
            return None

        attn_mean = attn.reshape(batch, int(self.heads), query_count, token_count).mean(dim=1)
        token_x = torch.linspace(-1.0, 1.0, token_w, device=attn.device, dtype=attn.dtype).view(1, token_w)
        token_y = torch.linspace(-1.0, 1.0, token_h, device=attn.device, dtype=attn.dtype).view(token_h, 1)
        token_x = token_x.expand(token_h, token_w).reshape(1, 1, token_count)
        token_y = token_y.expand(token_h, token_w).reshape(1, 1, token_count)
        route_x = (attn_mean * token_x).sum(dim=-1)
        route_y = (attn_mean * token_y).sum(dim=-1)
        query_x = None
        query_y = None
        if query_hw is not None:
            query_h, query_w = int(query_hw[0]), int(query_hw[1])
            if query_h * query_w == query_count:
                query_x = torch.linspace(-1.0, 1.0, query_w, device=attn.device, dtype=attn.dtype).view(1, query_w)
                query_y = torch.linspace(-1.0, 1.0, query_h, device=attn.device, dtype=attn.dtype).view(query_h, 1)
                query_x = query_x.expand(query_h, query_w).reshape(1, query_count).expand(batch, query_count)
                query_y = query_y.expand(query_h, query_w).reshape(1, query_count).expand(batch, query_count)
        return route_x, route_y, query_x, query_y

    def _record_route_stats(self, route_tensors, query_hw=None):
        if route_tensors is None:
            return
        route_x, route_y, query_x, query_y = route_tensors
        route_x = route_x.detach().float()
        route_y = route_y.detach().float()
        self.last_route_token_x_mean = route_x.mean().detach()
        self.last_route_token_y_mean = route_y.mean().detach()
        self.last_route_token_x_std = route_x.std(unbiased=False).detach()
        self.last_route_token_y_std = route_y.std(unbiased=False).detach()
        self.last_route_query_x_corr = torch.zeros((), device=route_x.device)
        self.last_route_query_y_corr = torch.zeros((), device=route_x.device)

        if query_x is None or query_y is None:
            return
        self.last_route_query_x_corr = self._corrcoef(query_x.detach().float(), route_x).detach()
        self.last_route_query_y_corr = self._corrcoef(query_y.detach().float(), route_y).detach()

    @staticmethod
    def _corrcoef(a, b):
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        denom = a.std(dim=-1, unbiased=False) * b.std(dim=-1, unbiased=False)
        corr = (a * b).mean(dim=-1) / denom.clamp_min(1e-6)
        return corr.mean()


class LocalReferenceCrossAttention(TokenCrossAttention):
    """Pointmap-style local reference attention over camera-view LiDAR tokens.

    The token encoder already rasterizes LiDAR as a target camera-view pointmap
    and pools it to a KITTI-aspect grid.  This attention keeps that grid as a
    spatial reference: each street latent query samples K/V only from the
    corresponding pointmap location and a small local neighborhood, instead of
    attending globally over all LiDAR tokens.
    """

    def __init__(self, *args, reference_window: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        window = max(1, int(reference_window))
        if window % 2 == 0:
            window += 1
        self.reference_window = window

    @staticmethod
    def _query_coordinates(query_hw, device, dtype):
        query_h, query_w = int(query_hw[0]), int(query_hw[1])
        qx = torch.linspace(-1.0, 1.0, query_w, device=device, dtype=dtype).view(1, query_w)
        qy = torch.linspace(-1.0, 1.0, query_h, device=device, dtype=dtype).view(query_h, 1)
        qx = qx.expand(query_h, query_w).reshape(query_h * query_w)
        qy = qy.expand(query_h, query_w).reshape(query_h * query_w)
        return qx, qy

    def _reference_grid(self, query_hw, token_hw, batch_heads, device, dtype):
        query_h, query_w = int(query_hw[0]), int(query_hw[1])
        token_h, token_w = int(token_hw[0]), int(token_hw[1])
        qx, qy = self._query_coordinates((query_h, query_w), device=device, dtype=dtype)
        radius = int(self.reference_window // 2)
        step_x = 2.0 / float(max(token_w - 1, 1))
        step_y = 2.0 / float(max(token_h - 1, 1))
        offsets = []
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                offsets.append((float(ox) * step_x, float(oy) * step_y))
        offset = torch.tensor(offsets, device=device, dtype=dtype).view(1, len(offsets), 2)
        base = torch.stack([qx, qy], dim=-1).unsqueeze(1)
        grid = (base + offset).clamp(-1.0, 1.0)
        return grid.unsqueeze(0).expand(int(batch_heads), -1, -1, -1)

    def _record_local_stats(self, sim, attn, sample_grid, query_hw):
        with torch.no_grad():
            sim_float = sim.detach().float()
            attn_float = attn.detach().float()
            token_count = max(1, int(attn_float.shape[-1]))
            entropy = -(attn_float.clamp_min(1e-12) * attn_float.clamp_min(1e-12).log()).sum(dim=-1)
            entropy_norm = entropy / math.log(max(2, token_count))
            self.last_sim_std_mean = sim_float.std(dim=-1, unbiased=False).mean().detach()
            self.last_sim_range_mean = (sim_float.max(dim=-1).values - sim_float.min(dim=-1).values).mean().detach()
            self.last_attn_entropy_norm = entropy_norm.mean().detach()
            self.last_attn_max_mean = attn_float.max(dim=-1).values.mean().detach()
            self.last_attn_std_mean = attn_float.std(dim=-1, unbiased=False).mean().detach()
            self.last_attn_token_count = float(token_count)
            self.last_attn_query_count = float(attn_float.shape[-2])

            batch_heads, query_count, local_count = attn.shape
            batch = max(1, int(batch_heads) // max(1, int(self.heads)))
            if batch * int(self.heads) != batch_heads:
                return
            attn_mean = attn.reshape(batch, int(self.heads), query_count, local_count).mean(dim=1)
            coords = sample_grid.reshape(batch, int(self.heads), query_count, local_count, 2).mean(dim=1)
            route_x = (attn_mean * coords[..., 0]).sum(dim=-1)
            route_y = (attn_mean * coords[..., 1]).sum(dim=-1)
            query_x, query_y = self._query_coordinates(query_hw, device=attn.device, dtype=attn.dtype)
            query_x = query_x.reshape(1, query_count).expand(batch, query_count)
            query_y = query_y.reshape(1, query_count).expand(batch, query_count)
            self._record_route_stats((route_x, route_y, query_x, query_y), query_hw=query_hw)

    def forward(self, x, context, mask=None, query_hw=None):
        if query_hw is None:
            return super().forward(x, context, mask=mask, query_hw=query_hw)
        query_count = int(x.shape[1])
        query_h, query_w = int(query_hw[0]), int(query_hw[1])
        if query_h * query_w != query_count:
            return super().forward(x, context, mask=mask, query_hw=query_hw)

        token_count = int(context.shape[1])
        token_h, token_w = self.infer_kitti_grid(token_count)
        if token_h * token_w != token_count:
            return super().forward(x, context, mask=mask, query_hw=query_hw)

        heads = self.heads
        context = self.prepare_context(context)
        context = self.add_coord_pos_encoding(context)
        x_for_q = self.add_coord_pos_encoding(x, spatial_hw=query_hw)

        q = self.to_q(x_for_q)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, "b n (h d) -> (b h) n d", h=heads)
        k_grid = rearrange(k, "b (th tw) (h d) -> (b h) d th tw", h=heads, th=token_h, tw=token_w)
        v_grid = rearrange(v, "b (th tw) (h d) -> (b h) d th tw", h=heads, th=token_h, tw=token_w)

        sample_grid = self._reference_grid(
            query_hw=(query_h, query_w),
            token_hw=(token_h, token_w),
            batch_heads=q.shape[0],
            device=q.device,
            dtype=q.dtype,
        )
        sampled_k = F.grid_sample(
            k_grid,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_v = F.grid_sample(
            v_grid,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_k = rearrange(sampled_k, "bh d n k -> bh n k d")
        sampled_v = rearrange(sampled_v, "bh d n k -> bh n k d")

        sim = (q.unsqueeze(2) * sampled_k).sum(dim=-1) * self.scale
        attn = sim.softmax(dim=-1)
        self._record_local_stats(sim, attn, sample_grid, query_hw=(query_h, query_w))
        out = (attn.unsqueeze(-1) * sampled_v).sum(dim=2)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=heads)
        return self.to_out(out)


class RayAlignedEvidenceAttention(nn.Module):
    """Fuse heterogeneous reference deltas around each target camera ray.

    Satellite and LiDAR branches first produce per-query reference evidence.
    This module then chooses among those evidence vectors with a tiny attention
    over the evidence set of each image-space query.  It initializes almost as
    the original CS2S satellite residual, so old satellite capability is not
    destroyed when this module is introduced.
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        sat_bias=4.0,
        lidar_bias=-4.0,
        null_bias=-6.0,
    ):
        super().__init__()
        inner_dim = int(heads) * int(dim_head)
        self.heads = int(heads)
        self.scale = float(dim_head) ** -0.5
        # Keep the initial evidence logits bias-only without deadlocking the
        # bilinear q-k router: a non-zero query lets the zero-initialized key
        # projection receive gradients on the first optimization step.
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = zero_module(nn.Linear(dim, inner_dim, bias=False))
        self.ray_proj = zero_module(nn.Linear(dim, dim, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.null_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.evidence_bias = nn.Parameter(
            torch.tensor([float(sat_bias), float(lidar_bias), float(null_bias)], dtype=torch.float32)
        )
        self.register_buffer("last_evidence_sat_weight", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_lidar_weight", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_lidar_weight_std", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_lidar_weight_min", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_lidar_weight_max", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_null_weight", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_entropy_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_lidar_mask_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_lidar_weight_masked", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_lidar_weight_background", torch.tensor(0.0), persistent=False)

    @staticmethod
    def ray_encoding_2d(height, width, dim, device, dtype):
        return TokenCrossAttention.coord_encoding_2d(height, width, dim, device=device, dtype=dtype)

    def _record_stats(self, attn):
        with torch.no_grad():
            weights = attn.detach().float().mean(dim=(0, 1, 2))
            lidar_weight = attn.detach().float().mean(dim=1)[..., 1]
            entropy = -(attn.detach().float().clamp_min(1e-12) * attn.detach().float().clamp_min(1e-12).log()).sum(dim=-1)
            entropy_norm = entropy / math.log(max(2, int(attn.shape[-1])))
            self.last_evidence_sat_weight.copy_(weights[0].to(self.last_evidence_sat_weight.device))
            self.last_evidence_lidar_weight.copy_(weights[1].to(self.last_evidence_lidar_weight.device))
            self.last_evidence_lidar_weight_std.copy_(
                lidar_weight.std(unbiased=False).to(self.last_evidence_lidar_weight_std.device)
            )
            self.last_evidence_lidar_weight_min.copy_(
                lidar_weight.min().to(self.last_evidence_lidar_weight_min.device)
            )
            self.last_evidence_lidar_weight_max.copy_(
                lidar_weight.max().to(self.last_evidence_lidar_weight_max.device)
            )
            self.last_evidence_null_weight.copy_(weights[2].to(self.last_evidence_null_weight.device))
            self.last_evidence_entropy_norm.copy_(
                entropy_norm.mean().detach().to(self.last_evidence_entropy_norm.device)
            )

    def _prepare_mask(self, mask, x, query_hw=None):
        if mask is None:
            return None
        b, n, _ = x.shape
        if mask.dim() == 4:
            if query_hw is None:
                return None
            h, w = int(query_hw[0]), int(query_hw[1])
            mask = F.interpolate(mask.float(), size=(h, w), mode="area")
            mask = rearrange(mask, "b c h w -> b (h w) c")
        elif mask.dim() == 3:
            if mask.shape[1] != n and mask.shape[-1] == n:
                mask = mask.transpose(1, 2)
            if mask.shape[-1] != 1:
                mask = mask.mean(dim=-1, keepdim=True)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        else:
            return None
        if mask.shape[0] != b or mask.shape[1] != n:
            return None
        return mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def forward(self, x, sat_ref, lidar_ref, query_hw=None, lidar_geometry_mask=None):
        if sat_ref is None:
            sat_ref = torch.zeros_like(x)
        if lidar_ref is None:
            lidar_ref = torch.zeros_like(x)

        b, n, c = x.shape
        null = self.null_token.to(device=x.device, dtype=x.dtype).expand(b, n, c)
        evidence = torch.stack([sat_ref, lidar_ref, null], dim=2)

        q_input = x
        if query_hw is not None:
            h, w = int(query_hw[0]), int(query_hw[1])
            if h * w == n:
                ray = self.ray_encoding_2d(h, w, c, device=x.device, dtype=x.dtype)
                q_input = q_input + self.ray_proj(ray).expand(b, n, c)

        q = self.to_q(q_input)
        k = self.to_k(evidence)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n e (h d) -> b h n e d", h=self.heads)

        logits = (q.unsqueeze(3) * k).sum(dim=-1) * self.scale
        bias = self.evidence_bias.to(device=logits.device, dtype=logits.dtype).view(1, 1, 1, 3)
        attn = (logits + bias).softmax(dim=-1)
        self._record_stats(attn)
        attn = self.dropout(attn)

        # Evidence vectors are already in the UNet feature dimension.  Using them
        # directly keeps the initialization close to the original satellite
        # reference residual instead of adding an extra random output projection.
        attn_mean = attn.mean(dim=1)
        if lidar_geometry_mask is None:
            return (attn_mean.unsqueeze(-1) * evidence).sum(dim=2)

        mask = self._prepare_mask(lidar_geometry_mask, x, query_hw=query_hw)
        if mask is None:
            return (attn_mean.unsqueeze(-1) * evidence).sum(dim=2)

        sat_w = attn_mean[:, :, 0:1]
        lidar_w = attn_mean[:, :, 1:2]
        null_w = attn_mean[:, :, 2:3]
        sat_null_denom = (sat_w + null_w).clamp_min(1e-6)
        sat_null_ref = (sat_w * sat_ref + null_w * null) / sat_null_denom
        lidar_correction = lidar_w * (lidar_ref - sat_null_ref)
        with torch.no_grad():
            mask_float = mask.detach().float()
            bg_float = 1.0 - mask_float
            lidar_weight = lidar_w.detach().float()
            self.last_evidence_lidar_mask_mean.copy_(mask_float.mean().to(self.last_evidence_lidar_mask_mean.device))
            self.last_evidence_lidar_weight_masked.copy_(
                ((lidar_weight * mask_float).sum() / mask_float.sum().clamp_min(1.0)).to(
                    self.last_evidence_lidar_weight_masked.device
                )
            )
            self.last_evidence_lidar_weight_background.copy_(
                ((lidar_weight * bg_float).sum() / bg_float.sum().clamp_min(1.0)).to(
                    self.last_evidence_lidar_weight_background.device
                )
            )
        return sat_null_ref + mask * lidar_correction


class TemporalEvidenceHub:
    """Carries per-step temporal payloads to fusion/attention modules.

    Payload fields (any may be absent):
      transport          — v1 latent homography transport (callable)
      strength           — v1 feature multiplier
      sat_tokens_prev    — route-C: previous frame's satellite patch tokens
                           (b, 196, ctx), class/dynamic tokens excluded
      sat_shift_xy       — route-C: (b, 2) ego-motion shift of this frame's
                           satellite crop center inside the previous crop,
                           normalized crop coords
      sat_token_grid     — route-C: (14, 14)
      history_tokens     — v2/v3: previous-frame latent features
      history_grid       — v3: current query anchors in previous history map
                           grid_sample coords, align_corners=False
      history_valid      — v3: valid geometry correspondences for anchors
      history_hw         — v2/v3: previous history feature map size

    Modules hold a reference to the hub, so no UNet forward signature changes.
    """

    def __init__(self):
        self.payload = None

    def set(self, payload):
        self.payload = payload

    def clear(self):
        self.payload = None

    def active(self):
        return self.payload is not None


class SatTemporalReferenceAttention(nn.Module):
    """Spatiotemporally-local reference attention over the previous frame's
    satellite patch tokens.

    Every frame's satellite condition is a camera-centered, heading-aligned
    crop of one georeferenced map, so consecutive crops overlap almost fully
    and differ by a known ego-motion shift. Each latent query attends to the
    previous frame's satellite tokens in a small window around the
    geometrically corresponding patch — a spatiotemporal generalization of
    LocalReferenceCrossAttention. The out projection is zero-initialised:
    the stream is exactly inert at the start and grows only where gradients
    ask for it.
    """

    def __init__(self, query_dim, context_dim, heads, dim_head, dropout=0.0, reference_window=3):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = zero_module(nn.Linear(inner, query_dim))
        self.reference_window = max(1, int(reference_window))
        self.last_conf_mean = None

    def _window_grid(self, query_hw, token_hw, device, dtype):
        qh, qw = int(query_hw[0]), int(query_hw[1])
        th, tw = int(token_hw[0]), int(token_hw[1])
        qx = torch.linspace(-1.0, 1.0, qw, device=device, dtype=dtype).view(1, qw).expand(qh, qw)
        qy = torch.linspace(-1.0, 1.0, qh, device=device, dtype=dtype).view(qh, 1).expand(qh, qw)
        radius = self.reference_window // 2
        step_x = 2.0 / max(tw - 1, 1)
        step_y = 2.0 / max(th - 1, 1)
        offsets = [
            (ox * step_x, oy * step_y)
            for oy in range(-radius, radius + 1)
            for ox in range(-radius, radius + 1)
        ]
        base = torch.stack([qx.reshape(-1), qy.reshape(-1)], dim=-1).unsqueeze(1)  # (N,1,2)
        off = torch.tensor(offsets, device=device, dtype=dtype).view(1, len(offsets), 2)
        return base + off  # (N, K, 2); clamped after the per-sample shift

    def forward(self, x, sat_tokens_prev, shift_xy, token_grid=(14, 14), query_hw=None):
        """x: (b, N, c) latent queries. sat_tokens_prev: (b, T, ctx) previous
        frame's satellite patch tokens WITHOUT class/dynamic tokens.
        shift_xy: (b, 2) ego-motion shift of this frame's crop center inside
        the previous crop, in normalized crop coordinates (+x right, +y down).
        Returns (b, N, c) or None when no previous tokens exist.
        """
        if sat_tokens_prev is None:
            return None
        b, n, c = x.shape
        th, tw = int(token_grid[0]), int(token_grid[1])
        qh, qw = (int(query_hw[0]), int(query_hw[1])) if query_hw is not None else (th, tw)
        grid = self._window_grid((qh, qw), (th, tw), x.device, x.dtype)  # (N, K, 2)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1) + shift_xy.to(grid.dtype).view(b, 1, 1, 2)
        grid = grid.clamp(-1.0, 1.0)

        kt = self.to_k(self.norm_kv(sat_tokens_prev)).transpose(1, 2).reshape(b, -1, th, tw).to(grid.dtype)
        vt = self.to_v(self.norm_kv(sat_tokens_prev)).transpose(1, 2).reshape(b, -1, th, tw).to(grid.dtype)
        ks = torch.nn.functional.grid_sample(kt, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        vs = torch.nn.functional.grid_sample(vt, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        win = self.reference_window ** 2  # KxK window around the query patch
        # grid_sample output: (b, inner, N, win). Group inner into heads:
        # (b, heads, dh, N, win) -> per-head window attention.
        K = ks.reshape(b, self.heads, self.dim_head, qh * qw, win)
        V = vs.reshape(b, self.heads, self.dim_head, qh * qw, win)

        q = self.to_q(self.norm_q(x)).to(K.dtype)  # (b, N, inner)
        q = q.view(b, qh * qw, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (b, heads, N, dh)
        q = q.reshape(b * self.heads, qh * qw, 1, self.dim_head)
        Kw = K.permute(0, 1, 3, 4, 2).reshape(b * self.heads, qh * qw, win, self.dim_head)
        Vw = V.permute(0, 1, 3, 4, 2).reshape(b * self.heads, qh * qw, win, self.dim_head)
        attn = torch.softmax(q @ Kw.transpose(-1, -2) * self.scale, dim=-1)  # (b*heads, N, 1, win)
        out = (attn @ Vw).reshape(b, self.heads, qh * qw, self.dim_head)
        out = out.permute(0, 2, 1, 3).reshape(b, qh * qw, -1)
        out = self.to_out(out.to(x.dtype))
        with torch.no_grad():
            self.last_conf_mean = float(attn.detach().float().max(dim=-1).values.mean())
        return out


class RayPosteriorEvidenceFusion(nn.Module):
    """Preserve the posterior satellite path and add LiDAR as independent evidence."""

    def __init__(self, dim, lidar_gate_bias=-2.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.lidar_gate = zero_module(nn.Linear(dim, 1))
        nn.init.constant_(self.lidar_gate.bias, float(lidar_gate_bias))
        self.temporal_gate = None
        self.register_buffer("last_lidar_confidence", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_lidar_mask_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_lidar_message_ratio", torch.tensor(0.0), persistent=False)

    def enable_temporal(self, gate_bias=-6.0):
        """Add the third (temporal) evidence stream. Zero-initialised weights
        and a strongly negative bias keep the stream inert at the start, so
        enabling is a no-op on the frozen single-frame behaviour."""
        if self.temporal_gate is None:
            self.temporal_gate = zero_module(nn.Linear(self.dim, 1))
            nn.init.constant_(self.temporal_gate.bias, float(gate_bias))
            self.register_buffer("last_temporal_confidence", torch.tensor(0.0), persistent=False)

    @staticmethod
    def _prepare_mask(mask, x, query_hw=None):
        if mask is None:
            return x.new_zeros((x.shape[0], x.shape[1], 1))
        if mask.dim() == 4:
            if query_hw is None:
                return x.new_zeros((x.shape[0], x.shape[1], 1))
            mask = F.interpolate(mask.float(), size=query_hw, mode="area")
            mask = rearrange(mask, "b c h w -> b (h w) c")
        elif mask.dim() == 3:
            if mask.shape[1] != x.shape[1] and mask.shape[-1] == x.shape[1]:
                mask = mask.transpose(1, 2)
            if mask.shape[-1] != 1:
                mask = mask.mean(dim=-1, keepdim=True)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        else:
            return x.new_zeros((x.shape[0], x.shape[1], 1))
        if mask.shape[:2] != x.shape[:2]:
            return x.new_zeros((x.shape[0], x.shape[1], 1))
        return mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def forward(self, x, sat_ref, lidar_ref, query_hw=None, lidar_geometry_mask=None, temporal_ref=None, temporal_validity=None, sat_temporal_ref=None):
        if sat_ref is None:
            sat_ref = torch.zeros_like(x)
        if lidar_ref is None:
            lidar_ref = torch.zeros_like(x)
        geometry_confidence = self._prepare_mask(lidar_geometry_mask, x, query_hw=query_hw)
        learned_confidence = torch.sigmoid(self.lidar_gate(self.norm(x + sat_ref + lidar_ref)))
        lidar_confidence = geometry_confidence * learned_confidence
        lidar_message = lidar_confidence * lidar_ref

        temporal_message = None
        if self.temporal_gate is not None and temporal_ref is not None:
            temporal_validity = self._prepare_mask(temporal_validity, x, query_hw=query_hw)
            temporal_confidence = torch.sigmoid(self.temporal_gate(self.norm(x + sat_ref + temporal_ref)))
            temporal_message = temporal_validity * temporal_confidence * temporal_ref

        with torch.no_grad():
            self.last_lidar_confidence.copy_(
                lidar_confidence.float().mean().to(self.last_lidar_confidence.device)
            )
            self.last_lidar_mask_mean.copy_(
                geometry_confidence.float().mean().to(self.last_lidar_mask_mean.device)
            )
            ratio = lidar_message.float().norm(dim=-1).mean() / sat_ref.float().norm(dim=-1).mean().clamp_min(1e-6)
            self.last_lidar_message_ratio.copy_(ratio.to(self.last_lidar_message_ratio.device))
            if temporal_message is not None:
                self.last_temporal_confidence.copy_(
                    (temporal_validity * temporal_confidence).float().mean().to(self.last_temporal_confidence.device)
                )
        if sat_temporal_ref is not None:
            # Route-C stream: the attention's out projection is zero-initialised,
            # so an additive merge is exactly inert at the start — no gate to
            # slowly open (the v1 lesson), the gradient scales the weights.
            out = sat_ref + lidar_message
            if temporal_message is not None:
                out = out + temporal_message
            return out + sat_temporal_ref
        if temporal_message is None:
            return sat_ref + lidar_message
        return sat_ref + lidar_message + temporal_message


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.,
        context_dim=None,
        gated_ff=True,
        checkpoint=False,
        use_lidar_cross_attention=False,
        lidar_context_dim=None,
        lidar_reference_window=3,
        ray_evidence_sat_bias=4.0,
        ray_evidence_lidar_bias=-4.0,
        ray_evidence_null_bias=-6.0,
        ray_fusion_mode="ray_evidence",
        use_lidar_ray_posterior=False,
        lidar_posterior_log_depth_sigma=0.35,
        lidar_posterior_strength=2.0,
        lidar_message_gate_bias=-2.0,
    ):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.temporal_hub = None
        self.last_fused_delta = None
        self.frozen_fused_delta = None
        self.sat_temporal_attn = None
        self._sat_payload_cache = None
        self.history_attn = None
        self.history_ratio = None
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            use_lidar_ray_posterior=use_lidar_ray_posterior,
            lidar_posterior_log_depth_sigma=lidar_posterior_log_depth_sigma,
            lidar_posterior_strength=lidar_posterior_strength,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.use_lidar_cross_attention = bool(use_lidar_cross_attention)
        self.ray_fusion_mode = str(ray_fusion_mode or "ray_evidence")
        if self.ray_fusion_mode not in {"ray_evidence", "ray_posterior"}:
            raise ValueError(f"Unsupported ray_fusion_mode: {self.ray_fusion_mode}")
        if self.use_lidar_cross_attention:
            self.norm_lidar = nn.LayerNorm(dim)
            self.norm_evidence = nn.LayerNorm(dim)
            self.attn_lidar = LocalReferenceCrossAttention(
                query_dim=dim,
                context_dim=default(lidar_context_dim, context_dim),
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                reference_window=lidar_reference_window,
            )
            if self.ray_fusion_mode == "ray_posterior":
                self.ray_evidence_attn = None
                self.ray_posterior_fusion = RayPosteriorEvidenceFusion(
                    dim=dim,
                    lidar_gate_bias=lidar_message_gate_bias,
                )
            else:
                self.ray_evidence_attn = RayAlignedEvidenceAttention(
                    dim=dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    sat_bias=ray_evidence_sat_bias,
                    lidar_bias=ray_evidence_lidar_bias,
                    null_bias=ray_evidence_null_bias,
                )
                self.ray_posterior_fusion = None

    def _build_temporal(self, latent_hw, x):
        """Consume the previous frame's cached fused posterior, transported to
        this module's resolution. Called OUTSIDE gradient checkpointing so the
        recompute pass sees identical tensors; the cache itself is never
        touched here."""
        payload = self.temporal_hub.payload if self.temporal_hub is not None else None
        delta = self.frozen_fused_delta if self.frozen_fused_delta is not None else self.last_fused_delta
        if payload is None or delta is None or "transport" not in payload:
            return None, None
        b, n, c = x.shape
        h, w = int(latent_hw[0]), int(latent_hw[1])
        grid, validity = payload["transport"]((h, w), x.device, x.dtype)
        delta_map = delta.reshape(b, h, w, c).permute(0, 3, 1, 2)
        warped = F.grid_sample(
            delta_map, grid.to(dtype=delta_map.dtype), mode="bilinear", padding_mode="zeros", align_corners=False
        )
        temporal_ref = warped.permute(0, 2, 3, 1).reshape(b, n, c)
        strength = float(payload.get("strength", 1.0))
        if strength != 1.0:
            temporal_ref = temporal_ref * strength
        return temporal_ref, validity

    def _build_sat_temporal(self, latent_hw, x):
        """Route-C payload fetch, outside checkpointing: previous frame's
        satellite tokens + this frame's ego-motion shift. Snapshot semantics:
        payload data is stashed once per frame boundary and reused for every
        DDIM step of the frame."""
        payload = self.temporal_hub.payload if self.temporal_hub is not None else None
        if payload is None or self.sat_temporal_attn is None:
            return None
        if "sat_tokens_prev" not in payload or payload.get("sat_tokens_prev") is None:
            return None
        key = (id(payload), )
        if self._sat_payload_cache is not None and self._sat_payload_cache[0] == key:
            pack = self._sat_payload_cache[1]
        else:
            pack = {
                "tokens": payload["sat_tokens_prev"],
                "shift": payload.get("sat_shift_xy"),
                "grid": payload.get("sat_token_grid", (14, 14)),
            }
            self._sat_payload_cache = (key, pack)
        import os as _os
        if _os.environ.get("SAT_TEMPORAL_DEBUG"):
            print(f"[sat-dbg] fetch tokens={pack['tokens'].shape} "
                  f"requires_grad={pack['tokens'].requires_grad} data_ptr={pack['tokens'].data_ptr()}", flush=True)
        return pack

    def _build_history_pack(self):
        """v2 history payload fetch (pre-checkpoint, like the v1/route-C packs).
        No cross-step caching: tokens are freshly built per frame boundary."""
        payload = self.temporal_hub.payload if self.temporal_hub is not None else None
        if payload is None or self.history_attn is None or "history_tokens" not in payload:
            return None
        pack = {
            "history_tokens": payload["history_tokens"],
            "history_hw": payload.get("history_hw"),
            "has_history": bool(payload.get("has_history", True)),
        }
        if "history_grid" in payload or "history_valid" in payload:
            if "history_grid" not in payload or "history_valid" not in payload:
                raise ValueError("history_grid and history_valid must both be in history payload")
            pack["history_grid"] = payload["history_grid"]
            pack["history_valid"] = payload["history_valid"]
        return pack

    def forward(self, x, context=None, lidar_context=None, lidar_evidence=None, lidar_geometry_mask=None, latent_hw=None, left_camera_k=None,  gt_shift_x=None, gt_shift_y=None, theta=None):
        if self.use_lidar_cross_attention and lidar_context is not None:
            temporal_ref, temporal_validity = self._build_temporal(latent_hw, x)
            sat_pack = self._build_sat_temporal(latent_hw, x)
            hist_pack = self._build_history_pack()
            out = checkpoint(
                self._forward,
                (x, context, lidar_context, lidar_evidence, lidar_geometry_mask, latent_hw, left_camera_k, gt_shift_x, gt_shift_y, theta, temporal_ref, temporal_validity, sat_pack, hist_pack),
                self.parameters(),
                self.checkpoint,
            )
            # Promote outside the checkpointed region: a recompute pass may
            # overwrite _pending_delta, but the recomputed value is identical
            # (checkpoint preserves RNG), so promotion is re-run safe.
            pending = getattr(self, "_pending_delta", None)
            if pending is not None:
                self.last_fused_delta = pending
                self._pending_delta = None
            return out
        return checkpoint(
            self._forward_without_lidar,
            (x, context, left_camera_k, gt_shift_x, gt_shift_y, theta),
            self.parameters(),
            self.checkpoint,
        )

    def _forward_without_lidar(self, x, context=None, left_camera_k=None,  gt_shift_x=None, gt_shift_y=None, theta=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, left_camera_k = left_camera_k, gt_shift_x = gt_shift_x, gt_shift_y = gt_shift_y, theta = theta) + x
        x = self.ff(self.norm3(x)) + x
        return x

    def _forward(self, x, context=None, lidar_context=None, lidar_evidence=None, lidar_geometry_mask=None, latent_hw=None, left_camera_k=None,  gt_shift_x=None, gt_shift_y=None, theta=None, temporal_ref=None, temporal_validity=None, sat_pack=None, hist_pack=None):
        x = self.attn1(self.norm1(x)) + x
        x_base = x
        sat_delta = self.attn2(
            self.norm2(x_base),
            context=context,
            left_camera_k=left_camera_k,
            gt_shift_x=gt_shift_x,
            gt_shift_y=gt_shift_y,
            theta=theta,
            ray_depth_evidence=lidar_evidence,
        )
        lidar_delta = self.attn_lidar(self.norm_lidar(x_base), lidar_context, query_hw=latent_hw)
        sat_temporal_delta = None
        if sat_pack is not None:
            n = x_base.shape[1]
            qh = int(latent_hw[0]) if latent_hw is not None else 16
            qw = n // qh
            sat_temporal_delta = self.sat_temporal_attn(
                x_base, sat_pack["tokens"], sat_pack["shift"],
                token_grid=sat_pack["grid"], query_hw=(qh, qw),
            )
            import os as _os
            if _os.environ.get("SAT_TEMPORAL_DEBUG") and sat_temporal_delta is not None:
                print(f"[sat-dbg] fwd delta requires_grad={sat_temporal_delta.requires_grad} "
                      f"absmax={float(sat_temporal_delta.abs().max())}", flush=True)
        if self.ray_fusion_mode == "ray_posterior":
            fused_delta = self.ray_posterior_fusion(
                self.norm_evidence(x_base),
                sat_delta,
                lidar_delta,
                query_hw=latent_hw,
                lidar_geometry_mask=lidar_geometry_mask,
                temporal_ref=temporal_ref,
                temporal_validity=temporal_validity,
                sat_temporal_ref=sat_temporal_delta,
            )
        else:
            fused_delta = self.ray_evidence_attn(
                self.norm_evidence(x_base),
                sat_delta,
                lidar_delta,
                query_hw=latent_hw,
                lidar_geometry_mask=lidar_geometry_mask,
            )
        # v2 history readout: the query sees the current conditions' fused
        # summary (detached — routing signal only), the residual is added on
        # top of the fused delta. Zero-init keeps this inert until trained.
        hist_delta = None
        if hist_pack is not None and self.history_attn is not None:
            if getattr(self.history_attn, "uses_geometry", False):
                hist_delta = self.history_attn(
                    x_base,
                    fused_delta.detach(),
                    hist_pack["history_tokens"],
                    has_history=hist_pack["has_history"],
                    history_grid=hist_pack.get("history_grid"),
                    history_valid=hist_pack.get("history_valid"),
                    query_hw=latent_hw,
                    history_hw=hist_pack.get("history_hw"),
                )
            else:
                hist_delta = self.history_attn(
                    x_base,
                    fused_delta.detach(),
                    hist_pack["history_tokens"],
                    has_history=hist_pack["has_history"],
                )
        if hist_delta is not None:
            with torch.no_grad():
                denom = float(fused_delta.detach().float().norm(dim=-1).mean())
                self.history_ratio = (
                    float(hist_delta.detach().float().norm(dim=-1).mean()) / max(denom, 1e-6)
                )
            x = x_base + fused_delta + hist_delta
        else:
            x = x_base + fused_delta
        x = self.ff(self.norm3(x)) + x
        # Stash the fused posterior for the next frame's temporal evidence.
        # The caller (forward) promotes it to last_fused_delta OUTSIDE the
        # checkpointed region, so gradient-checkpoint recomputes can never
        # corrupt the previous frame's cache.
        if self.temporal_hub is not None and self.ray_fusion_mode == "ray_posterior":
            self._pending_delta = fused_delta.detach()
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.,
        context_dim=None,
        checkpoint=False,
        use_lidar_cross_attention=False,
        lidar_context_dim=None,
        lidar_reference_window=3,
        ray_evidence_sat_bias=4.0,
        ray_evidence_lidar_bias=-4.0,
        ray_evidence_null_bias=-6.0,
        ray_fusion_mode="ray_evidence",
        use_lidar_ray_posterior=False,
        lidar_posterior_log_depth_sigma=0.35,
        lidar_posterior_strength=2.0,
        lidar_message_gate_bias=-2.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                checkpoint=checkpoint,
                use_lidar_cross_attention=use_lidar_cross_attention,
                lidar_context_dim=lidar_context_dim,
                lidar_reference_window=lidar_reference_window,
                ray_evidence_sat_bias=ray_evidence_sat_bias,
                ray_evidence_lidar_bias=ray_evidence_lidar_bias,
                ray_evidence_null_bias=ray_evidence_null_bias,
                ray_fusion_mode=ray_fusion_mode,
                use_lidar_ray_posterior=use_lidar_ray_posterior,
                lidar_posterior_log_depth_sigma=lidar_posterior_log_depth_sigma,
                lidar_posterior_strength=lidar_posterior_strength,
                lidar_message_gate_bias=lidar_message_gate_bias,
            )
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, lidar_context=None, lidar_evidence=None, lidar_geometry_mask=None, left_camera_k=None,  gt_shift_x=None, gt_shift_y=None, theta=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context, lidar_context=lidar_context, lidar_evidence=lidar_evidence, lidar_geometry_mask=lidar_geometry_mask, latent_hw=(h, w), left_camera_k = left_camera_k, gt_shift_x = gt_shift_x, gt_shift_y = gt_shift_y, theta = theta)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
