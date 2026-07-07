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
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

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

    def forward(self, x, context=None, mask=None, left_camera_k=None,  gt_shift_x=None, gt_shift_y=None, theta=None):
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

            reference_points_rebatch = self.gen_KITTI_sat2grd.sat2grd_h(int(feat_size), int(num_query**(1/2)/2), int(num_query**(1/2)*2), left_camera_k, gt_shift_x, gt_shift_y, theta)
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
            attention_weights = self.attention_weights(x).view(bs, x_len, heads, self.num_points)
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.transpose(1, 2).reshape(bs, heads, x_len, self.num_points)#bs*heads, num_points
            
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
        self.to_q = zero_module(nn.Linear(dim, inner_dim, bias=False))
        self.to_k = zero_module(nn.Linear(dim, inner_dim, bias=False))
        self.ray_proj = zero_module(nn.Linear(dim, dim, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.null_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.evidence_bias = nn.Parameter(
            torch.tensor([float(sat_bias), float(lidar_bias), float(null_bias)], dtype=torch.float32)
        )
        self.register_buffer("last_evidence_sat_weight", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_evidence_lidar_weight", torch.tensor(0.0), persistent=False)
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
            entropy = -(attn.detach().float().clamp_min(1e-12) * attn.detach().float().clamp_min(1e-12).log()).sum(dim=-1)
            entropy_norm = entropy / math.log(max(2, int(attn.shape[-1])))
            self.last_evidence_sat_weight.copy_(weights[0].to(self.last_evidence_sat_weight.device))
            self.last_evidence_lidar_weight.copy_(weights[1].to(self.last_evidence_lidar_weight.device))
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
        lidar_gate_init=1e-3,
        lidar_evidence_channels=0,
        lidar_attention_mode="token",
        lidar_reference_window=3,
        lidar_fusion_mode="sequential",
        ray_evidence_sat_bias=4.0,
        ray_evidence_lidar_bias=-4.0,
        ray_evidence_null_bias=-6.0,
    ):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.use_lidar_cross_attention = bool(use_lidar_cross_attention)
        self.lidar_evidence_channels = int(lidar_evidence_channels or 0)
        if self.use_lidar_cross_attention:
            self.norm_lidar = nn.LayerNorm(dim)
            self.norm_evidence = nn.LayerNorm(dim)
            self.lidar_attention_mode = str(lidar_attention_mode or "token")
            if self.lidar_attention_mode not in {"token", "reference"}:
                raise ValueError(f"unknown lidar_attention_mode: {self.lidar_attention_mode}")
            self.lidar_fusion_mode = str(lidar_fusion_mode or "sequential")
            if self.lidar_fusion_mode not in {"sequential", "ray_evidence"}:
                raise ValueError(f"unknown lidar_fusion_mode: {self.lidar_fusion_mode}")
            if self.lidar_attention_mode == "reference":
                self.attn_lidar = LocalReferenceCrossAttention(
                    query_dim=dim,
                    context_dim=default(lidar_context_dim, context_dim),
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    reference_window=lidar_reference_window,
                )
            else:
                self.attn_lidar = TokenCrossAttention(
                    query_dim=dim,
                    context_dim=default(lidar_context_dim, context_dim),
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )
            if self.lidar_fusion_mode == "ray_evidence":
                self.ray_evidence_attn = RayAlignedEvidenceAttention(
                    dim=dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                    sat_bias=ray_evidence_sat_bias,
                    lidar_bias=ray_evidence_lidar_bias,
                    null_bias=ray_evidence_null_bias,
                )
            else:
                gate_init = min(max(float(lidar_gate_init), 1e-6), 1.0 - 1e-6)
                self.lidar_gate = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init))))
                if self.lidar_evidence_channels > 0:
                    self.evidence_router = nn.Sequential(
                        nn.Linear(self.lidar_evidence_channels, dim),
                        nn.SiLU(),
                        zero_module(nn.Linear(dim, 1)),
                    )
                self.lidar_gate_cap = 1.0

    def forward(self, x, context=None, lidar_context=None, lidar_evidence=None, lidar_geometry_mask=None, latent_hw=None, left_camera_k=None,  gt_shift_x=None, gt_shift_y=None, theta=None):
        if self.use_lidar_cross_attention and lidar_context is not None:
            return checkpoint(
                self._forward,
                (x, context, lidar_context, lidar_evidence, lidar_geometry_mask, latent_hw, left_camera_k, gt_shift_x, gt_shift_y, theta),
                self.parameters(),
                self.checkpoint,
            )
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

    def _lidar_gate(self, x, lidar_evidence=None, latent_hw=None):
        gate_cap = torch.as_tensor(
            getattr(self, "lidar_gate_cap", 1.0),
            dtype=x.dtype,
            device=x.device,
        ).clamp(0.0, 1.0)
        gate = torch.sigmoid(self.lidar_gate.to(dtype=x.dtype, device=x.device))
        if (
            lidar_evidence is None
            or self.lidar_evidence_channels <= 0
            or not hasattr(self, "evidence_router")
            or latent_hw is None
        ):
            return gate_cap * gate
        h, w = latent_hw
        evidence = F.interpolate(lidar_evidence.float(), size=(int(h), int(w)), mode="area")
        evidence = rearrange(evidence, "b c h w -> b (h w) c").to(device=x.device, dtype=x.dtype)
        if evidence.shape[-1] != self.lidar_evidence_channels:
            return gate_cap * gate
        return gate_cap * torch.sigmoid(
            self.evidence_router(evidence) + self.lidar_gate.to(dtype=x.dtype, device=x.device)
        )

    def _forward(self, x, context=None, lidar_context=None, lidar_evidence=None, lidar_geometry_mask=None, latent_hw=None, left_camera_k=None,  gt_shift_x=None, gt_shift_y=None, theta=None):
        x = self.attn1(self.norm1(x)) + x
        if (
            lidar_context is not None
            and getattr(self, "lidar_fusion_mode", "sequential") == "ray_evidence"
            and hasattr(self, "ray_evidence_attn")
        ):
            x_base = x
            sat_delta = self.attn2(
                self.norm2(x_base),
                context=context,
                left_camera_k=left_camera_k,
                gt_shift_x=gt_shift_x,
                gt_shift_y=gt_shift_y,
                theta=theta,
            )
            lidar_delta = self.attn_lidar(self.norm_lidar(x_base), lidar_context, query_hw=latent_hw)
            x = x_base + self.ray_evidence_attn(
                self.norm_evidence(x_base),
                sat_delta,
                lidar_delta,
                query_hw=latent_hw,
                lidar_geometry_mask=lidar_geometry_mask,
            )
            x = self.ff(self.norm3(x)) + x
            return x

        x = self.attn2(self.norm2(x), context=context, left_camera_k = left_camera_k, gt_shift_x = gt_shift_x, gt_shift_y = gt_shift_y, theta = theta) + x
        if lidar_context is not None:
            lidar_delta = self.attn_lidar(self.norm_lidar(x), lidar_context, query_hw=latent_hw)
            x = self._lidar_gate(x, lidar_evidence=lidar_evidence, latent_hw=latent_hw) * lidar_delta + x
        x = self.ff(self.norm3(x)) + x
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
        lidar_gate_init=1e-3,
        lidar_evidence_channels=0,
        lidar_attention_mode="token",
        lidar_reference_window=3,
        lidar_fusion_mode="sequential",
        ray_evidence_sat_bias=4.0,
        ray_evidence_lidar_bias=-4.0,
        ray_evidence_null_bias=-6.0,
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
                lidar_gate_init=lidar_gate_init,
                lidar_evidence_channels=lidar_evidence_channels,
                lidar_attention_mode=lidar_attention_mode,
                lidar_reference_window=lidar_reference_window,
                lidar_fusion_mode=lidar_fusion_mode,
                ray_evidence_sat_bias=ray_evidence_sat_bias,
                ray_evidence_lidar_bias=ray_evidence_lidar_bias,
                ray_evidence_null_bias=ray_evidence_null_bias,
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
