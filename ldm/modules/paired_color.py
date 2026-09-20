"""Mild pair-shared RGB augmentation and fixed-geometry forward gathering."""
import torch

def sample_pair_color(batch_size, generator, probability=0.7):
    if not 0 <= probability <= 1 or batch_size < 1:
        raise ValueError('invalid pair augmentation probability or batch size')
    active=torch.rand(batch_size,generator=generator)<probability
    def draw(width,spread):
        value=1+(torch.rand(batch_size,width,generator=generator)*2-1)*spread
        return torch.where(active[:,None],value,torch.ones_like(value))
    return dict(active=active,brightness=draw(1,.1),contrast=draw(1,.1),saturation=draw(1,.1),channel_gain=draw(3,.05))

def apply_pair_color(rgb, params):
    if rgb.ndim!=4 or rgb.shape[1]!=3 or not rgb.is_floating_point():
        raise ValueError('RGB must be floating point [B,3,H,W]')
    if len(params['active'])!=rgb.shape[0]:raise ValueError('one transform is required per pair')
    def value(name):return params[name].to(device=rgb.device,dtype=rgb.dtype)[:,:,None,None]
    luma=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    out=luma+value('saturation')*(rgb-luma)
    out=((out-.5)*value('contrast')+.5)*value('brightness')*value('channel_gain')
    # Preserve unaugmented samples bit-for-bit, including rounding arithmetic.
    active=params['active'].to(rgb.device)[:,None,None,None]
    return torch.where(active,out.clamp(0,1),rgb)

def warp_augmented_rgb(previous_rgb, source_flat_index, valid_mask):
    b,c,h,w=previous_rgb.shape
    if source_flat_index.shape!=(b,h,w) or valid_mask.shape!=(b,1,h,w) or valid_mask.dtype!=torch.bool:
        raise ValueError('source index and mask must match RGB resolution')
    index=source_flat_index.to(device=previous_rgb.device,dtype=torch.long)
    valid=valid_mask.to(previous_rgb.device)
    if torch.any((index[valid[:,0]]<0)|(index[valid[:,0]]>=h*w)):
        raise ValueError('valid source index out of bounds')
    gathered=torch.gather(previous_rgb.flatten(2),2,index.clamp(0,h*w-1).flatten(1)[:,None].expand(-1,c,-1))
    return gathered.reshape(b,c,h,w)*valid.to(previous_rgb.dtype)
