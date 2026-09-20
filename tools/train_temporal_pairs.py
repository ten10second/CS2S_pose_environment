#!/usr/bin/env python3
"""Train a single UNet with aligned history input and RGB/edge supervision."""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from tools.temporal_data import load_settings_args,load_base,encode_conditions,encode_latent,stack_samples,evaluation_device,parse_sample_id
from tools.cache_centered_conditions import load_reference, reference_path, compare_rgb, previous_rgb_sample
from tools.infer_temporal import tree_to
from ldm.modules.temporal_pair_training import (configure_trainables,make_history_condition,epsilon_prediction_loss,seed_training_step,save_history_checkpoint)
from ldm.modules.temporal_image_loss import temporal_image_loss
from ldm.modules.temporal_amp import optimizer_step_with_retry
from ldm.modules.paired_color import sample_pair_color,apply_pair_color,warp_augmented_rgb


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',required=True)
    p.add_argument('--manifest',required=True,help='JSON items with previous/current IDs and reference_npz')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--device',default='cuda:4')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--wait-reference-seconds',type=int,default=0,help='Wait for atomically written references while the producer runs')
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--rgb-weight',type=float,required=True)
    p.add_argument('--edge-weight',type=float,required=True)
    p.add_argument('--color-probability',type=float,default=.7)
    p.add_argument('--history-off',action='store_true',help='Matched no-history control, with same expanded UNet')
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--max-pairs',type=int,default=0,help='Explicit subset for smoke tests; 0 means all')
    p.add_argument('--max-steps',type=int,default=0,help='Explicit bounded smoke; 0 means all epochs')
    p.add_argument('--no-checkpoint',action='store_true',help='Smoke only: discard updates when process exits')
    p.add_argument('--save-every-epochs',type=int,default=20)
    p.add_argument('--keep-checkpoints',type=int,default=2)
    a=p.parse_args(argv)
    if min(a.epochs,a.batch_size,a.save_every_epochs,a.keep_checkpoints)<1 or min(a.num_workers,a.max_pairs,a.max_steps,a.wait_reference_seconds)<0:
        p.error('invalid count')
    if not all(math.isfinite(v) and v>=0 for v in (a.rgb_weight,a.edge_weight)) or not math.isfinite(a.lr) or a.lr<=0:
        p.error('loss weights must be nonnegative and LR positive')
    if not 0<=a.color_probability<=1:p.error('color probability must be in [0,1]')
    if a.no_checkpoint and not a.max_steps:p.error('--no-checkpoint requires a bounded --max-steps smoke')
    return a


class ReferencePairs(Dataset):
    """Existing reference files + original geographic training manifest; no geometry model in loader."""
    def __init__(self,base,entries,root,wait_reference_seconds=0):
        self.base,self.entries,self.root=base,list(entries),Path(root)
        self.wait_reference_seconds=wait_reference_seconds
        self.index={r['sample_id']:i for i,r in enumerate(base.records)}
        seen=set()
        for e in self.entries:
            if e.get('split')!='train' or e.get('source_split','train')!='train':
                raise ValueError('training accepts only original train split')
            pair=(e['previous'],e['current'])
            if pair in seen:raise ValueError('duplicate training pair')
            seen.add(pair)
            if any(s not in self.index for s in pair):raise ValueError('pair absent from original training manifest')
            pd,pf=parse_sample_id(pair[0]);cd,cf=parse_sample_id(pair[1])
            if pd!=cd or cf!=pf+1:raise ValueError('pair must be consecutive within a drive')
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        e=self.entries[index]
        ref=load_reference(reference_path(self.root,e),e,self.wait_reference_seconds)
        sample=self.base[self.index[e['current']]]
        prev=previous_rgb_sample(self.base,self.base.records[self.index[e['previous']]])['grd_left_imgs']
        compare_rgb('previous RGB',ref['previous_rgb'],prev)
        compare_rgb('target RGB',ref['current_rgb'],sample['grd_left_imgs'])
        return dict(ref,sample=sample,name=e['name'])


def collate_references(items):return items


def concat_conditions(values):
    """Batch the complete LiDAR feature/mask pyramid, preserving sample order."""
    first=values[0]
    if torch.is_tensor(first):
        return torch.cat(values,dim=0)
    if isinstance(first,dict):
        if any(set(v)!=set(first) for v in values):raise ValueError('condition keys differ')
        return {k:concat_conditions([v[k] for v in values]) for k in first}
    if isinstance(first,(list,tuple)):
        if any(len(v)!=len(first) for v in values):raise ValueError('condition pyramid lengths differ')
        return type(first)(concat_conditions([v[i] for v in values]) for i in range(len(first)))
    if all(v is None for v in values):return None
    raise TypeError('unsupported conditioning leaf: '+str(type(first)))


def prepare_batch(model,items,device,seed,color_probability=.7,history_off=False):
    conditions=[]
    # Encode each LiDAR tensor individually to bound the 576-channel input peak.
    for item in items:
        cond,_=encode_conditions(model,stack_samples([item['sample']]),device)
        conditions.append(cond)
    cond=concat_conditions(conditions)
    prev=torch.stack([i['previous_rgb'] for i in items]).to(device)
    target=torch.stack([i['current_rgb'] for i in items]).to(device)
    params=sample_pair_color(len(items),torch.Generator().manual_seed(seed),color_probability)
    prev,target=apply_pair_color(prev,params),apply_pair_color(target,params)
    valid,measured,estimated=[torch.stack([i[k] for i in items]).to(device) for k in ('valid','measured','estimated')]
    source=torch.stack([i['source_flat_index'] for i in items]).to(device)
    warp=warp_augmented_rgb(prev,source,valid)
    history=None if history_off else make_history_condition(model,warp,valid,measured,estimated)
    z=encode_latent(model,target).float()
    return cond,z,history,target


def main(argv=None):
    args=parse_args(argv);device=evaluation_device(args.device)
    torch.set_num_threads(1)
    out=Path(args.out_dir)
    if out.exists():raise ValueError('refusing overwrite: '+str(out))
    out.mkdir(parents=True)
    settings=load_settings_args(args.settings)
    settings.batch_size=args.batch_size;settings.num_workers=args.num_workers
    seed_training_step(args.seed,device)
    model,base,cfg=load_base(settings,device)
    from utils.util import instantiate_from_config
    dataset=instantiate_from_config(cfg.data.params.train)
    payload=json.loads(Path(args.manifest).read_text());entries=payload['items'] if isinstance(payload,dict) else payload
    if not isinstance(entries,list) or not entries:raise ValueError('empty training manifest')
    # Membership is checked against the original split; no ad hoc resplitting.
    if args.max_pairs:entries=entries[:args.max_pairs]
    pairs=ReferencePairs(dataset,entries,Path(args.manifest).parent,args.wait_reference_seconds)
    params=configure_trainables(model)
    optimizer=torch.optim.AdamW(params,lr=args.lr)
    scaler=GradScaler(enabled=device.type=='cuda')
    run=dict(args=vars(args),base_checkpoint=base,training_pairs=len(pairs),trainable_parameters=sum(p.numel() for p in params),
             objective='epsilon_mse + sqrt_alpha * (rgb_weight * charbonnier + edge_weight * sobel)',
             checkpoint_kind='weights_only',history_channels=4,mask_channels=['valid','measured','estimated'])
    (out/'args.json').write_text(json.dumps(run,indent=2))
    step=0;start=time.monotonic();limit=False
    for epoch in range(1,args.epochs+1):
        loader=DataLoader(pairs,batch_size=args.batch_size,shuffle=True,drop_last=False,num_workers=args.num_workers,
                          collate_fn=collate_references,generator=torch.Generator().manual_seed(args.seed+epoch),
                          **({'multiprocessing_context':'spawn'} if args.num_workers else {}))
        for items in loader:
            step+=1;seed=args.seed+1009*step
            seed_training_step(seed,device)
            cond,z,history,target=prepare_batch(model,items,device,seed,args.color_probability,args.history_off)
            # Preserve the existing satellite condition dropout, shared across controls by seed.
            drop=torch.rand((len(items),1,1),device=device)<float(model.satellite_condition_dropout_prob)
            cond=dict(cond);cond['context']=cond['context']*(~drop)
            def closure():
                with autocast(enabled=device.type=='cuda'):
                    _,aux=epsilon_prediction_loss(model.DDPM,z,cond,history,seed)
                return temporal_image_loss(aux['model_out'],aux['noise'],aux['x_noisy'],aux['t'],model.DDPM.alphas_cumprod,
                                           target,model.pre_AE_model,float(model.scale_factor),args.rgb_weight,args.edge_weight)
            loss,metrics,norm,retries=optimizer_step_with_retry(closure,params,optimizer,scaler)
            input_grad=model.DDPM.denoise_model.input_blocks[0][0].weight.grad
            rec=dict(step=step,epoch=epoch,names=[i['name'] for i in items],loss=float(loss.detach()),
                     metrics={k:float(v) for k,v in metrics.items()},gradient_norm=norm,amp_retries=retries,
                     history_input_grad=float(input_grad[:,4:].float().norm()),seconds=time.monotonic()-start,
                     cuda_peak_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            (out/'status.json').write_text(json.dumps(dict(rec,phase='training'),indent=2))
            if step<=3 or step%25==0:print(json.dumps(rec),flush=True)
            if args.max_steps and step>=args.max_steps:limit=True;break
        if not args.no_checkpoint and (epoch%args.save_every_epochs==0 or epoch==args.epochs or limit):
            save_history_checkpoint(out/'checkpoints'/('step_%07d.pt'%step),model,base,step,vars(args))
            saved=sorted((out/'checkpoints').glob('step_*.pt'))
            for p in saved[:-args.keep_checkpoints]:p.unlink()
        if limit:break
    done=dict(step=step,phase='smoke_complete' if args.max_steps else 'complete',checkpoint_saved=not args.no_checkpoint)
    (out/'status.json').write_text(json.dumps(done,indent=2));(out/'done.json').write_text(json.dumps(done,indent=2))
    print(json.dumps(done),flush=True)


if __name__=='__main__':main()
