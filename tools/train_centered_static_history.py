"""Frozen-base centered history training with pair-shared color augmentation."""
from __future__ import annotations
import argparse
import hashlib
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from torch.cuda.amp import GradScaler,autocast
from torch.utils.data.distributed import DistributedSampler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ldm.modules.paired_color import sample_pair_color,apply_pair_color,warp_augmented_rgb
from ldm.modules.temporal_amp import optimizer_step_with_retry
from ldm.modules.temporal_pair_training import epsilon_prediction_loss,anchor_trainable_loss,seed_training_step,state_dict_sha256
from tools.train_temporal_pairs import load_base,rank_info,encode_latent
from tools.train_static_history import (load_settings_args,recursive_batch_to_device,stable_pair_seed,
    fixed_epsilon_monitor,frozen_model_versions)
from tools.train_dense_static_history import named_adapter_grad_norms,save_rgb
from tools.infer_temporal import tree_to,sample_frame,decode,load_static_adapter,tensor_hash


def parse_args():
 p=argparse.ArgumentParser(__doc__)
 for key in ('settings','data-root','cache-root','out-dir'):p.add_argument('--'+key,required=True)
 p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=12)
 p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=3407)
 p.add_argument('--color-probability',type=float,default=.7);p.add_argument('--save-every-epochs',type=int,default=5)
 p.add_argument('--keep-checkpoints',type=int,default=2);p.add_argument('--local-rank',type=int,default=-1)
 p.add_argument('--smoke-steps',type=int,default=0);p.add_argument('--resume',default='')
 return p.parse_args()


def load_json(path):return json.loads(Path(path).read_text())

def atomic_json(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def plain_unet(model):
 m=model.DDPM.denoise_model
 return m.module if hasattr(m,'module') else m


@contextmanager
def unwrapped(model):
 original=model.DDPM.denoise_model
 model.DDPM.denoise_model=plain_unet(model)
 try:yield
 finally:model.DDPM.denoise_model=original


def item_history(item,device):
 valid=item['valid'].unsqueeze(0).to(device)
 rgb=item.get('warp_rgb')
 if rgb is None:rgb=warp_augmented_rgb(item['previous_rgb'][None].to(device),item['source_flat_index'][None].to(device),valid)[0].cpu()
 return dict(latent=torch.zeros_like(item['z_cur']).to(device),dense_rgb=rgb[None].to(device),
    dense_valid=valid,dense_measured=item['measured'][None].to(device),dense_estimated=item['estimated'][None].to(device),
    enabled=torch.ones(1,dtype=torch.bool,device=device))


def eval_item(item,device):
 return dict(item,history=item_history(item,device))


def evaluate(model,items,step,out,device,seed,timesteps=(100,500,900)):
 reader=plain_unet(model).temporal_history;reader.eval()
 records=[]
 with unwrapped(model):
  for index,item in enumerate(items):
   current=eval_item(item,device);donor=items[(index+1)%len(items)]
   wrong=dict(current);wrong['history']=dict(current['history'])
   with torch.no_grad(),autocast(enabled=device.type=='cuda'):
    wrong['history']['dense_features']=reader.encode_history(item_history(donor,device)).detach()
   for t in timesteps:
    fixed_seed=stable_pair_seed(seed,item['name'],f'monitor_t{t}')
    outputs={}
    for mode,example,use_history in [('off',current,False),('correct',current,True),('wrong',wrong,True)]:
     rec=fixed_epsilon_monitor(model,example,fixed_seed,device,use_history,t)
     rec.update(step=step,name=item['name'],split=item['split'],mode=mode)
     outputs[mode]=rec;records.append(rec)
    # Exact H=0 control, while valid and fusion remain active.
    if index==0:
     zero=dict(current);zero['history']=dict(current['history'])
     zero['history']['dense_features']=torch.zeros_like(wrong['history']['dense_features'])
     rec=fixed_epsilon_monitor(model,zero,fixed_seed,device,True,t)
     if rec['pred_hash']!=outputs['off']['pred_hash']:raise RuntimeError('centered H=0 did not reproduce OFF')
     rec.update(step=step,name=item['name'],split=item['split'],mode='zero_features');records.append(rec)
 with (out/'monitor.jsonl').open('a') as f:
  for r in records:f.write(json.dumps(r)+'\n')
 result={}
 for split in sorted({r['split'] for r in records}):
  result[split]={m:sum(r['loss'] for r in records if r['split']==split and r['mode']==m)/sum(1 for r in records if r['split']==split and r['mode']==m)
      for m in ('off','correct','wrong')}
 return result


def render_observations(model,items,step,out,device,seed):
 with unwrapped(model):
  for item in items:
   folder=out/'samples'/f'step_{step:07d}'/item['name'];folder.mkdir(parents=True,exist_ok=True)
   for mode,history in [('off',None),('correct',item_history(item,device))]:
    if mode=='off' and (out/'samples'/'base_off'/f"{item['name']}.png").exists():continue
    z,_,_=sample_frame(model,item['cond'],tuple(item['z_cur'].shape),stable_pair_seed(seed,item['name'],'sample'),device,history,50,7.5)
    rgb=decode(model,z)[0].cpu()
    if mode=='off':
     target=out/'samples'/'base_off';target.mkdir(parents=True,exist_ok=True);save_rgb(target/f"{item['name']}.png",rgb)
    else:save_rgb(folder/'correct.png',rgb)


def save_checkpoint(model,optimizer,scaler,step,epoch,out,base,args,cache_identity):
 reader=plain_unet(model).temporal_history
 payload=dict(version='temporal_static_adapter_v4',artifact_kind='temporal_static_centered_adapter',
   model_mode='static_centered',input_variant='types',reference_kind='dense',hidden_dim=reader.hidden_dim,
   fusion_dim=reader.fusion_dim,time_embed_dim=reader.time_embed_dim,base_checkpoint=base,
   state_dict={k:v.detach().cpu() for k,v in reader.state_dict().items()},optimizer=optimizer.state_dict(),
   scaler=scaler.state_dict(),step=step,epochs_completed=epoch,args=vars(args),cache_identity=cache_identity,
   adapter_state_sha256=state_dict_sha256(reader.state_dict()))
 path=out/'checkpoints'/f'centered_adapter_step_{step:07d}.pt';temp=path.with_suffix('.tmp')
 torch.save(payload,temp);temp.replace(path)
 checkpoints=sorted((out/'checkpoints').glob('centered_adapter_step_*.pt'))
 for stale in checkpoints[:-args.keep_checkpoints]:stale.unlink()
 return str(path)


def main():
 args=parse_args()
 if min(args.epochs,args.batch_size,args.save_every_epochs,args.keep_checkpoints)<1 or args.lr<=0:raise ValueError('invalid training budget')
 info=rank_info(args);device=info['device'];torch.set_num_threads(1)
 seed_training_step(args.seed,device)
 root=Path(args.data_root);cache_root=Path(args.cache_root);out=Path(args.out_dir)
 if info['is_main']:
  if out.exists() and not args.resume:raise ValueError('output exists; choose new run or explicit resume')
  (out/'checkpoints').mkdir(parents=True,exist_ok=bool(args.resume))
 if info['distributed']:torch.distributed.barrier()
 manifest=load_json(cache_root/'manifest.json');cache_done=load_json(cache_root/'cache_done.json')
 if not cache_done.get("done"):raise ValueError("condition cache incomplete")
 cache_identity=dict(fingerprint=cache_done["fingerprint"],manifest_sha256=hashlib.sha256((cache_root/"manifest.json").read_bytes()).hexdigest())
 # Data/cache contract is identity-based, never array-position-based.
 main_entries=load_json(root/'pairs.json');main_entries=main_entries.get('pairs',main_entries) if isinstance(main_entries,dict) else main_entries
 observation_entries=load_json(root/'observation_pairs.json');observation_entries=observation_entries.get('pairs',observation_entries) if isinstance(observation_entries,dict) else observation_entries
 all_entries={e['name']:e for e in main_entries+observation_entries}
 rows=manifest['items'] if isinstance(manifest,dict) else manifest
 paths={r['name']:cache_root/r['cache'] for r in rows}
 loaded={}
 for name,e in all_entries.items():
  item=torch.load(paths[name],map_location='cpu')
  if (item['previous'],item['current'],item['split'])!=(e['previous'],e['current'],e['split']):raise ValueError('cache identity mismatch')
  loaded[name]=item
 train=[loaded[e['name']] for e in main_entries if e['split']=='train']
 heldout=[loaded[e['name']] for e in main_entries if e['split']=='heldout']
 observations=[loaded[e['name']] for e in observation_entries]
 if len(train)<100 or len(heldout)<10:raise ValueError('expanded natural dataset not ready')
 settings=load_settings_args(args.settings);settings.batch_size=args.batch_size
 model,base,cfg=load_base(settings,device)
 from ldm.modules.temporal_pair_training import assert_same_base
 assert_same_base(cache_done['fingerprint']['base_checkpoint'],base)
 if cache_done['fingerprint']['settings_sha256']!=hashlib.sha256(Path(args.settings).read_bytes()).hexdigest():raise ValueError('settings differ from cached conditions')
 model.eval().requires_grad_(False)
 raw=model.DDPM.denoise_model;raw.configure_temporal_history(mode='static_centered',hidden_dim=64,input_variant='types')
 reader=raw.temporal_history;reader.requires_grad_(True)
 params=list(reader.parameters())
 optimizer=torch.optim.AdamW(params,lr=args.lr);scaler=GradScaler(enabled=device.type=='cuda')
 step=0;start_epoch=0
 if args.resume:
  payload=load_static_adapter(args.resume,reader,base)
  if payload.get('cache_identity')!=cache_identity:raise ValueError('resume data/cache differs')
  for key in ('batch_size','seed','color_probability','lr'):
   if payload['args'][key]!=vars(args)[key]:raise ValueError('resume setting differs: '+key)
  if payload['args'].get('smoke_steps'):raise ValueError('smoke checkpoint is not resumable')
  if payload['args'].get('world_size')!=info['world']:raise ValueError('resume world size differs')
  optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
  step=payload['step'];start_epoch=payload['epochs_completed']
 if info['distributed']:
  from torch.nn.parallel import DistributedDataParallel as DDP
  model.DDPM.denoise_model=DDP(raw,device_ids=[info['local']],output_device=info['local'],find_unused_parameters=False,broadcast_buffers=False)
 sampler=DistributedSampler(train,num_replicas=info['world'],rank=info['rank'],shuffle=True,seed=args.seed,drop_last=False)
 batches_per_epoch=math.ceil(len(sampler)/args.batch_size)
 args.world_size=info['world']
 fixed_versions=None
 if info['is_main']:
  with unwrapped(model):fixed_versions=frozen_model_versions(model)
  atomic_json(out/'args.json',dict(args=vars(args),base_checkpoint=base,train_pairs=len(train),heldout_pairs=len(heldout),observation_pairs=len(observations),
    steps_per_epoch=batches_per_epoch,total_steps=batches_per_epoch*args.epochs,global_batch=args.batch_size*info['world'],
    color={'probability':args.color_probability,'brightness':[.9,1.1],'contrast':[.9,1.1],'saturation':[.9,1.1],'channel_gain':[.95,1.05],'shared_pair':True},
    frozen_versions_before=fixed_versions,cache_identity=cache_identity))
  from omegaconf import OmegaConf
  OmegaConf.save(cfg,out/'cfg_resolved.yaml')
  reader.eval()
  first=eval_item(observations[0],device)
  with unwrapped(model):
   off=fixed_epsilon_monitor(model,first,args.seed+17,device,False,500)
   on=fixed_epsilon_monitor(model,first,args.seed+17,device,True,500)
  if not args.resume and off['pred_hash']!=on['pred_hash']:raise RuntimeError('zero-init changed prediction')
  atomic_json(out/'preflight.json',dict(zero_init_matches=off['pred_hash']==on['pred_hash'],off_hash=off['pred_hash']))
  if not args.resume:save_checkpoint(model,optimizer,scaler,0,0,out,base,args,cache_identity)
 if info['distributed']:torch.distributed.barrier()
 started=time.time();stop=False
 for epoch in range(start_epoch,args.epochs):
  sampler.set_epoch(epoch);indices=list(sampler);reader.train()
  for batch_index,offset in enumerate(range(0,len(indices),args.batch_size)):
   items=[train[i] for i in indices[offset:offset+args.batch_size]]
   batch_n=len(items);step+=1
   step_seed=args.seed+epoch*1000003+batch_index*1009+info['rank']*1000000007
   seed_training_step(step_seed,device)
   color=sample_pair_color(batch_n,torch.Generator().manual_seed(step_seed+17),args.color_probability)
   previous=torch.stack([x['previous_rgb'] for x in items]).to(device)
   current=torch.stack([x['current_rgb'] for x in items]).to(device)
   previous=apply_pair_color(previous,color);current=apply_pair_color(current,color)
   valid=torch.stack([x['valid'] for x in items]).to(device)
   warped=warp_augmented_rgb(previous,torch.stack([x['source_flat_index'] for x in items]).to(device),valid)
   # VAE stays frozen; every target is encoded after this step's color transform.
   z_cur=encode_latent(model,current).float()
   history=dict(latent=torch.zeros_like(z_cur),dense_rgb=warped,dense_valid=valid,
      dense_measured=torch.stack([x['measured'] for x in items]).to(device),
      dense_estimated=torch.stack([x['estimated'] for x in items]).to(device),enabled=torch.ones(batch_n,device=device,dtype=torch.bool))
   cond=recursive_batch_to_device([x['cond'] for x in items],device)
   sat_drop=torch.rand((batch_n,1,1),device=device)<float(model.satellite_condition_dropout_prob)
   cond=dict(cond);cond['context']=cond['context']*(~sat_drop)
   def closure():
    with autocast(enabled=device.type=='cuda'):
     loss,aux=epsilon_prediction_loss(model.DDPM,z_cur,cond,history,step_seed)
     return anchor_trainable_loss(loss,params),aux
   loss,aux,grad_norm,retries=optimizer_step_with_retry(closure,params,optimizer,scaler)
   with unwrapped(model):grad=named_adapter_grad_norms(model)
   if step>1 and min(grad['encoder_grad_norm'],grad['fusion_grad_norm'],grad['output_grad_norm'])<=0:raise RuntimeError('missing centered history gradients')
   rec=dict(step=step,epoch=epoch+1,batch_index=batch_index,rank=info['rank'],batch_size=batch_n,
      names=[x['name'] for x in items],seed=step_seed,loss=float(loss.detach()),grad_norm=grad_norm,
      t_mean=float(aux['t'].float().mean()),augmentation_fraction=float(color['active'].float().mean()),
      noise_hash=tensor_hash(aux['noise']),target_latent_hash=tensor_hash(z_cur),
      color_parameters={k:v.tolist() for k,v in color.items()},
      amp_retries=retries,seconds=time.time()-started,peak_gpu_gb=torch.cuda.max_memory_allocated(device)/2**30,
      **grad,**{k:float(v) for k,v in reader.last_metrics.items()})
   with (out/f'metrics_rank{info["rank"]}.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   if info['is_main'] and (step<=3 or step%10==0):print(json.dumps(rec),flush=True)
   if args.smoke_steps and step>=args.smoke_steps:stop=True;break
  if info['distributed']:torch.distributed.barrier()
  if info['is_main']:
   reader.eval()
   completed=epoch+1 if not stop else 0
   if stop or completed%args.save_every_epochs==0 or completed==args.epochs:
    # Observe original10 and heldout natural pairs without color augmentation.
    selected=observations if stop else observations+heldout
    monitor=evaluate(model,selected,step,out,device,args.seed+101,timesteps=(500,) if stop else (100,500,900))
    checkpoint=save_checkpoint(model,optimizer,scaler,step,completed,out,base,args,cache_identity)
    with unwrapped(model):versions=frozen_model_versions(model)
    if versions!=fixed_versions:raise RuntimeError('frozen base changed')
    if not stop:render_observations(model,observations,step,out,device,args.seed)
    atomic_json(out/'status.json',dict(step=step,epochs_completed=completed,checkpoint=checkpoint,monitor=monitor,
      frozen_hashes_unchanged=True,smoke=bool(args.smoke_steps)))
  if info['distributed']:torch.distributed.barrier()
  if stop:break
 if info['is_main']:atomic_json(out/'done.json',dict(done=True,smoke=bool(args.smoke_steps),step=step,epochs_completed=0 if stop else args.epochs))
 if info['distributed']:torch.distributed.destroy_process_group()

if __name__=='__main__':main()
