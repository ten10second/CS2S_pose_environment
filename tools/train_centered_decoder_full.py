#!/usr/bin/env python3
"""One matched full-data epoch for centered-history B/C; frozen conditions encoded online."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools import train_centered_decoder_probe as probe
from tools.cache_centered_conditions import load_reference, compare_rgb, tree_to_cpu, tree_hash
from tools.train_temporal_pairs import encode_conditions, encode_latent, load_base, stack_samples
from tools.train_static_history import index_dataset, load_settings_args, recursive_batch_to_device
from tools.train_centered_static_history import atomic_json, plain_unet
from ldm.modules.paired_color import sample_pair_color, apply_pair_color, warp_augmented_rgb
from ldm.modules.temporal_pair_training import seed_training_step, epsilon_prediction_loss, assert_same_base
from ldm.modules.temporal_amp import optimizer_step_with_retry
from tools.infer_temporal import tensor_hash


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=['B', 'C'], required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--settings', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--cache-root', required=True, help='Existing fixed evaluation cache only')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--color-probability', type=float, default=.7)
    p.add_argument('--keep-checkpoints', type=int, default=2)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--min-free-gb', type=float, default=2.)
    return p.parse_args()


def validate_entries(payload):
    entries = payload['items']
    if not entries:
        raise ValueError('Empty full-data manifest')
    keys = [(e['previous'], e['current']) for e in entries]
    if len(keys) != len(set(keys)) or len({e['name'] for e in entries}) != len(entries):
        raise ValueError('Duplicate training pairs or names')
    if any(e['split'] != 'train' or e.get('source_split', 'train') != 'train' for e in entries):
        raise ValueError('Training manifest must contain only train pairs')
    return entries


class FullPairDataset(Dataset):
    def __init__(self, base, entries, seed):
        self.base, self.entries, self.seed = base, entries, seed
        self.index = index_dataset(base)
        for e in entries:
            if e['current'] not in self.index or e['previous'] not in self.index:
                raise ValueError('Pair outside training manifest: ' + e['name'])
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, i):
        e = self.entries[i]
        # Stable per-pair CPU seeds, independent of worker scheduling/group.
        value = self.seed + i * 1009
        random.seed(value); np.random.seed(value % 2**32); torch.manual_seed(value)
        path = Path(e['reference_npz'])
        deadline = time.monotonic() + 3600
        while not path.is_file():
            for folder in (path.parent.parent, path.parent.parent.parent):
                failure = folder / 'failed.json'
                if failure.is_file():
                    raise RuntimeError('Reference producer failed: ' + failure.read_text())
            if time.monotonic() > deadline:
                raise TimeoutError('Waiting for reference ' + str(path))
            time.sleep(2)
        ref = load_reference(path, e)
        sample = self.base[self.index[e['current']]]
        compare_rgb('current_rgb', ref['current_rgb'], sample['grd_left_imgs'])
        return dict(e, reference=ref, sample=sample)


def identity_collate(items):
    return items


def epoch_loader(dataset, batch_size, workers, seed):
    # Manifest is already shuffled once; never wrap/pad/drop its last partial batch.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=workers, collate_fn=identity_collate,
                      generator=torch.Generator().manual_seed(seed),
                      **({'prefetch_factor': 1, 'multiprocessing_context': 'spawn'} if workers else {}))


def main():
    args = parse_args()
    if min(args.batch_size, args.keep_checkpoints) < 1 or args.num_workers < 0:
        raise ValueError('Invalid batch size, workers or retention')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    seed_training_step(args.seed, device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out/'checkpoints').mkdir()
    payload = json.loads(Path(args.manifest).read_text())
    entries = validate_entries(payload)
    selection = probe.load_selection(args.selection)
    cache_id = probe.cache_identity(Path(args.cache_root))
    groups = probe.load_cache_items(Path(args.cache_root), selection)
    # All fixed observation frames must remain outside the train set.
    observed = {x[k] for x in groups['observation'] for k in ('previous', 'current')}
    heldout = {x[k] for x in groups['heldout'] for k in ('previous', 'current')}
    train_frames = {x[k] for x in entries for k in ('previous', 'current')}
    if train_frames & (observed | heldout):
        raise ValueError('Training overlaps fixed observation/heldout frames')
    settings = load_settings_args(args.settings)
    settings.batch_size = args.batch_size
    model, base, cfg = load_base(settings, device)
    assert_same_base(cache_id['cache_done']['fingerprint']['base_checkpoint'], base)
    if cache_id['cache_done']['fingerprint']['settings_sha256'] != probe.file_sha256(args.settings):
        raise ValueError('Settings differ from fixed evaluation cache')
    model.eval().requires_grad_(False)
    param_groups, names = probe.configure_trainables(model, args.group)
    frozen_before = probe.filtered_state_hash(model, set(names))
    adapter_before = probe.adapter_state_hash(model)
    decoder_before = probe.decoder_state_hash(model)
    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler(enabled=True)
    params = [p for g in param_groups for p in g['params']]
    from utils.util import instantiate_from_config
    dataset = FullPairDataset(instantiate_from_config(cfg.data.params.train), entries, args.seed)
    loader = epoch_loader(dataset, args.batch_size, args.num_workers, args.seed)
    total = len(loader)
    if args.smoke_steps:
        total = min(total, args.smoke_steps)
    checkpoints = {total, max(1, total//2)}
    eval_steps = {total} if args.smoke_steps else {0, max(1, total//2), total}
    cache_id = dict(cache_id, full_manifest_sha256=probe.file_sha256(args.manifest))
    atomic_json(out/'args.json', dict(args=vars(args), base_checkpoint=base,
        full_pair_count=len(entries), total_steps=total, expected_last_batch=len(entries)%args.batch_size or args.batch_size,
        full_manifest_sha256=cache_id['full_manifest_sha256'], selection=selection,
        fresh_base=True, fresh_centered_adapter=True, trainable_names=names,
        trainable_parameter_count=sum(p.numel() for p in params), frozen_hash_before=frozen_before,
        initial_adapter_sha256=adapter_before, initial_decoder_sha256=decoder_before,
        optimizer_groups=[dict(name=g['name'],lr=g['lr']) for g in param_groups]))
    if 0 in eval_steps:
        summary = probe.evaluate_fixed(model, groups, 0, out, device, args.seed+101, args.group)
        atomic_json(out/'status.json', dict(step=0,total_steps=total,phase='training',summary=summary))
    plain_unet(model).temporal_history.train()
    started = time.time()
    processed = []
    for step, items in enumerate(loader, 1):
        if step > total:
            break
        if shutil.disk_usage(out).free / 2**30 < args.min_free_gb:
            raise RuntimeError('Insufficient output space')
        conditions = []
        # Encode separately to bound 576-channel LiDAR feature peak memory.
        for item in items:
            cond, rgb = encode_conditions(model, stack_samples([item['sample']]), device)
            compare_rgb('current_rgb', item['reference']['current_rgb'], rgb[0].detach().cpu())
            conditions.append(tree_to_cpu(cond))
        del cond, rgb
        step_seed = args.seed + step*1009
        seed_training_step(step_seed, device)
        color = sample_pair_color(len(items), torch.Generator().manual_seed(step_seed+17), args.color_probability)
        prev = torch.stack([x['reference']['previous_rgb'] for x in items]).to(device)
        curr = torch.stack([x['reference']['current_rgb'] for x in items]).to(device)
        prev_aug, curr_aug = apply_pair_color(prev, color), apply_pair_color(curr, color)
        valid = torch.stack([x['reference']['valid'] for x in items]).to(device)
        index = torch.stack([x['reference']['source_flat_index'] for x in items]).to(device)
        warped = warp_augmented_rgb(prev_aug, index, valid)
        z_cur = encode_latent(model, curr_aug).float()
        history = None if args.group == 'C' else dict(latent=torch.zeros_like(z_cur), dense_rgb=warped,
            dense_valid=valid, dense_measured=torch.stack([x['reference']['measured'] for x in items]).to(device),
            dense_estimated=torch.stack([x['reference']['estimated'] for x in items]).to(device),
            enabled=torch.ones(len(items),dtype=torch.bool,device=device))
        condition_hash = tree_hash(conditions)
        cond = recursive_batch_to_device(conditions, device)
        sat_drop = torch.rand((len(items),1,1),device=device) < float(model.satellite_condition_dropout_prob)
        cond['context'] = cond['context'] * (~sat_drop)
        def closure():
            with autocast(enabled=True):
                return epsilon_prediction_loss(model.DDPM, z_cur, cond, history, step_seed)
        loss, aux, norm, retries = optimizer_step_with_retry(closure, params, optimizer, scaler)
        grad = probe.trainable_grad_summary(model,names)
        if grad['missing_grad'] or grad['decoder_grad_norm'] <= 0 or (args.group=='B' and grad['adapter_grad_norm']<=0):
            raise RuntimeError('Invalid gradients: ' + str(grad))
        batch_names = [x['name'] for x in items]
        processed.extend(batch_names)
        rec = dict(step=step,group=args.group,loss=float(loss.detach().cpu()), batch_names=batch_names,
            batch_size=len(items), pairs_seen=len(processed), epoch=len(processed)/len(entries),
            grad_norm=norm,amp_retries=retries,adapter_grad_norm=grad['adapter_grad_norm'],decoder_grad_norm=grad['decoder_grad_norm'],
            noise_hash=tensor_hash(aux['noise']),target_latent_hash=tensor_hash(z_cur),condition_hash=condition_hash,
            color_hash=hashlib.sha256(json.dumps({k:v.tolist() for k,v in color.items()},sort_keys=True).encode()).hexdigest(),
            t_mean=float(aux['t'].float().mean().cpu()),satellite_dropout_fraction=float(sat_drop.float().mean().cpu()),
            seconds=round(time.time()-started,2),peak_gpu_reserved_gb=round(torch.cuda.max_memory_reserved(device)/2**30,3))
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(rec,sort_keys=True)+'\n')
        atomic_json(out/'status.json',dict(step=step,total_steps=total,phase='training',**{k:rec[k] for k in ('loss','pairs_seen','epoch','seconds')}))
        if step<=3 or step%25==0:
            print(json.dumps(rec,sort_keys=True),flush=True)
        if step in checkpoints:
            probe.save_probe_checkpoint(model,step,out,base,args,names,cache_id,frozen_before,adapter_before,decoder_before)
            if probe.filtered_state_hash(model,set(names)) != frozen_before:
                raise RuntimeError('Frozen weights changed')
        if step in eval_steps:
            plain_unet(model).temporal_history.eval()
            atomic_json(out/'status.json',dict(step=step,total_steps=total,phase='evaluation'))
            summary = probe.evaluate_fixed(model,groups,step,out,device,args.seed+101,args.group,
                                            (500,) if args.smoke_steps else probe.EVAL_TIMESTEPS)
            if step == total and not args.smoke_steps:
                atomic_json(out/'status.json',dict(step=step,total_steps=total,phase='generation',summary=summary))
                probe.render_samples(model,groups,step,out,device,args.seed+202,args.group,True,50,7.5)
            plain_unet(model).temporal_history.train()
        if args.smoke_steps and step==total:
            break
    if not args.smoke_steps and processed != [x['name'] for x in entries]:
        raise RuntimeError('Epoch coverage mismatch: dropped/repeated/reordered pairs')
    frozen_after = probe.filtered_state_hash(model,set(names))
    if frozen_after != frozen_before:
        raise RuntimeError('Frozen weights changed')
    atomic_json(out/'done.json',dict(done=True,group=args.group,steps=total,pairs_seen=len(processed),
        full_pair_count=len(entries),exactly_one_epoch=not bool(args.smoke_steps),
        epoch_order_sha256=hashlib.sha256(json.dumps(processed).encode()).hexdigest(),
        frozen_hash_unchanged=True,frozen_hash_after=frozen_after))
    atomic_json(out/'status.json',dict(step=total,total_steps=total,phase='done',summary=summary))

if __name__=='__main__':
    main()
