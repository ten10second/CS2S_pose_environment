"""Frozen Stage-D probes: reproduce first, then evaluate prespecified held-out pairs.

No optimizer, training, sampling rollout, or model-source modification. A scoped
Python profiler observes existing loss/attention tensors without changing forward.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data._utils.collate import default_collate

TOOLS = Path(__file__).resolve().parent
sys.path[:0] = [str(TOOLS), str(TOOLS.parent)]
from train_kitti_geometry_history import (
    GeometryPairs, GeometryTrainingStep, encode_history, fixed_probe,
    make_dataset, move_batch_to_device, pinned_denoising, split_drive_pairs,
)
from temporal_history import enable_history_attention, parse_history_block_indices
from generate_kitti_geometry_history import load_geometry_history_checkpoint
from generate_kitti_raea_samples import load_checkpoint_into_model
from ldm.modules.temporal_history_attention import GeometryHistoryAttention


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def stable_seed(seed, key):
    return (seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % (2**31)


def select_spaced_pairs(rows, pairs, held, excluded_endpoints, per_drive=32, min_gap=10):
    """Outcome-independent, frame-spaced selection, capped without replacement."""
    selected = []
    for drive in sorted(held):
        candidates = sorted((p for p in pairs if rows[p[1]]['drive'] == drive
                             and not set(p[:2]).intersection(excluded_endpoints)),
                            key=lambda p: rows[p[1]]['frame_index'])
        spaced = []
        for pair in candidates:
            if not spaced or rows[pair[1]]['frame_index'] - rows[spaced[-1][1]]['frame_index'] >= min_gap:
                spaced.append(pair)
        if len(spaced) > per_drive:
            spaced = [spaced[i] for i in np.linspace(0, len(spaced) - 1, per_drive, dtype=int)]
        selected.extend(spaced)
    return selected


def make_selection(rows, train, val, metadata, per_drive=32, min_gap=10):
    originals, expanded = [], []
    seed, world = metadata['args']['seed'], metadata['world_size']
    for rank in range(world):
        vi, ti = len(val) * rank // world, len(train) * rank // world
        originals.append(dict(cohort='original', split='val', rank=rank, pair=list(val[vi]),
                              wrong_pair=list(train[ti]), latent_seed=seed + vi,
                              wrong_latent_seed=seed + ti, probe_seed=seed + 1))
    excluded = {i for item in originals for i in item['pair'][:2]}
    chosen = select_spaced_pairs(rows, val, metadata['val_drives'], excluded, per_drive, min_gap)
    for index, pair in enumerate(chosen):
        key = rows[pair[1]]['sample_id']
        wrong = train[stable_seed(seed, 'wrong:' + key) % len(train)]
        expanded.append(dict(cohort='expanded', split='val', rank=index % world,
                             pair=list(pair), wrong_pair=list(wrong),
                             latent_seed=stable_seed(seed, 'latent:' + key),
                             wrong_latent_seed=stable_seed(seed, 'wrong_latent:' + key),
                             probe_seed=stable_seed(seed, 'probe:' + key)))
    for item in originals + expanded:
        row, wrong = rows[item['pair'][1]], rows[item['wrong_pair'][1]]
        item.update(pair_id=row['sample_id'], drive=row['drive'], frame_index=row['frame_index'],
                    wrong_pair_id=wrong['sample_id'], wrong_drive=wrong['drive'])
        if row['drive'] == wrong['drive']:
            raise ValueError('wrong history must come from another drive')
    return dict(original=originals, expanded=expanded, per_drive_target=per_drive,
                minimum_current_frame_gap=min_gap, expanded_counts=dict(Counter(x['drive'] for x in expanded)),
                selection_policy='Greedy frame spacing then uniform index subsampling; exclude original endpoints; no outcome filtering',
                scoring_mask='Original correct-geometry history_valid at RGB latent resolution, identical across conditions',
                sampling_limit='Short held-out drives are capped by available spaced pairs; frames remain temporally correlated')


def region_metrics(raw, valid):
    """Use the same physical scoring mask for every history condition."""
    raw, valid = raw.detach().cpu().double(), valid.detach().cpu().bool()
    if raw.ndim != 4 or tuple(raw.shape[0:1] + raw.shape[2:]) != tuple(valid.shape):
        raise ValueError('loss/scoring-mask shape mismatch; do not silently resize scoring mask')
    if not torch.isfinite(raw).all():
        raise FloatingPointError('nonfinite RGB squared error')
    pixel = raw.mean(dim=1)
    count, total = int(valid.sum()), valid.numel()
    return dict(all=float(pixel.mean()), valid=float(pixel[valid].mean()) if count else None,
                invalid=float(pixel[~valid].mean()) if count < total else None,
                valid_fraction=count / total, valid_cells=count, total_cells=total)


def attention_metrics(local, output, block_index):
    x, cond = local['x'].detach().float(), local['cond_summary'].detach().float()
    out = output.detach().float()
    supported = local.get('local_valid')
    if supported is None:
        mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        null = torch.ones(x.shape[:2], device=x.device)
    else:
        mask = supported.any(dim=-1)
        null = local['attn'][..., -1].detach().float().mean(dim=1)
    norm_out, norm_x, norm_cond = (a.norm(dim=-1) for a in (out, x, cond))
    def ratio(a, b, region=None):
        if region is not None:
            if not region.any():
                return None
            a, b = a[region], b[region]
        return float(a.mean()) / max(float(b.mean()), 1e-6)
    return dict(block_index=block_index, query_hw=list(local.get('query_hw') or ()),
                query_valid_fraction=float(mask.float().mean()), null_all=float(null.mean()),
                null_valid=float(null[mask].mean()) if mask.any() else None,
                residual_to_x_all=ratio(norm_out, norm_x),
                residual_to_x_valid=ratio(norm_out, norm_x, mask),
                residual_to_cond_all=ratio(norm_out, norm_cond),
                residual_norm_all=float(norm_out.mean()), x_norm_all=float(norm_x.mean()))


@contextmanager
def capture_existing_tensors(ddpm, blocks, valid):
    """Observe function return locals; never replace model outputs or request x0."""
    if sys.getprofile() is not None:
        raise RuntimeError('refusing to replace an existing Python profiler')
    ddpm_code = ddpm.p_losses.__func__.__code__
    attention_code = GeometryHistoryAttention.forward.__code__
    block_ids = {id(b.history_attn): b.history_block_index for b in blocks}
    captured, attention = [], []
    def observe(frame, event, result):
        if event != 'return' or result is None:
            return
        if frame.f_code is attention_code and id(frame.f_locals.get('self')) in block_ids:
            attention.append(attention_metrics(frame.f_locals, result, block_ids[id(frame.f_locals['self'])]))
        elif frame.f_code is ddpm_code and frame.f_locals.get('self') is ddpm:
            local = frame.f_locals
            if local.get('loss_mask') is not None and local.get('loss_mask_weight', 0) > 0:
                raise ValueError('weighted loss_eps_base requires a different decomposition')
            if len(attention) != len(blocks):
                raise RuntimeError(f'expected {len(blocks)} attention captures, got {len(attention)}')
            captured.append(dict(region_eps=region_metrics(local['loss_raw'], valid), attention=list(attention)))
            attention.clear()
    sys.setprofile(observe)
    try:
        yield captured
    finally:
        sys.setprofile(None)


def repeat_disabled(module, batch, latent, geom, timestep, seed, amp):
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.no_grad(), autocast(enabled=amp), pinned_denoising(module.model.DDPM, timestep, seed + 1):
            loss = module(batch, latent, geom, False, False)
    # fixed_probe uses the OUTER model loss, which also includes model-level
    # auxiliaries; DDPM.last_loss_metrics['loss_total'] is not that same scalar.
    return {**module.model.DDPM.last_loss_metrics, 'loss_total': float(loss)}


def evaluate_item(module, blocks, dataset, rows, args, spec, reproduce=False):
    device = next(module.model.parameters()).device
    def load(pair):
        return move_batch_to_device(default_collate([GeometryPairs(dataset, rows, [pair], args.kitti_root)[0]]), device)
    item, wrong_item = load(spec['pair']), load(spec['wrong_pair'])
    def encode(item, seed):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            return encode_history(module.model, item['prev'])
    latent = encode(item, spec['latent_seed'])
    wrong = encode(wrong_item, spec['wrong_latent_seed'])
    del wrong_item
    geom = {k: item[k] for k in ('history_grid', 'history_valid')}
    inputs = (module, item['cur'], latent, geom, wrong, [250, 750], spec['probe_seed'], not args.no_amp)
    plain = fixed_probe(*inputs) if reproduce else None
    with capture_existing_tensors(module.model.DDPM, blocks, geom['history_valid']) as captures:
        results = fixed_probe(*inputs)
    if len(captures) != len(results):
        raise RuntimeError('missing diffusion tensor capture')
    if plain is not None and plain != results:
        raise RuntimeError('instrumentation changed fixed_probe scalar outputs')
    repeats = {t: repeat_disabled(module, item['cur'], latent, geom, t, spec['probe_seed'], not args.no_amp)
               for t in (250, 750)}
    latent_hash = hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()
    wrong_hash = hashlib.sha256(wrong.detach().cpu().numpy().tobytes()).hexdigest()
    for result, extra in zip(results, captures):
        if abs(extra['region_eps']['all'] - result['loss_eps_base']) > 1e-7:
            raise RuntimeError('regional RGB mean does not reproduce loss_eps_base')
        if result['condition'] == 'disabled':
            result['disabled_repeat_equal'] = all(float(repeats[result['t']][k]) == result[k]
                                                  for k in ('loss_eps_base', 'loss_total', 'loss_lidar_bottleneck_depth_log_l1'))
            if not result['disabled_repeat_equal']:
                raise RuntimeError('disabled repeat is not bitwise stable')
        result.update(extra)
        result.update({k: spec[k] for k in ('cohort', 'split', 'pair_id', 'drive', 'frame_index', 'rank',
                                          'wrong_pair_id', 'wrong_drive', 'latent_seed', 'wrong_latent_seed', 'probe_seed')})
        result.update(history_latent_sha256=latent_hash, wrong_latent_sha256=wrong_hash)
    for t in (250, 750):
        if len({r['loss_lidar_bottleneck_depth_log_l1'] for r in results if r['t'] == t}) != 1:
            raise RuntimeError('bottleneck depth changed across history conditions')
    return results


def check_original(records, stage_dir, rank, step):
    prior = [json.loads(line) for line in (stage_dir / f'probes_rank{rank}.jsonl').read_text().splitlines()]
    prior = {(r['pair_id'], r['t'], r['condition']): r for r in prior if r['step'] == step and r['split'] == 'val'}
    mismatches = []
    for record in records:
        key = record['pair_id'], record['t'], record['condition']
        expected = prior[key]
        for metric, value in expected.items():
            if metric.startswith('loss_') and record[metric] != value:
                mismatches.append(dict(key=key, metric=metric, expected=value, actual=record[metric]))
    return dict(passed=not mismatches and len(records) == 8, comparisons=len(records), mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--phase', choices=('prepare', 'reproduce', 'expanded'), required=True)
    args_cli = parser.parse_args()
    stage, out = args_cli.stage_dir.resolve(), args_cli.out_dir.resolve()
    metadata = json.loads((stage / 'run.json').read_text())
    args = SimpleNamespace(**metadata['args'])
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if sha_file(args.manifest) != metadata['manifest_sha256']:
        raise ValueError('training manifest changed')
    train, val, held = split_drive_pairs(rows, metadata['val_drives'])
    checkpoint = stage / 'geometry_history_step_1000.pt'
    if args_cli.phase == 'prepare':
        out.mkdir(parents=True, exist_ok=False)
        selection = make_selection(rows, train, val, metadata)
        selection.update(stage_dir=str(stage), checkpoint=str(checkpoint), checkpoint_sha256=sha_file(checkpoint),
                         base_ckpt=str(args.ckpt), base_ckpt_sha256=sha_file(args.ckpt), config_sha256=sha_file(args.config),
                         evaluator_sha256=sha_file(__file__),
                         manifest_sha256=metadata['manifest_sha256'], world_size=metadata['world_size'],
                         timesteps=[250, 750], conditions=['disabled', 'correct', 'wrong_geometry', 'wrong_history'])
        write_json_new(out / 'selection.json', selection)
        print(json.dumps({'event': 'selection_fixed', 'counts': selection['expanded_counts'], 'total': len(selection['expanded'])}), flush=True)
        return
    selection = json.loads((out / 'selection.json').read_text())
    if selection['checkpoint_sha256'] != sha_file(checkpoint) or selection['config_sha256'] != sha_file(args.config):
        raise ValueError('checkpoint/config changed after selection')
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')))
    if world != metadata['world_size']:
        raise ValueError('use original world_size for rank probe replay')
    if args_cli.phase == 'expanded':
        for r in range(world):
            if json.loads((out / f'reproduction_rank{r}.json').read_text()).get('passed') is not True:
                raise RuntimeError('original probes must reproduce before expanded evaluation')
    result_path = out / (f'original_rank{rank}.jsonl' if args_cli.phase == 'reproduce' else f'evaluation_rank{rank}.jsonl')
    if result_path.exists():
        raise FileExistsError(result_path)
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    from omegaconf import OmegaConf
    from utils.util import instantiate_from_config
    cfg = OmegaConf.load(args.config)
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    dataset = make_dataset(args, cfg)
    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.ckpt)
    hub, encoder, blocks = enable_history_attention(model, geometry=True,
        block_indices=parse_history_block_indices(args.block_indices), history_dim=args.history_dim, heads=4, dim_head=32)
    model.cuda().eval()
    encoder.cuda().eval()
    load_args = SimpleNamespace(ckpt=args.ckpt, block_indices=tuple(b.history_block_index for b in blocks), history_dim=args.history_dim)
    payload = load_geometry_history_checkpoint(checkpoint, load_args, encoder, blocks)
    if payload['step'] != 1000:
        raise ValueError('expected step 1000')
    module = GeometryTrainingStep(model, encoder, hub)
    for param in module.parameters():
        param.requires_grad_(False)
    versions = [(p, p._version) for p in module.parameters()]
    cohort = 'original' if args_cli.phase == 'reproduce' else 'expanded'
    selected = [s for s in selection[cohort] if s['rank'] == rank]
    with result_path.open('x') as handle:
        for index, spec in enumerate(selected):
            records = evaluate_item(module, blocks, dataset, rows, args, spec, reproduce=cohort == 'original')
            for record in records:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
            if cohort == 'original':
                verdict = check_original(records, stage, rank, 1000)
                write_json_new(out / f'reproduction_rank{rank}.json', verdict)
                if not verdict['passed']:
                    raise RuntimeError(f'original probe mismatch: {verdict}')
            print(json.dumps(dict(event='pair_complete', rank=rank, index=index + 1, total=len(selected), pair_id=spec['pair_id'])), flush=True)
    if any(p._version != version for p, version in versions):
        raise RuntimeError('model parameters changed during evaluation')
    write_json_new(out / f'{cohort}_complete_rank{rank}.json', dict(passed=True, pairs=len(selected), parameters_unchanged=True))


if __name__ == '__main__':
    main()
