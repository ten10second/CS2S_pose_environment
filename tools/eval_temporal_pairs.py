#!/usr/bin/env python3
"""Matched 50-step generation on fixed GT-history pairs, with image artifacts."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.temporal_data import load_settings_args, load_base, evaluation_device, encode_conditions, stack_samples
from tools.cache_centered_conditions import load_reference, reference_path, build_warped_rgb, compare_rgb, previous_rgb_sample
from tools.infer_temporal import sample_frame, decode, tree_to
from ldm.modules.temporal_pair_training import make_history_condition, load_history_checkpoint
from ldm.modules.temporal_image_loss import _sobel_xy


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings', required=True)
    p.add_argument('--eval-selection', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:5')
    p.add_argument('--seed', type=int, default=4307)
    p.add_argument('--ddim-steps', type=int, default=50)
    p.add_argument('--guidance', type=float, default=7.5)
    args = p.parse_args(argv)
    if args.ddim_steps < 1:
        p.error('ddim steps must be positive')
    return args


def select_entries(payload):
    entries = []
    seen = set()
    for group in ('observation', 'heldout'):
        for source in payload[group]:
            entry = dict(source, eval_group=group)
            pair = (entry['previous'], entry['current'])
            if pair in seen:
                raise ValueError('duplicate evaluation pair')
            if entry['source_split'] not in ('train', 'heldout'):
                raise ValueError('unknown original split')
            seen.add(pair)
            entries.append(entry)
    if not entries:
        raise ValueError('empty evaluation selection')
    return entries


def image_metrics(rgb, target):
    rgb, target = rgb.float().cpu(), target.float().cpu()
    if rgb.shape != target.shape or not torch.isfinite(rgb).all():
        raise ValueError('invalid generated RGB')
    edge = (_sobel_xy(rgb) - _sobel_xy(target)).abs()
    b, _, h, w = edge.shape
    return {'rgb_mae': float((rgb - target).abs().mean()),
            'rgb_mse': float((rgb - target).square().mean()),
            'sobel_mae': float(edge.view(b, 3, 2, h, w).mean((1, 3, 4)).sum(1).mean())}


def to_pil(rgb):
    return Image.fromarray((rgb[0].float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype('uint8'))


def save_sheet(path, panels):
    width, height = panels[0][1].shape[-1], panels[0][1].shape[-2]
    canvas = Image.new('RGB', (width * 2, (height + 24) * ((len(panels) + 1) // 2)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (label, rgb) in enumerate(panels):
        x, y = (i % 2) * width, (i // 2) * (height + 24)
        draw.text((x + 4, y + 4), label, fill='black')
        canvas.paste(to_pil(rgb), (x, y + 24))
    canvas.save(path)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise ValueError('refusing overwrite: ' + str(out))
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    device = evaluation_device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    selection = Path(args.eval_selection)
    entries = select_entries(json.loads(selection.read_text()))
    model, base, cfg = load_base(load_settings_args(args.settings), device)
    model.eval().requires_grad_(False)
    from utils.util import instantiate_from_config
    datasets = {'train': instantiate_from_config(cfg.data.params.train),
                'heldout': instantiate_from_config(cfg.data.params.test)}
    indexes = {k: {r['sample_id']: i for i, r in enumerate(ds.records)} for k, ds in datasets.items()}
    out.mkdir(parents=True)
    (out / 'args.json').write_text(json.dumps(dict(args=vars(args), base=base, pairs=entries,
        zero_rgb_definition='Black RGB encoded through the same VAE; all masks retained',
        metric_scope='Whole image; not building-specific identity or sharpness metrics'), indent=2))
    prepared = []
    # Sample the untouched base before loading temporal weights into this same UNet.
    for i, entry in enumerate(entries):
        dataset = datasets[entry['source_split']]
        index = indexes[entry['source_split']]
        ref = load_reference(reference_path(selection.parent, entry), entry)
        sample = dataset[index[entry['current']]]
        prev = previous_rgb_sample(dataset, dataset.records[index[entry['previous']]])['grd_left_imgs']
        compare_rgb('previous', ref['previous_rgb'], prev)
        compare_rgb('current', ref['current_rgb'], sample['grd_left_imgs'])
        cond, _ = encode_conditions(model, stack_samples([sample]), device)
        warp = build_warped_rgb(ref['previous_rgb'], ref['source_flat_index'], ref['valid']).unsqueeze(0)
        masks = [ref[k].unsqueeze(0).to(device) for k in ('valid', 'measured', 'estimated')]
        history = make_history_condition(model, warp.to(device), *masks)
        zero = make_history_condition(model, torch.zeros_like(warp).to(device), *masks)
        seed = args.seed + i
        z, _, noise_hash = sample_frame(model, cond, history['latent'].shape, seed, device,
                                       None, args.ddim_steps, args.guidance)
        rgb = decode(model, z).cpu()
        prepared.append(dict(entry=entry, seed=seed, cond=tree_to(cond, 'cpu'),
            history=tree_to(history, 'cpu'), zero=tree_to(zero, 'cpu'), noise_hash=noise_hash,
            previous=ref['previous_rgb'].unsqueeze(0), target=ref['current_rgb'].unsqueeze(0),
            warp=warp, outputs={'base': rgb}))
        print(json.dumps({'phase': 'base', 'pair': i + 1, 'total': len(entries)}), flush=True)
    info = load_history_checkpoint(args.checkpoint, model, base)
    model.eval().requires_grad_(False)
    records = []
    for i, item in enumerate(prepared):
        for mode, history in (('correct', item['history']), ('off', None), ('zero_rgb', item['zero'])):
            z, _, noise_hash = sample_frame(model, item['cond'], item['history']['latent'].shape,
                item['seed'], device, history, args.ddim_steps, args.guidance)
            if noise_hash != item['noise_hash']:
                raise RuntimeError('sampling noise differs across controls')
            item['outputs'][mode] = decode(model, z).cpu()
        pair_dir = out / ('%02d_%s' % (i, item['entry']['eval_group']))
        pair_dir.mkdir()
        panels = [('Previous GT', item['previous']), ('Warp reference', item['warp']),
                  ('Current GT', item['target'])] + list(item['outputs'].items())
        save_sheet(pair_dir / 'comparison.png', panels)
        for label, rgb in panels:
            to_pil(rgb).save(pair_dir / (label.replace(' ', '_') + '.png'))
        row = dict(entry=item['entry'], seed=item['seed'], noise_hash=item['noise_hash'],
                   metrics={k: image_metrics(v, item['target']) for k, v in item['outputs'].items()},
                   correct_off_mae=float((item['outputs']['correct'] - item['outputs']['off']).abs().mean()),
                   correct_zero_rgb_mae=float((item['outputs']['correct'] - item['outputs']['zero_rgb']).abs().mean()))
        records.append(row)
        (pair_dir / 'metrics.json').write_text(json.dumps(row, indent=2))
        print(json.dumps({'phase': 'temporal', 'pair': i + 1, 'total': len(entries)}), flush=True)
    groups = {}
    for group in ('observation', 'heldout'):
        subset = [r for r in records if r['entry']['eval_group'] == group]
        if not subset:
            continue
        groups[group] = {mode: {key: sum(r['metrics'][mode][key] for r in subset) / len(subset)
            for key in ('rgb_mae', 'rgb_mse', 'sobel_mae')} for mode in ('base', 'correct', 'off', 'zero_rgb')}
    (out / 'summary.json').write_text(json.dumps(dict(checkpoint=info, groups=groups, pairs=records), indent=2))
    (out / 'done.json').write_text(json.dumps(dict(done=True, pairs=len(records), ddim_steps=args.ddim_steps), indent=2))


if __name__ == '__main__':
    main()
