import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from ldm.modules.temporal_pair_training import make_history_condition,save_history_checkpoint,load_history_checkpoint
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel
from tools.train_temporal_pairs import ReferencePairs,parse_args,concat_conditions

class VAE(torch.nn.Module):
    def encode(self,x):
        z=F.avg_pool2d(torch.cat([x,x[:,:1]],1),8)
        return SimpleNamespace(mode=lambda:z)

class MinimalPipelineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_fractional_masks_encode_rgb_without_resizing_rgb_first(self):
        model=SimpleNamespace(pre_AE_model=VAE(),scale_factor=.5)
        rgb=torch.ones(1,3,8,8,requires_grad=True)
        valid=torch.ones(1,1,8,8,dtype=torch.bool)
        measured=valid.clone();measured[:,:,:,4:]=False
        history=make_history_condition(model,rgb,valid,measured,valid&~measured)
        torch.testing.assert_close(history['masks'],torch.tensor([1,.5,.5]).reshape(1,3,1,1))
        torch.testing.assert_close(history['latent'],torch.full((1,4,1,1),.5))
        self.assertFalse(history['latent'].requires_grad)
        other=make_history_condition(model,torch.zeros_like(rgb),valid,measured,valid&~measured)
        self.assertFalse(torch.equal(other['latent'],history['latent']))
        self.assertTrue(torch.equal(other['masks'],history['masks']))
    def test_empty_reference_is_neutral_and_partition_errors_rejected(self):
        model=SimpleNamespace(pre_AE_model=VAE(),scale_factor=.5)
        rgb=torch.rand(1,3,8,8);empty=torch.zeros(1,1,8,8,dtype=torch.bool)
        h=make_history_condition(model,rgb,empty,empty,empty)
        self.assertEqual(float(h['latent'].abs().sum()),0)
        with self.assertRaisesRegex(ValueError,'partition'):
            make_history_condition(model,rgb,~empty,~empty,~empty)
    def test_new_checkpoint_roundtrip_and_old_rejection(self):
        def new():return UNetModel(image_size=8,in_channels=4,model_channels=32,out_channels=4,num_res_blocks=1,attention_resolutions=(),channel_mult=(1,2),num_heads=1)
        model=new();model.configure_history_input()
        with torch.no_grad():model.input_blocks[0][0].weight[:,4:].normal_()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weights.pt';base={'sha256':'same'}
            save_history_checkpoint(p,model,base,2,{'rgb_weight':.1})
            target=new();info=load_history_checkpoint(p,target,base)
            self.assertEqual(info['step'],2)
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,target.state_dict()[k]),k)
            with self.assertRaisesRegex(ValueError,'identity'):
                load_history_checkpoint(p,new(),{'sha256':'different'})
            torch.save({'version':'temporal_static_adapter_v2'},p)
            target=new()
            with self.assertRaisesRegex(ValueError,'incompatible'):
                load_history_checkpoint(p,target,base)
            self.assertEqual(target.input_blocks[0][0].in_channels,4)
    def test_pair_manifest_rejects_heldout_crossdrive_and_duplicates(self):
        a='date/drive/0001';b='date/drive/0002'
        base=SimpleNamespace(records=[{'sample_id':a},{'sample_id':b}])
        pair={'name':'p','previous':a,'current':b,'split':'train'}
        self.assertEqual(len(ReferencePairs(base,[pair],'.')),1)
        for rows in ([dict(pair,split='heldout')],[pair,pair],[dict(pair,current='date/other/0002')]):
            with self.assertRaises(ValueError):ReferencePairs(base,rows,'.')
    def test_reference_wait_is_forwarded_without_dropping_missing_pairs(self):
        a='date/drive/0001';b='date/drive/0002'
        base=SimpleNamespace(records=[{'sample_id':a},{'sample_id':b}])
        pair={'name':'p','previous':a,'current':b,'split':'train'}
        ds=ReferencePairs(base,[pair],'.',123)
        with patch('tools.train_temporal_pairs.load_reference',side_effect=FileNotFoundError('pending')) as load:
            with self.assertRaises(FileNotFoundError):ds[0]
            self.assertEqual(load.call_args[0][2],123)
        self.assertEqual(len(ds),1)
    def test_nested_lidar_batch_preserves_every_sample_at_each_scale(self):
        def cond(v):
            return {'context':torch.full((1,3,4),v), 'lidar_context':
                    {'features':[torch.full((1,2,4,8),v),torch.full((1,4,2,4),v+1)],
                     'masks':[torch.ones(1,1,4,8)*v,torch.ones(1,1,2,4)*v]},'optional':None}
        batch=concat_conditions([cond(2.),cond(7.)])
        self.assertEqual(batch['context'][:,0,0].tolist(),[2.,7.])
        self.assertEqual(batch['lidar_context']['features'][0][:,0,0,0].tolist(),[2.,7.])
        self.assertEqual(batch['lidar_context']['features'][1][:,0,0,0].tolist(),[3.,8.])
        self.assertEqual(batch['lidar_context']['masks'][0][:,0,0,0].tolist(),[2.,7.])
        self.assertIsNone(batch['optional'])
    def test_training_has_no_legacy_modes_and_requires_explicit_loss_weights(self):
        args=parse_args(['--settings','s','--manifest','m','--out-dir','o','--rgb-weight','.1','--edge-weight','.01'])
        self.assertFalse(hasattr(args,'temporal_mode'))
        self.assertEqual(args.keep_checkpoints,2)

if __name__=='__main__':unittest.main()
