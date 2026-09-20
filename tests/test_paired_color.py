import unittest
import torch
from ldm.modules.paired_color import sample_pair_color,apply_pair_color,warp_augmented_rgb
class PairedColorTests(unittest.TestCase):
 def test_shared_transform_identity_and_reproducibility(self):
  rgb=torch.rand(4,3,8,16)
  p=sample_pair_color(4,torch.Generator().manual_seed(11),1.)
  q=sample_pair_color(4,torch.Generator().manual_seed(11),1.)
  self.assertTrue(all(torch.equal(p[k],q[k]) for k in p))
  self.assertTrue(torch.equal(apply_pair_color(rgb,p),apply_pair_color(rgb.clone(),p)))
  self.assertFalse(torch.equal(apply_pair_color(rgb,p),rgb))
  identity=sample_pair_color(4,torch.Generator().manual_seed(11),0.)
  self.assertTrue(torch.equal(apply_pair_color(rgb,identity),rgb))
 def test_color_before_gather_and_black_holes(self):
  rgb=torch.rand(2,3,8,16);index=torch.arange(128).reshape(1,8,16).expand(2,-1,-1).flip(-1).clone()
  valid=torch.ones(2,1,8,16,dtype=torch.bool);valid[:,:,:2,:3]=False;index[~valid[:,0]]=-1
  params=sample_pair_color(2,torch.Generator().manual_seed(2),1.)
  augmented=apply_pair_color(rgb,params)
  actual=warp_augmented_rgb(augmented,index,valid)
  expected=augmented.flip(-1)*valid
  self.assertTrue(torch.equal(actual,expected));self.assertEqual(float(actual[~valid.expand_as(actual)].sum()),0.)
 def test_same_point_colors_transform_identically_across_frames(self):
  previous=torch.rand(2,3,8,16);current=torch.rand_like(previous)
  current[:,:,3,5]=previous[:,:,1,2]
  params=sample_pair_color(2,torch.Generator().manual_seed(7),1.)
  a=apply_pair_color(previous,params);b=apply_pair_color(current,params)
  self.assertTrue(torch.equal(a[:,:,1,2],b[:,:,3,5]))
 def test_reject_invalid_correspondence(self):
  with self.assertRaises(ValueError):warp_augmented_rgb(torch.ones(1,3,2,2),torch.full((1,2,2),9),torch.ones(1,1,2,2,dtype=torch.bool))
if __name__=='__main__':unittest.main()
