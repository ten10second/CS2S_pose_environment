"""A1 integration: checkpoint boundaries and per-call sampling caches."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from ldm.modules.static_history import AdaptiveStaticHistoryAdapter, DenseStaticHistoryAdapter
from tools.infer_temporal import load_static_adapter
from tools.train_dense_static_history import save_dense_checkpoint
from test_persistent_sampling import Sampler


def history():
    valid = torch.ones(2, 1, 32, 64, dtype=torch.bool)
    return dict(latent=torch.randn(2, 4, 4, 8), dense_rgb=torch.rand(2, 3, 32, 64),
                dense_valid=valid, dense_measured=valid.clone(), dense_estimated=~valid,
                enabled=torch.tensor([True, False]))


class AdaptiveProbeTests(unittest.TestCase):
    def test_v3_roundtrip_and_reject_old_architecture(self):
        a = AdaptiveStaticHistoryAdapter(8, 16, hidden_dim=4, input_variant='rgb')
        model = SimpleNamespace(DDPM=SimpleNamespace(denoise_model=SimpleNamespace(temporal_history=a)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'adaptive.pt'
            save_dense_checkpoint(path, model, None, 2, {'sha256':'base'}, {'reference_kind':'dense'})
            b = AdaptiveStaticHistoryAdapter(8, 16, hidden_dim=4, input_variant='rgb')
            data = load_static_adapter(path,b,{'sha256':'base'})
            self.assertEqual(data['version'],'temporal_static_adapter_v3')
            for k,v in a.state_dict().items(): self.assertTrue(torch.equal(v,b.state_dict()[k]))
            with self.assertRaisesRegex(ValueError,'static_adaptive'):
                load_static_adapter(path,DenseStaticHistoryAdapter(8,hidden_dim=4,input_variant='rgb'),{'sha256':'base'})
            with self.assertRaisesRegex(ValueError,'fusion/time width'):
                load_static_adapter(path,AdaptiveStaticHistoryAdapter(8,32,hidden_dim=4,input_variant='rgb'),{'sha256':'base'})

    def test_sampler_caches_once_rebuilds_for_rgb_change_and_preserves_cfg_order(self):
        sampler = Sampler()
        adapter = AdaptiveStaticHistoryAdapter(8,16,hidden_dim=4).eval()
        sampler.model.denoise_model.temporal_history = adapter
        h = history(); noise=torch.randn(2,4,4,8)
        seen=[]
        hook=adapter.encoder.register_forward_hook(lambda m,a,o:seen.append(o.clone()))
        kwargs=dict(S=8,batch_size=2,shape=[4,4,8],x_T=noise,conditioning=torch.ones(2,16,12),unconditional_guidance_scale=7.5)
        sampler.sample(history=h,**kwargs)
        self.assertEqual(len(seen),1)
        self.assertNotIn('dense_features',h)
        for _,_,read in sampler.model.denoise_model.calls:
            self.assertTrue(torch.equal(read['dense_features'],torch.cat([seen[0]]*2)))
            self.assertEqual(read['enabled'].tolist(),[True,False,True,False])
        changed=dict(h,dense_rgb=h['dense_rgb'][:,[2,1,0]].clone(),dense_features=seen[0])
        sampler.model.denoise_model.calls.clear()
        sampler.sample(history=changed,**kwargs)
        self.assertEqual(len(seen),2)
        self.assertFalse(torch.equal(seen[0],seen[1]))
        self.assertTrue(torch.equal(sampler.model.denoise_model.calls[0][2]['dense_features'],torch.cat([seen[1]]*2)))
        hook.remove()

if __name__ == '__main__': unittest.main()
