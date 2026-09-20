import json
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from tools.train_centered_decoder_full import validate_entries, epoch_loader, FullPairDataset

class FakeDataset:
    def __init__(self):
        self.records=[{'sample_id':'p'}, {'sample_id':'c'}]
    def __getitem__(self, i):
        return {'grd_left_imgs':torch.zeros(3,128,512),
                'draw':(random.random(),float(np.random.rand()),float(torch.rand(())))}

class FullEpochTests(unittest.TestCase):
    def test_partial_batch_not_repeated_or_dropped(self):
        rows=list(range(19))
        batches=list(epoch_loader(rows,8,0,3407))
        self.assertEqual([len(x) for x in batches],[8,8,3])
        self.assertEqual([x for b in batches for x in b],rows)
    def test_reject_duplicate_pairs_and_heldout(self):
        a={'name':'a','previous':'p','current':'c','split':'train'}
        self.assertEqual(validate_entries({'items':[a]}),[a])
        with self.assertRaises(ValueError):
            validate_entries({'items':[a,dict(a,name='b')]})
        with self.assertRaises(ValueError):
            validate_entries({'items':[dict(a,split='heldout')]})
    def test_online_sample_reproducible_independent_of_prior_rng(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'reference.npz'
            valid=np.ones((128,512),dtype=bool)
            np.savez_compressed(path,prev_rgb=np.zeros((128,512,3),dtype=np.uint8),
                current_rgb=np.zeros((128,512,3),dtype=np.uint8),support_mask=valid,
                measured_mask=valid,estimated_mask=~valid,source_flat_index=np.arange(128*512).reshape(128,512))
            e={'name':'pair','previous':'p','current':'c','reference_npz':str(path),'split':'train'}
            dataset=FullPairDataset(FakeDataset(),[e],3407)
            first=dataset[0]['sample']['draw']
            random.seed(19); np.random.seed(888); torch.manual_seed(919)
            second=dataset[0]['sample']['draw']
            self.assertEqual(first,second)
    def test_producer_failure_is_not_silent_wait(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'refs'/'missing.npz'
            (Path(d)/'failed.json').write_text(json.dumps({'error':'test'}))
            e={'name':'pair','previous':'p','current':'c','reference_npz':str(path),'split':'train'}
            with self.assertRaisesRegex(RuntimeError,'producer failed'):
                FullPairDataset(FakeDataset(),[e],3407)[0]

if __name__=='__main__':
    unittest.main()
