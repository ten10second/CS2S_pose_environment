import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler

from tools.train_temporal_pairs import ConsecutivePairDataset


class PairDataTests(unittest.TestCase):
    def test_history_reads_only_rgb_with_the_original_transform(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'previous.png'
            pixels = np.arange(6*8*3, dtype=np.uint8).reshape(6,8,3)
            Image.fromarray(pixels).save(path)
            class Base:
                records = [dict(date='d',drive='v',frame_index=i,image_02_path=str(path)) for i in range(2)]
                seen = []
                @staticmethod
                def grd_transform(image):
                    return torch.from_numpy(np.array(image.resize((4,3)), copy=True)).permute(2,0,1).float()/255
                def __getitem__(self, index):
                    self.seen.append(index)
                    if index == 0:
                        raise AssertionError('unused previous-frame sensor cache was loaded')
                    return {'grd_left_imgs':torch.ones(3,3,4)}
            base=Base()
            pair=ConsecutivePairDataset(base,build_geometry=False)[0]
            with Image.open(path) as image:
                expected=base.grd_transform(image.convert('RGB'))
            self.assertTrue(torch.equal(pair['prev']['grd_left_imgs'],expected))
            self.assertEqual(set(pair['prev']),{'grd_left_imgs'})
            self.assertEqual(base.seen,[1])

    def test_full_epoch_b4_ddp_covers_every_pair(self):
        data=list(range(14463))
        seen=[]
        for rank in range(2):
            sampler=DistributedSampler(data,num_replicas=2,rank=rank,shuffle=True,seed=3407)
            loader=DataLoader(data,batch_size=4,sampler=sampler,drop_last=True)
            self.assertEqual(len(loader),1808)
            seen.extend(int(i) for batch in loader for i in batch)
        self.assertEqual(len(seen),14464)
        self.assertEqual(set(seen),set(data))


if __name__ == '__main__': unittest.main()
