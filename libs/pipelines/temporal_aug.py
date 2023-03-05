from typing import Optional, List
from torch import Tensor
import numpy as np
from mmaction.datasets import PIPELINES


@PIPELINES.register_module()
class FrameShuffle:
    """Shuffle the frames order
    """

    def __init__(self,
                 num_segments=8,
                 seed=42,
                 perm: Optional[List] = None,
                 dim=0):
        self.num_segments = num_segments
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.perm = perm

    def __call__(self, results):
        if self.perm is None:
            perm = self.rng.permutation(self.num_segments)
        else:
            perm = self.perm
        imgs = results['imgs']
        if isinstance(imgs, list):
            shuffle = [imgs[idx] for idx in perm]
            results['imgs'] = shuffle
        elif isinstance(imgs, (Tensor, np.ndarray)):
            results['imgs'] = imgs[perm]
        else:
            raise ValueError('Not support type', type(imgs))
        return results

    def __repr__(self):
        if self.perm is None:
            repr_str = f'{self.__class__.__name__}(random seed={self.seed}, num_segments={self.num_segments})'
        else:
            repr_str = f'{self.__class__.__name__}(perm = {self.perm})'
        return repr_str
