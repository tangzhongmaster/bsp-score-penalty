"""
NumPy implementation of the three two-dimensional datasets (checkerboard, gaussian_8, spiral_2).
The generation formulas are the ones of toy_datasets/*.py, without the TensorFlow Datasets wrapper.

Interface matches what run_lib.py / run_lib_eval.py expect:
    ds = get_dataset(config)
    iter_ds = iter(ds)
    batch = next(iter_ds)['position']   # plain torch tensor, shape (batch_size, 2)
(The original code called `next(iter_ds)['position']._numpy()`, a TF
EagerTensor method; call sites are patched in run_lib.py/run_lib_eval.py to
drop the now-unnecessary `._numpy()`.)
"""
import numpy as np
import torch


def _checkerboard(n):
    r = 3
    x1 = np.random.rand(n) * 4 - 2
    x2_ = np.random.rand(n) - np.random.randint(0, 2, (n,)).astype('float64') * 2
    x2 = x2_ + np.floor(x1) % 2
    return (np.stack((x1, x2), axis=1) * r).astype(np.float64)


def _gaussian_8(n):
    scale = 8
    sq2 = 1 / np.sqrt(2)
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (sq2, sq2), (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]
    centers = np.array([(scale * x, scale * y) for x, y in centers])
    idx = np.random.randint(0, len(centers), size=(n,))
    return (sq2 * (np.random.randn(n, 2) + centers[idx])).astype(np.float64)


def _spiral_2(n):
    r = 2.3 / 3.0
    half = n // 2
    theta = np.sqrt(np.random.rand(half)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(theta) * theta + np.random.rand(half) * 0.5
    d1y = np.sin(theta) * theta + np.random.rand(half) * 0.5
    pts = np.concatenate([np.stack([d1x, d1y], axis=1), np.stack([-d1x, -d1y], axis=1)], axis=0) * r
    return pts.astype(np.float64)


_GENERATORS = {
    'checkerboard': _checkerboard,
    'gaussian_8': _gaussian_8,
    'spiral_2': _spiral_2,
}


class _InfiniteBatchIterator:
    """Yields dict batches forever, matching the original ds.repeat().shuffle().batch() behavior."""

    def __init__(self, dataset_name, batch_size, dtype):
        if dataset_name not in _GENERATORS:
            raise NotImplementedError('Dataset {} not yet supported.'.format(dataset_name))
        self.gen_fn = _GENERATORS[dataset_name]
        self.batch_size = batch_size
        self.dtype = dtype

    def __iter__(self):
        return self

    def __next__(self):
        pts = self.gen_fn(self.batch_size)
        position = torch.from_numpy(pts).to(self.dtype)
        return {'position': position}


def get_dataset(config):
    """Construct a data iterator. Drop-in replacement for the TFDS version.

    Args:
      config: (ConfigDict) same config object used elsewhere in this repo.
    Returns:
      An object supporting iter(...) that yields {'position': tensor} batches.
    """
    return _InfiniteBatchIterator(config.data.dataset, config.training.batch_size, config.data.dtype)
