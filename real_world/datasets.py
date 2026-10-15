"""
torchvision data loading for CIFAR-10, CIFAR-100, SVHN and MNIST, without TensorFlow Datasets or JAX.
LSUN, FFHQ and CelebA-HQ are not supported by this file.

Interface matches what run_lib.py / utils.py expect:
    train_ds, eval_ds, _ = get_dataset(config)
    batch = next(iter(train_ds))['image']   # plain torch tensor, (N,H,W,C), float in [0,1]
(The original code called `next(iter_ds)['image']._numpy()`, a TF
EagerTensor method; call sites in run_lib.py/utils.py are patched to drop
the now-unnecessary `._numpy()`.)
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as T


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


_TORCHVISION_BUILDERS = {
    'CIFAR10': torchvision.datasets.CIFAR10,
    'CIFAR100': torchvision.datasets.CIFAR100,
    'SVHN': torchvision.datasets.SVHN,
    'MNIST': torchvision.datasets.MNIST,
}


class _DictWrapper(torch.utils.data.Dataset):
    """Wraps a torchvision dataset to yield {'image': HWC float32 in [0,1], 'label': ...}
    dicts, applying random flip / uniform dequantization / resize the same
    way the original TF preprocessing pipeline did."""

    def __init__(self, base_dataset, image_size, random_flip, uniform_dequantization):
        self.base = base_dataset
        self.image_size = image_size
        self.random_flip = random_flip
        self.uniform_dequantization = uniform_dequantization

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = np.array(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        if img.ndim == 2:  # e.g. MNIST grayscale -> add channel dim
            img = img[:, :, None]
        if self.random_flip and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            # Nearest/bilinear resize via torch, kept minimal since CIFAR/SVHN/MNIST
            # are already at (or near) the target resolution in this repo's configs.
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(self.image_size, self.image_size),
                                                 mode='bilinear', align_corners=False, antialias=True)
            img = t.squeeze(0).permute(1, 2, 0).numpy()
        if self.uniform_dequantization:
            img = (np.random.uniform(size=img.shape).astype(np.float32) + img * 255.) / 256.
        return {'image': torch.from_numpy(img), 'label': label if label is not None else -1}


class _InfiniteLoader:
    """Wraps a DataLoader to yield batches forever, matching ds.repeat() semantics."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            return next(self._iter)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collections.ConfigDict parsed from config files.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, iterate over the eval loader once (no infinite repeat).
    Returns:
      train_ds, eval_ds, None (the third slot -- `dataset_builder` in the original
      code -- is unused everywhere it's called in this repo, so it's dropped).
    """
    if config.data.dataset not in _TORCHVISION_BUILDERS:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not supported by this lightweight '
            f'loader (only CIFAR10/CIFAR100/SVHN/MNIST). For LSUN/FFHQ/CelebA-HQ, '
            f'restore datasets_tfds_original.py.bak and its TF/TFDS dependency.'
        )
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    builder = _TORCHVISION_BUILDERS[config.data.dataset]

    if config.data.dataset == 'SVHN':
        train_base = builder(root='./data', split='train', download=True)
        eval_base = builder(root='./data', split='test', download=True)
    else:
        train_base = builder(root='./data', train=True, download=True)
        eval_base = builder(root='./data', train=False, download=True)

    train_wrapped = _DictWrapper(train_base, config.data.image_size,
                                  config.data.random_flip, uniform_dequantization)
    eval_wrapped = _DictWrapper(eval_base, config.data.image_size,
                                 False, uniform_dequantization)

    train_loader = torch.utils.data.DataLoader(
        train_wrapped, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2, persistent_workers=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_wrapped, batch_size=batch_size, shuffle=(not evaluation), drop_last=True,
        num_workers=2, persistent_workers=True,
    )

    train_ds = _InfiniteLoader(train_loader)
    eval_ds = eval_loader if evaluation else _InfiniteLoader(eval_loader)
    return train_ds, eval_ds, None
