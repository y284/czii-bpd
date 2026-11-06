import json
import random

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from . import config as C

STEP = (C.Q_HI - C.Q_LO) / 255


def load_volume(run, split="train"):
    """Denoised tomogram, normalised to [P_LO, P_HI] -> [0, 1] and stored as uint8."""
    path = C.DATA_DIR / split / "static/ExperimentRuns" / run / "VoxelSpacing10.000/denoised.zarr"
    v = (zarr.open(str(path), mode="r")[0][:] - C.P_LO) / (C.P_HI - C.P_LO)
    return np.clip(np.rint((v - C.Q_LO) / STEP), 0, 255).astype(np.uint8)


def dequantize(v):
    v = v.float() if torch.is_tensor(v) else v.astype(np.float32)
    return v * STEP + C.Q_LO


def load_picks(run):
    """{particle: (n, 3) centres as (z, y, x) voxel coordinates}."""
    picks = {}
    for name in C.PARTICLES:
        path = C.DATA_DIR / "train/overlay/ExperimentRuns" / run / "Picks" / f"{name}.json"
        points = json.loads(path.read_text())["points"]
        picks[name] = np.array([[p["location"][a] for a in "zyx"] for p in points]).reshape(-1, 3) / C.SCALE
    return picks


def render_labels(picks, shape, stride=1):
    """One sphere of radius 0.8 * log2(radius in voxels) per particle, on the grid of a model with this stride.

    stride 1 reproduces the competition labels (sphere centred on the box corner + r).
    stride 2: 0.5x trilinear puts output voxel j at input 2j + 0.5, so centres map to (c - 0.5) / 2.
    """
    shape = tuple(s // stride for s in shape)
    label = np.zeros(shape, np.int8)
    for name, (cls, radius_a, _) in C.PARTICLES.items():
        r = 0.8 * np.log2(radius_a / C.SCALE) / stride
        for c in (picks[name] - (stride - 1) / 2) / stride:
            lo = np.maximum(np.round(c - r).astype(int), 0)
            hi = np.minimum(np.round(c + r).astype(int) + 1, shape)
            if stride == 1:
                c = lo + r
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
            z, y, x = np.ogrid[box]
            label[box][(z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r] = cls
    return label


class PatchDataset(Dataset):
    """Random crops with flips, 90° rotations in (y, x) and a small mean/std jitter.

    data: {run: (uint8 volume, {stride: labels})}. Crop starts are multiples of the stride so both grids stay aligned.
    """

    def __init__(self, data, stride=1, length=1024, shift=0.02, patch=C.PATCH):
        self.data, self.stride, self.length, self.shift, self.patch = data, stride, length, shift, patch

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        s, p = self.stride, self.patch
        volume, labels = self.data[random.choice(list(self.data))]
        start = [s * random.randrange((d - q) // s) for q, d in zip(p, volume.shape)]
        v = dequantize(volume[tuple(slice(a, a + q) for a, q in zip(start, p))])
        y = labels[s][tuple(slice(a // s, (a + q) // s) for a, q in zip(start, p))]

        for axis in range(3):
            if torch.rand(1) < 0.5:
                v, y = np.flip(v, axis), np.flip(y, axis)
        k = random.randrange(4)
        v, y = np.rot90(v, k, (1, 2)), np.rot90(y, k, (1, 2))

        d_mean, d_std = ((torch.rand(2) * 2 - 1) * self.shift).tolist()
        m = v.mean()
        v = (v - m) * (1 + d_std) + m * (1 + d_mean)
        return {"volume": torch.from_numpy(v.copy()), "label": torch.from_numpy(y.astype(np.int64))}
