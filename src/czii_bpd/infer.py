"""Sliding-window inference over full-depth windows, particle extraction and the competition metric."""

import time
from itertools import product

import cc3d
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from . import config as C
from .data import dequantize
from .utils import autocast, sync

# (forward, inverse) pairs acting on the last three dims
VIEWS = {
    "plain": [(lambda x: x, lambda y: y)],
    "tta": [
        (lambda x: x, lambda y: y),
        (lambda x: x.flip(-3), lambda y: y.flip(-3)),
        (lambda x: x.rot90(1, (-1, -2)), lambda y: y.rot90(1, (-2, -1))),
    ],
}
WINDOW = 128  # (y, x) size of a window; windows span the whole depth


def starts(size, patch=WINDOW, even=False):
    """Evenly overlapping window starts covering `size`, one extra window on axes longer than 300."""
    if size <= patch:
        return [0]
    n = int(np.ceil(size / patch)) + (size > 300)
    step = patch - (n * patch - size) / (n - 1)
    pos = {min(int(i * step), size - patch) for i in range(n)}
    return sorted({p - p % 2 for p in pos} if even else pos)


def torch_forward(model, device):
    def forward(x):
        with autocast(device):
            return model(x)

    return forward


@torch.no_grad()
def predict(forward, volume, device, stride, views):
    """Class probabilities (float16, on the output grid) of a uint8 tomogram.

    `forward` maps a (views, depth, 128, 128) float32 batch to logits. Windows are weighted 1.1 inside and 0.1
    within 8 voxels of their (y, x) border.
    """
    w, m = WINDOW // stride, 8 // stride
    weight = torch.full((w, w), 0.1, dtype=torch.float16, device=device)
    weight[m:-m, m:-m] += 1
    shape = tuple(s // stride for s in volume.shape)
    probs = torch.zeros((C.NUM_CLASSES, *shape), dtype=torch.float16, device=device)
    count = torch.zeros(shape[1:], dtype=torch.float16, device=device)
    volume = torch.from_numpy(volume).to(device)

    even = stride > 1
    for y0, x0 in product(starts(volume.shape[1], even=even), starts(volume.shape[2], even=even)):
        x = dequantize(volume[:, y0 : y0 + WINDOW, x0 : x0 + WINDOW])[None]
        y = forward(torch.cat([f(x) for f, _ in views])).float().softmax(1)
        p = sum(inv(y[i : i + 1]) for i, (_, inv) in enumerate(views))[0] / len(views)
        box = (slice(y0 // stride, y0 // stride + w), slice(x0 // stride, x0 // stride + w))
        probs[(slice(None), slice(None), *box)] += p * weight
        count[box] += weight
    return (probs / count[None]).cpu().numpy()


def pick(prob, name, stride):
    """(n, 3) centres as (x, y, z) in Å of the components above the competition threshold that are large enough."""
    _, radius_a, thresh = C.PARTICLES[name]
    r = 0.8 * np.log2(radius_a / C.SCALE)
    min_voxels = 4 / 3 * np.pi * r**3 / C.BLOB_FACTOR / stride**3
    stats = cc3d.statistics(cc3d.connected_components(prob > thresh, connectivity=18))
    zyx = stats["centroids"][1:][stats["voxel_counts"][1:] > min_voxels]
    return (zyx * stride + (stride - 1) / 2)[:, ::-1] * C.SCALE


def truth_angstrom(picks):
    """Picks in (z, y, x) voxels -> (x, y, z) in Å, the frame `pick` returns."""
    return {name: c[:, ::-1] * C.SCALE for name, c in picks.items()}


def f_beta(pred, truth, max_dist, beta=4):
    """F-beta of one-to-one matched predictions, a hit being within `max_dist`."""
    tp = 0
    if len(pred) and len(truth):
        dist = np.linalg.norm(pred[:, None] - truth[None], axis=-1)
        i, j = linear_sum_assignment(dist)
        tp = int((dist[i, j] <= max_dist).sum())
    p = tp / len(pred) if len(pred) else 0
    r = tp / len(truth) if len(truth) else 0
    return (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0


def evaluate(forward, volume, truth, device, stride, views):
    """Weighted F4 on one tomogram, per-class F4 and predicted counts, network and particle-extraction seconds."""
    sync(device)
    start = time.perf_counter()
    probs = predict(forward, volume, device, stride, views)
    sync(device)
    network_s, start = time.perf_counter() - start, time.perf_counter()
    per_class = {}
    for name, (cls, radius_a, _) in C.PARTICLES.items():
        pred = pick(probs[cls], name, stride)
        per_class[name] = (f_beta(pred, truth[name], radius_a / 2), len(pred))
    f4 = sum(C.WEIGHTS[n] * s for n, (s, _) in per_class.items()) / sum(C.WEIGHTS.values())
    return {
        "f4": f4,
        "per_class": per_class,
        "network_s": network_s,
        "components_s": time.perf_counter() - start,
        "probs": probs,
    }
