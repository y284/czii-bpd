import pytest
import torch

from czii_bpd import config as C
from czii_bpd.model import UNet3D


@pytest.mark.parametrize("variant", ["full", "half"])
def test_output_resolution(variant):
    model = UNet3D(half=C.STRIDE[variant] == 2).eval()
    x = torch.zeros(1, 184, 64, 64)  # tomogram depth, 64 x 64 window
    s = C.STRIDE[variant]
    assert model(x).shape == (1, C.NUM_CLASSES, x.shape[1] // s, 64 // s, 64 // s)
