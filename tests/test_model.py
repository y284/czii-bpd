import pytest
import torch

from czii_bpd import config as C
from czii_bpd.model import build_model, load_model


@pytest.mark.parametrize("variant", list(C.VARIANTS))
def test_output_resolution(variant):
    x = torch.zeros(1, 184, 64, 64)  # tomogram depth, 64 x 64 window
    s = C.STRIDE[variant]
    assert build_model(variant).eval()(x).shape == (1, C.NUM_CLASSES, 184 // s, 64 // s, 64 // s)


@pytest.mark.parametrize("variant", list(C.VARIANTS))
def test_checkpoint_loads_into_its_variant(tmp_path, variant):
    torch.save(build_model(variant).state_dict(), tmp_path / "m.bin")
    assert load_model(tmp_path / "m.bin").variant == variant
