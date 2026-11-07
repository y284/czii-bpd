import numpy as np
import pytest
import torch

from czii_bpd import config as C
from czii_bpd.export import export_onnx
from czii_bpd.infer import VIEWS
from czii_bpd.model import build_model

ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("variant", list(C.VARIANTS))
def test_onnx_export_matches_pytorch(tmp_path, variant):
    model = build_model(variant).eval()
    x = torch.randn(len(VIEWS["tta"]), 32, 32, 32)
    export_onnx(model, x, tmp_path / "m.onnx")
    out = ort.InferenceSession(str(tmp_path / "m.onnx")).run(None, {"volume": x.numpy()})[0]
    assert np.abs(out - model(x).detach().numpy()).max() < 1e-4
