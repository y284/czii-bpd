import numpy as np
import pytest
import torch

from czii_bpd.export import export_onnx
from czii_bpd.infer import VIEWS
from czii_bpd.model import UNet3D

ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("half", [False, True])
def test_onnx_export_matches_pytorch(tmp_path, half):
    model = UNet3D(half=half).eval()
    x = torch.randn(len(VIEWS["tta"]), 32, 32, 32)
    export_onnx(model, x, tmp_path / "m.onnx")
    out = ort.InferenceSession(str(tmp_path / "m.onnx")).run(None, {"volume": x.numpy()})[0]
    assert np.abs(out - model(x).detach().numpy()).max() < 1e-4
