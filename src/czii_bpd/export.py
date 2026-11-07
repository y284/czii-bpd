"""Export to ONNX and TensorRT (FP16) and check them on the held-out tomogram against PyTorch autocast.

Every backend runs the same inference (3-view TTA, full-depth windows, competition thresholds); the table reports
its F4, network and particle-extraction time, and the largest probability difference from PyTorch.

    python -m czii_bpd.export --checkpoints outputs/model_full_seed42_best.bin outputs/model_half_seed42_best.bin
"""

import argparse
import copy
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import load_picks, load_volume
from .infer import VIEWS, WINDOW, evaluate, torch_forward, truth_angstrom
from .model import load_model
from .utils import amp_dtype

warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")


def export_onnx(model, x, path):
    """TorchScript exporter under no_grad: tracing with autograd keeps every activation in memory."""
    with torch.no_grad():
        torch.onnx.export(
            model, x, path, input_names=["volume"], output_names=["logits"], opset_version=17, dynamo=False
        )


def ort_forward(path, x, out):
    """forward(batch) for ONNX Runtime, reading and writing the fixed GPU buffers x and out."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    i = x.device.index or 0
    cuda = {"device_id": i, "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC"}
    session = ort.InferenceSession(path, providers=[("CUDAExecutionProvider", cuda)])
    assert "CUDAExecutionProvider" in session.get_providers(), "ONNX Runtime fell back to CPU"
    binding = session.io_binding()
    binding.bind_input("volume", "cuda", i, np.float16, list(x.shape), x.data_ptr())
    binding.bind_output("logits", "cuda", i, np.float16, list(out.shape), out.data_ptr())

    def forward(batch):
        x.copy_(batch)
        session.run_with_iobinding(binding)
        return out

    return forward


def trt_forward(path, x, out):
    """forward(batch) for a TensorRT FP16 engine: strongly typed on recent TensorRT, FP16 flag on older ones."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = trt.NetworkDefinitionCreationFlag
    typed = hasattr(flags, "STRONGLY_TYPED")
    if typed:
        network = builder.create_network(1 << int(flags.STRONGLY_TYPED))
    else:
        network = builder.create_network(
            1 << int(flags.EXPLICIT_BATCH) if int(trt.__version__.split(".")[0]) < 10 else 0
        )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(path).read_bytes()):
        raise RuntimeError(parser.get_error(0))
    config = builder.create_builder_config()
    if not typed:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    context = trt.Runtime(logger).deserialize_cuda_engine(serialized).create_execution_context()
    context.set_tensor_address("volume", x.data_ptr())
    context.set_tensor_address("logits", out.data_ptr())

    def forward(batch):
        x.copy_(batch)
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return out

    return forward


def free():
    gc.collect()
    torch.cuda.empty_cache()


def check(path, valid, views, device, out_dir):
    model = load_model(path, device).requires_grad_(False)
    name, stride = model.variant, 2 if model.half_res else 1
    shape = (len(views), valid["volume"].shape[0], WINDOW, WINDOW)
    x16 = torch.zeros(shape, dtype=torch.float16, device=device)
    with torch.no_grad():
        out_shape = model(x16[:1].float()).shape[1:]
    onnx = str(out_dir / f"{name}_fp16.onnx")
    model16 = copy.deepcopy(model).half()
    export_onnx(model16, x16, onnx)
    del model16
    free()
    out16 = lambda: torch.empty((len(views), *out_shape), dtype=torch.float16, device=device)  # noqa: E731

    backends = {
        f"PyTorch autocast {str(amp_dtype(device)).split('.')[-1]}": lambda: torch_forward(model, device),
        "ONNX Runtime FP16": lambda: ort_forward(onnx, x16, out16()),
        "TensorRT FP16": lambda: trt_forward(onnx, x16, out16()),
    }
    rows, reference = [], None
    for backend, make in backends.items():
        try:
            forward = make()
            with torch.no_grad():
                for _ in range(3):
                    forward(x16.float())
            res = evaluate(forward, valid["volume"], valid["truth"], device, stride, views)
        except Exception as e:
            print(f"{name}: {backend} failed -> {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        probs = res.pop("probs")
        reference = probs if reference is None else reference
        diff = max(float(np.abs(a.astype(np.float32) - b).max()) for a, b in zip(probs, reference))
        rows.append(
            {
                "variant": name,
                "backend": backend,
                "network_s": round(res["network_s"], 2),
                "components_s": round(res["components_s"], 2),
                "f4": round(res["f4"], 4),
                "max |Δ prob|": f"{diff:.1e}",
            }
        )
        print(rows[-1], flush=True)
        del forward, probs
        free()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--fold", type=int, default=0, help="index of the held-out run in RUNS")
    p.add_argument("--eval", default="tta", choices=list(VIEWS))
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "the export check needs a GPU"
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    run = C.RUNS[args.fold]
    valid = {"volume": load_volume(run), "truth": truth_angstrom(load_picks(run))}
    df = pd.DataFrame([r for path in args.checkpoints for r in check(path, valid, VIEWS[args.eval], device, args.out)])
    df["total_s"] = df.network_s + df.components_s
    base = df[(df.variant == "full") & df.backend.str.startswith("PyTorch")]
    if len(base):
        df["network speed-up"] = (base.network_s.iloc[0] / df.network_s).round(2)
        df["total speed-up"] = (base.total_s.iloc[0] / df.total_s).round(2)
    (args.out / "results_export.json").write_text(
        json.dumps(
            {"valid_run": run, "eval": args.eval, "results": df.to_dict("records")}, indent=2, ensure_ascii=False
        )
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
