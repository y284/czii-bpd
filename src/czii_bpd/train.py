"""Train one model per (variant, seed) on 6 runs; every epoch is scored on the held-out run.

The checkpoint with the best held-out F4 is kept as model_<variant>_seed<seed>_best.bin, the last epoch as
model_<variant>_seed<seed>.bin.

    python -m czii_bpd.train --variants full half --seeds 42 90 --fold 0 --eval tta
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from .data import PatchDataset, load_picks, load_volume, render_labels
from .infer import VIEWS, evaluate, torch_forward, truth_angstrom
from .model import build_model
from .utils import amp_dtype, autocast, run_on_devices, seed_everything


def train(device, job, data, valid, args):
    variant, seed, model = job
    tag = f"{variant} seed {seed} · {device}"
    stride = C.STRIDE[variant]
    model.to(device).train()
    loader = DataLoader(
        PatchDataset(data, stride, args.samples, 0.02 if seed == 42 else 0.03),
        args.batch_size,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(seed),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and amp_dtype(device) == torch.float16)
    history, best, start = [], -1, time.time()

    for epoch in range(1, args.epochs + 1):
        opt = torch.optim.Adam(model.parameters(), args.lr)  # re-created every epoch, as in the competition runs
        total = 0.0
        for step, batch in enumerate(loader, 1):
            with autocast(device):
                loss = F.cross_entropy(
                    model(batch["volume"].to(device)), batch["label"].to(device), label_smoothing=0.01
                )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            if step % args.log_every == 0 or step == len(loader):
                done, elapsed = (epoch - 1) * len(loader) + step, time.time() - start
                left = elapsed / done * (args.epochs * len(loader) - done)
                print(
                    f"[{tag}] epoch {epoch}/{args.epochs} step {step}/{len(loader)} loss {total / step:.4f} | "
                    f"{elapsed / 60:.0f} min elapsed, ~{left / 60:.0f} min left",
                    flush=True,
                )

        model.eval()
        res = evaluate(torch_forward(model, device), valid["volume"], valid["truth"], device, stride, VIEWS[args.eval])
        model.train()
        history.append(
            {
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "loss": round(total / len(loader), 4),
                "f4": round(res["f4"], 4),
                **{n: round(s, 3) for n, (s, _) in res["per_class"].items()},
            }
        )
        if res["f4"] > best:
            best = res["f4"]
            torch.save(model.state_dict(), args.out / f"model_{variant}_seed{seed}_best.bin")
        counts = ", ".join(f"{n[:4]} {s:.2f} {k}/{len(valid['truth'][n])}" for n, (s, k) in res["per_class"].items())
        print(
            f"[{tag}] epoch {epoch} loss {history[-1]['loss']:.4f} {args.eval} F4 {res['f4']:.4f} ({counts})",
            flush=True,
        )

    torch.save(model.state_dict(), args.out / f"model_{variant}_seed{seed}.bin")
    top = max(history, key=lambda h: h["f4"])
    return {
        "variant": variant,
        "seed": seed,
        "best_epoch": top["epoch"],
        "best_f4": top["f4"],
        "last_f4": history[-1]["f4"],
        "train_min": round((time.time() - start) / 60, 1),
    }, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(C.VARIANTS), choices=list(C.VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 90])
    p.add_argument("--fold", type=int, default=0, help="index of the held-out run in RUNS")
    p.add_argument("--eval", default="tta", choices=list(VIEWS), help="views used to score each epoch")
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--samples", type=int, default=1024, help="random crops per epoch")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=64)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--smoke", action="store_true", help="2 epochs of 32 crops, to check the setup")
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.samples = 2, 32
    args.out.mkdir(parents=True, exist_ok=True)

    valid_run = C.RUNS[args.fold]
    strides = {C.STRIDE[v] for v in args.variants}
    data = {}
    for run in C.RUNS:
        if run != valid_run:
            picks, volume = load_picks(run), load_volume(run)
            data[run] = (volume, {s: render_labels(picks, volume.shape, s) for s in strides})
            print("loaded", run, flush=True)
    valid = {"volume": load_volume(valid_run), "truth": truth_angstrom(load_picks(valid_run))}

    jobs = []
    for variant, seed in product(args.variants, args.seeds):
        seed_everything(seed)
        jobs.append((variant, seed, build_model(variant)))
    results = run_on_devices(jobs, lambda device, job: train(device, job, data, valid, args))

    summary, history = [s for s, _ in results], [h for _, hs in results for h in hs]
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    (args.out / "results_training.json").write_text(
        json.dumps({"valid_run": valid_run, "eval": args.eval, "summary": summary, "history": history}, indent=2)
    )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
