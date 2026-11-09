#!/usr/bin/env python3
"""Profile classifier inference and write Perfetto traces.

Each trace file is one eval-mode forward: the recorded pass closest to the
median of five. Five warmup forwards run first and are left out of the file.
The five timed forwards use different inputs. DASH and the 1D CNNs get a new
random vector of the production length each time (the compute does not depend
on the values). Latent and DAEP-init each get a different real Henna spectrum,
because padding changes the encoder's attention cost.

  dash_zeroz      data/pre_trained_models/dash/zeroZ          32x32 CNN
  dash_z          data/pre_trained_models/website_final/dash_z     1D CNN
  dash_noz        data/pre_trained_models/website_final/dash_noz   1D CNN
  latent_z        website_final/latent_z + try_5/Dered36_5 encoder
  latent_noz      website_final/latent_noz + try_5_noz/Nodered36_5 encoder
  daep_init       data/pre_trained_models/daep_init/try_5/split_36
  daep_init_noz   data/pre_trained_models/daep_init/try_5_noz/split_36

Latent and DAEP-init traces both include ``encode_raw`` plus the classification
head. Latent uses the frozen Universal encoder that wrote the embeddings
(``best_ckpt.pt`` next to ``latent_raw_z_best.npz``). DAEP-init uses the matched
encoder-init checkpoint (same encoder and one-hidden-layer head as the latent MLP).

Traces are Chrome JSON. Open one at https://ui.perfetto.dev (Open trace file).
The file contains one span named ``inference``. ``summary.json`` stores that
span and the five timed durations in milliseconds.

    python WiserepData/profile_inference.py
    python WiserepData/profile_inference.py --batch-size 16 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

WISEREP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WISEREP_DIR.parent
ZMODEL_DIR = PROJECT_ROOT / "zmodel_training"
ASTRODASH_APP = PROJECT_ROOT.parent / "astrodash" / "app"
for path in (WISEREP_DIR, ZMODEL_DIR, ASTRODASH_APP):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import constants as const
from astrodash.infrastructure.ml.classifiers.architectures import AstroDashPyTorchNet
from dash_retrain import DashCNN1D
from daep_init_ensemble_plots import _build_model, _load_cfg
from train_latent import LatentClassifier
from TwinsClassifier_Wiserep import REPO
from TwinsModel_Wiserep import build_daep, device_from_str, repo_import_setup

MODELS_ROOT = PROJECT_ROOT / "data" / "pre_trained_models"
HENNA_ROOT = PROJECT_ROOT / "data" / "wiserep_henna"
DEFAULT_OUT = PROJECT_ROOT / "data" / "perfetto_traces"

MODEL_ORDER = (
    "dash_zeroz",
    "dash_z",
    "dash_noz",
    "latent_z",
    "latent_noz",
    "daep_init",
    "daep_init_noz",
)

# Six controlled variants. Original DASH (dash_zeroz) is profiled but not plotted.
PLOT_LABELS = {
    "dash_z": "1D CNN +z",
    "dash_noz": "1D CNN −z",
    "latent_z": "DAEP diffusion +z",
    "latent_noz": "DAEP diffusion −z",
    "daep_init": "DAEP no diffusion +z",
    "daep_init_noz": "DAEP no diffusion −z",
}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_state_dict(path: Path, device: torch.device) -> dict:
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    if isinstance(blob, dict) and any(str(k).endswith("weight") for k in blob):
        return blob
    raise TypeError(f"{path} is not a state dict (keys={list(blob)[:8] if isinstance(blob, dict) else type(blob)})")


def spectrum_batches(
    data_dir: Path,
    batch_size: int,
    n: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """``n`` batches of distinct Henna rows. Mask follows training (True = ignore)."""
    flux_mm = np.load(data_dir / "flux.npy", mmap_mode="r")
    mask_mm = np.load(data_dir / "mask.npy", mmap_mode="r")
    wave = np.asarray(np.load(data_dir / "wavelength.npy"), dtype=np.float32)
    if n * batch_size > flux_mm.shape[0]:
        raise ValueError(f"{data_dir} has {flux_mm.shape[0]} spectra, need {n * batch_size}")
    batches = []
    for i in range(n):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        flux = np.array(flux_mm[sl], dtype=np.float32, copy=True)
        pad = ~np.array(mask_mm[sl], copy=True).astype(bool)
        batches.append(
            {
                "flux": torch.from_numpy(flux).to(device),
                "wavelength": torch.from_numpy(np.repeat(wave[None, :], batch_size, axis=0)).to(device),
                "phase": torch.zeros(batch_size, dtype=torch.float32, device=device),
                "mask": torch.from_numpy(pad).to(device),
            }
        )
    return batches


def load_dash_zeroz(
    batch_size: int,
    device: torch.device,
    n_inputs: int,
) -> tuple[nn.Module, list[torch.Tensor]]:
    path = MODELS_ROOT / "dash" / "zeroZ" / "pytorch_model.pth"
    state = load_state_dict(path, device)
    n_types = int(state["output.weight"].shape[0])
    model = AstroDashPyTorchNet(n_types).to(device)
    model.load_state_dict(state)
    inputs = [torch.randn(batch_size, 32 * 32, device=device) for _ in range(n_inputs)]
    return model, inputs


def load_dash_cnn(
    variant: str,
    batch_size: int,
    device: torch.device,
    n_inputs: int,
) -> tuple[nn.Module, list[torch.Tensor]]:
    path = MODELS_ROOT / "website_final" / variant / "model.pth"
    model = DashCNN1D(
        input_length=const.TARGET_LENGTH,
        num_classes=const.NUM_CLASSES,
    ).to(device)
    model.load_state_dict(load_state_dict(path, device))
    inputs = [torch.randn(batch_size, const.TARGET_LENGTH, device=device) for _ in range(n_inputs)]
    return model, inputs


class LatentEncoderClassifier(nn.Module):
    """Universal ``encode_raw`` then the latent MLP. Score network is not used."""

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.encoder.encode_raw(batch)
        return self.head({"z": z})


def _encoder_ckpt_path(cfg: dict) -> Path:
    latent_npz = Path(cfg["latent_npz"])
    candidates = [
        latent_npz.parent / "best_ckpt.pt",
        MODELS_ROOT / "latent_encoders" / latent_npz.parent.name / "best_ckpt.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No Universal encoder checkpoint for {latent_npz}")


def load_latent(
    variant: str,
    batch_size: int,
    device: torch.device,
    n_inputs: int,
) -> tuple[nn.Module, list[dict[str, torch.Tensor]]]:
    run_dir = MODELS_ROOT / "website_final" / variant
    ckpt = torch.load(run_dir / "classifier_best.pt", map_location=device, weights_only=False)
    cfg = json.loads((run_dir / "cfg_used.json").read_text(encoding="utf-8"))
    if isinstance(ckpt.get("cfg"), dict):
        cfg.update(ckpt["cfg"])
    latent_shape = cfg["latent_shape"]
    embed_dim = int(np.prod(latent_shape[1:]))
    head = LatentClassifier(
        embed_dim=embed_dim,
        n_cls=const.NUM_CLASSES,
        head_hidden=int(cfg["ff_dim"]),
        head_dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    head.load_state_dict(ckpt["model_state_dict"], strict=True)

    enc_path = _encoder_ckpt_path(cfg)
    enc_ckpt = torch.load(enc_path, map_location=device, weights_only=False)
    repo_import_setup(str(REPO.resolve()))
    daep = build_daep(enc_ckpt["cfg"]).to(device)
    daep.load_state_dict(enc_ckpt["state_dict"], strict=True)
    model = LatentEncoderClassifier(daep.encoder, head).to(device)
    data_dir = HENNA_ROOT / ("noderedshift" if cfg.get("use_redshift") is False else "deredshifted")
    print(f"  encoder {enc_path}", flush=True)
    return model, spectrum_batches(data_dir, batch_size, n_inputs, device)


def load_daep_init(
    comparison_name: str,
    batch_size: int,
    device: torch.device,
    n_inputs: int,
) -> tuple[nn.Module, list[dict[str, torch.Tensor]]]:
    noz = "noz" in comparison_name
    run_dir = MODELS_ROOT / "daep_init" / ("try_5_noz" if noz else "try_5") / "split_36"
    ckpt = torch.load(run_dir / "classifier_best.pt", map_location=device, weights_only=False)
    cfg = _load_cfg(run_dir, ckpt)
    model = _build_model(cfg, device, head_state=ckpt["head_state_dict"])
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.head.load_state_dict(ckpt["head_state_dict"])
    data_dir = HENNA_ROOT / ("noderedshift" if noz else "deredshifted")
    print(f"  checkpoint {run_dir / 'classifier_best.pt'}", flush=True)
    return model, spectrum_batches(data_dir, batch_size, n_inputs, device)


LOADERS = {
    "dash_zeroz": lambda b, d, n: load_dash_zeroz(b, d, n),
    "dash_z": lambda b, d, n: load_dash_cnn("dash_z", b, d, n),
    "dash_noz": lambda b, d, n: load_dash_cnn("dash_noz", b, d, n),
    "latent_z": lambda b, d, n: load_latent("latent_z", b, d, n),
    "latent_noz": lambda b, d, n: load_latent("latent_noz", b, d, n),
    "daep_init": lambda b, d, n: load_daep_init("daep_comparison_init", b, d, n),
    "daep_init_noz": lambda b, d, n: load_daep_init("daep_comparison_init_noz", b, d, n),
}


def crop_trace_to_median_inference(trace_path: Path, n: int) -> tuple[list[float], float]:
    """Keep one ``inference`` span: the median of the last ``n``. Earlier spans are warmup."""
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    events = data["traceEvents"]
    spans = sorted(
        (
            e
            for e in events
            if isinstance(e, dict) and e.get("ph") == "X" and e.get("name") == "inference"
        ),
        key=lambda e: e["ts"],
    )
    if len(spans) < n:
        raise RuntimeError(f"{trace_path} has {len(spans)} inference spans, need {n}")
    recorded = spans[-n:]
    median_dur = float(np.median([e["dur"] for e in recorded]))
    kept = min(recorded, key=lambda e: (abs(e["dur"] - median_dur), -e["ts"]))
    t0 = kept["ts"]
    t1 = kept["ts"] + kept["dur"]

    def overlaps(event: dict) -> bool:
        if event.get("ph") == "M" or "ts" not in event:
            return True
        ts = event["ts"]
        dur = event.get("dur") or 0
        return ts + dur >= t0 and ts <= t1

    data["traceEvents"] = [e for e in events if not isinstance(e, dict) or overlaps(e)]
    trace_path.write_text(json.dumps(data), encoding="utf-8")
    span_ms = [float(e["dur"]) / 1000.0 for e in recorded]
    return span_ms, float(kept["dur"]) / 1000.0


def write_trace(
    model: nn.Module,
    inputs: list,
    device: torch.device,
    trace_path: Path,
    *,
    recorded: int,
    with_stack: bool,
) -> tuple[list[float], float]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    # Export after the context exits: kineto results are finalized in __exit__.
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=with_stack,
        with_modules=True,
        acc_events=True,
    ) as prof:
        for inp in inputs:
            with record_function("inference"):
                with torch.inference_mode():
                    model(inp)
                sync(device)
    prof.export_chrome_trace(str(trace_path))
    return crop_trace_to_median_inference(trace_path, recorded)


def profile_one(
    name: str,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    *,
    warmup: int,
    trace_steps: int,
    with_stack: bool,
) -> dict:
    n_inputs = warmup + trace_steps
    print(f"[load] {name}", flush=True)
    model, inputs = LOADERS[name](batch_size, device, n_inputs)
    model.eval()
    params = n_parameters(model)
    if len(inputs) != n_inputs:
        raise RuntimeError(f"{name} produced {len(inputs)} inputs, expected {n_inputs}")

    print(f"[warmup] {name} x{warmup} outside the profiler", flush=True)
    for inp in inputs[:warmup]:
        with torch.inference_mode():
            model(inp)
        sync(device)

    # Profile warmup inputs again, then the held-out spectra. The saved file keeps the latter.
    profiled = list(inputs[:warmup]) + list(inputs[warmup:])
    trace_path = out_dir / f"{name}.perfetto.json"
    print(f"[trace] {name} -> {trace_path}", flush=True)
    span_ms, trace_span_ms = write_trace(
        model,
        profiled,
        device,
        trace_path,
        recorded=trace_steps,
        with_stack=with_stack,
    )

    row = {
        "name": name,
        "parameters": params,
        "batch_size": batch_size,
        "device": str(device),
        "warmup": warmup,
        "trace_steps": trace_steps,
        "span_ms": [round(v, 3) for v in span_ms],
        "median_ms": float(np.median(span_ms)),
        "trace_span_ms": round(trace_span_ms, 3),
        "trace": str(trace_path),
    }
    spans = "  ".join(f"{v:.3f}" for v in row["span_ms"])
    print(
        f"[done] {name}  spans {spans}  trace {row['trace_span_ms']:.3f} ms",
        flush=True,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return row


def plot_inference_times(models: list[dict], out_path: Path) -> None:
    """Horizontal bars are medians. Dots are the five timed forwards."""
    by_name = {row["name"]: row for row in models}
    missing = [name for name in PLOT_LABELS if name not in by_name]
    if missing:
        print(f"[plot] skipped, missing {missing}", flush=True)
        return
    rows = sorted((by_name[name] for name in PLOT_LABELS), key=lambda row: row["median_ms"])
    labels = [PLOT_LABELS[row["name"]] for row in rows]
    medians = [row["median_ms"] for row in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y, medians, color="#4C78A8", height=0.62, zorder=2)
    for i, row in enumerate(rows):
        ax.scatter(
            row["span_ms"],
            np.full(len(row["span_ms"]), i),
            s=28,
            color="#F58518",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        ax.text(
            max(row["span_ms"]) + 0.18,
            i,
            f"{row['median_ms']:.2f}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Inference time (ms)")
    ax.set_title("Forward pass inference time by model")
    xmax = max(max(row["span_ms"]) for row in rows)
    ax.set_xlim(0, xmax + 1.4)
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Perfetto traces of classifier inference.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5, help="Forwards dropped from the trace.")
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=5,
        help="Forwards timed after warmup. The trace file keeps the one closest to the median.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--with-stack", action="store_true", help="Record Python stacks (larger traces).")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Subset of model families.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    device = device_from_str(args.device)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  batch_size={args.batch_size}  out={out_dir}", flush=True)

    rows = []
    for name in args.only:
        rows.append(
            profile_one(
                name,
                args.batch_size,
                device,
                out_dir,
                warmup=args.warmup,
                trace_steps=args.trace_steps,
                with_stack=args.with_stack,
            )
        )

    summary_path = out_dir / "summary.json"
    previous = {}
    if summary_path.is_file() and set(args.only) != set(MODEL_ORDER):
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("device") == str(device) and old.get("batch_size") == args.batch_size:
            previous = {row["name"]: row for row in old.get("models", [])}
    for row in rows:
        previous[row["name"]] = row
    ordered = [previous[name] for name in MODEL_ORDER if name in previous]
    summary = {
        "device": str(device),
        "batch_size": args.batch_size,
        "note": (
            "span_ms are five timed forwards. trace_span_ms is the one written to the Perfetto file, "
            "the duration closest to the median. Each forward uses a different input and includes the GPU sync. "
            "latent_z and latent_noz include Universal encode_raw plus the MLP. "
            "Open each trace at https://ui.perfetto.dev."
        ),
        "models": ordered,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nrecorded inference spans, ms (batch", args.batch_size, "on", device, ")")
    for row in ordered:
        spans = "  ".join(f"{v:7.3f}" for v in row["span_ms"])
        print(f"  {row['name']:<16} {spans}   median {row['median_ms']:.3f}")
    print(f"\ntraces in {out_dir}")
    print("open a *.perfetto.json file at https://ui.perfetto.dev")
    plot_inference_times(ordered, out_dir / "inference_times.png")


if __name__ == "__main__":
    main()
