"""
WISeREP spectra -> DAEP encoder (no diffusion) -> flatten -> one-hidden-layer MLP -> CE.

The encoder config in ``Test/cfg_used.json`` matches the Universal diffusion
encoder (16 x 64). The head matches ``LatentClassifier``: one hidden layer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from iau_train_val_test_split import IAU_SPLIT_SEED, split_row_indices_by_iau_train_val_test
from TwinsModel_Wiserep import build_daep, default_cfg, device_from_str, repo_import_setup
from TwinsTrain_Wiserep import collate_fixed, load_data, set_seeds, to_device

USE_REDSHIFT_VALUES = [False, True]
ITER_VALUES = range(5)

# Paths
_WISEREP_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _WISEREP_DIR.parent

TEST_ROOT = _WISEREP_DIR / "Test"
HENNA_ROOT = _PROJECT_ROOT / "data" / "wiserep_henna"

# DAEP encoder diffusion code path
REPO = _WISEREP_DIR / "Perceiver-diffusion-autoencoder"

DAEP_CFG_JSON = TEST_ROOT / "cfg_used.json"


def data_dir_for(use_redshift: bool) -> pathlib.Path:
    return HENNA_ROOT / ("deredshifted" if use_redshift else "noderedshift")


def out_dir_for(use_redshift: bool, iter_idx: int) -> pathlib.Path:
    return TEST_ROOT / ("daep_comparison_init" if use_redshift else "daep_comparison_init_noz") / f"iter_{iter_idx}"


# --latent-dirs writes data/pre_trained_models/daep_init/<try_N>/split_<seed>/
DEFAULT_INIT_OUT_ROOT = _PROJECT_ROOT / "data" / "pre_trained_models" / "daep_init"

# Labels
LABEL_COLUMN = "Obj. Type"
IAU_COLUMN = "IAU name"

# 5-class label mapping
LABEL_MAP: Dict[str, str] = {
    "SN Ia": "SN Ia",
    "SN Ia-CSM": "SN Ia",
    "SN Ia-91T-like": "SN Ia",
    "SN Ia-SC": "SN Ia",
    "SN Ia-91bg-like": "SN Ia",
    "SN Ia-pec": "SN Ia",
    "SN Ia-Ca-rich": "SN Ia",
    "SN Iax[02cx-like]": "SN Ia",
    "Computed-Ia": "SN Ia",
    "SN Ib": "SN Ib/c",
    "SN Ic": "SN Ib/c",
    "SN Ib/c": "SN Ib/c",
    "SN Ib-Ca-rich": "SN Ib/c",
    "SN Ib-pec": "SN Ib/c",
    "SN Ibn": "SN Ib/c",
    "SN Ic-BL": "SN Ib/c",
    "SN Ic-Ca-rich": "SN Ib/c",
    "SN Ic-pec": "SN Ib/c",
    "SN Icn": "SN Ib/c",
    "SN Ib/c-Ca-rich": "SN Ib/c",
    "SN Ibn/Icn": "SN Ib/c",
    "SN II": "SN II",
    "SN IIP": "SN II",
    "SN IIL": "SN II",
    "SN II-pec": "SN II",
    "SN IIb": "SN II",
    "Computed-IIP": "SN II",
    "Computed-IIb": "SN II",
    "SN IIn": "SN IIn",
    "SN IIn-pec": "SN IIn",
    "SLSN-I": "SLSN-I",
    "SLSN-II": "SLSN-I",
    "SLSN-R": "SLSN-I",
}

CLASS_NAMES = ["SN Ia", "SN Ib/c", "SN II", "SN IIn", "SLSN-I"]
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Classifier training (mean-pool MLP on frozen encoder)
IAU_TRAIN_FRAC = 0.8
IAU_VAL_FRAC = 0.1
IAU_TEST_FRAC = 0.1
SEED = 0
DEVICE = "auto"

BATCH_SIZE = 16
LR = 4e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
# Encoder dropout stays in cfg["dropout"] (0.15, same as the diffusion encoder).
# The classification MLP uses this, same as LatentClassifier.
HEAD_DROPOUT = 0.25


def build_daep_cfg() -> dict:
    cfg = default_cfg()
    cfg.update(json.loads(DAEP_CFG_JSON.read_text(encoding="utf-8")))
    print("DAEP cfg merged from", DAEP_CFG_JSON.resolve())
    print(
        "  encoder cfg: bottleneck_dim=", cfg.get("bottleneck_dim"),
        "bottleneck_length=", cfg.get("bottleneck_length"),
        "model_dim=", cfg.get("model_dim"),
        "ff_dim=", cfg.get("ff_dim"),
        "dropout=", cfg.get("dropout"),
        "head_dropout=", cfg.get("head_dropout", HEAD_DROPOUT),
        "n_hidden_layers=", cfg.get("n_hidden_layers", 1),
        flush=True,
    )
    return cfg


def load_preprocessed_for_classifier(data_dir: pathlib.Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (meta, wave, flux, mask_valid) for split_row_indices_by_iau_train_val_test
    and LabeledSpectra
    """
    data_dir = pathlib.Path(data_dir).resolve()
    print("[load] Preprocess outputs (CSV + npy) from", data_dir, flush=True)
    meta, wave, flux, mask = load_data(data_dir)
    mask_valid = mask.astype(bool, copy=False)
    print(
        f"meta rows={len(meta)}  flux={flux.shape}  mask_valid={mask_valid.shape}  wave len={len(wave)}",
        flush=True,
    )
    return meta, wave, flux, mask_valid

def row_class_idx(meta: pd.DataFrame, row: int, label_col: str) -> int:
    """Map label_col -> one of 5 classes via LABEL_MAP; -1 if unknown."""
    s = str(meta.loc[row, label_col]).strip()
    if s.lower() == "nan" or s == "":
        return -1
    canonical = LABEL_MAP.get(s)
    if canonical is None:
        return -1
    return int(CLASS_TO_IDX.get(canonical, -1))


def filter_indices_mapped(meta: pd.DataFrame, indices: np.ndarray, label_col: str) -> np.ndarray:
    """Keep only row indices whose label_col maps to a class in CLASS_NAMES."""
    keep = [int(i) for i in indices if row_class_idx(meta, int(i), label_col) >= 0]
    return np.asarray(keep, dtype=np.int64)


def inverse_frequency_class_weights(meta: pd.DataFrame, train_indices: np.ndarray, label_col: str) -> torch.Tensor:
    """Per-class CE weights proportional to 1 / frequency on the training indices"""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in train_indices:
        y = row_class_idx(meta, int(i), label_col)
        if y >= 0:
            counts[y] += 1.0
    total = float(counts.sum())
    freq = counts / total
    w = 1.0 / np.maximum(freq, 1e-12)
    return torch.tensor(w, dtype=torch.float32)


def report_split_label_counts(meta: pd.DataFrame, idx: np.ndarray, label_col: str, tag: str) -> None:
    counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    unk = 0
    for i in idx:
        y = row_class_idx(meta, int(i), label_col)
        if y < 0:
            unk += 1
        else:
            counts[CLASS_NAMES[y]] += 1
    print(f"[labels] {tag}  per-class={counts}  unmappable={unk}", flush=True)


class LabeledSpectra(Dataset):
    def __init__(self, meta, wave, flux, mask_valid, indices, label_col):
        self.meta = meta.reset_index(drop=True)
        self.wave = wave.astype(np.float32)
        self.flux = flux.astype(np.float32)
        self.mask_valid = mask_valid.astype(bool)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_col = label_col

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        ph = 0.0
        for col in ("Phase (days)", "phase_used", "phase_rest", "phase_obs"):
            if col in self.meta.columns:
                ph = self.meta.loc[row, col]
                break
        try:
            ph = float(ph) if np.isfinite(float(ph)) else 0.0
        except Exception:
            ph = 0.0
        pad = ~self.mask_valid[row]
        return {
            "idx": row,
            "flux": torch.from_numpy(self.flux[row]),
            "wavelength": torch.from_numpy(self.wave),
            "phase": torch.tensor(ph, dtype=torch.float32),
            "mask": torch.from_numpy(pad).bool(),
            "y": torch.tensor(row_class_idx(self.meta, row, self.label_col), dtype=torch.long),
        }


def collate_labeled(batch: List[dict]):
    y = torch.stack([b.pop("y") for b in batch], dim=0)
    b = collate_fixed(batch)
    b["y"] = y
    return b


def n_hidden_layers_from_state(head_state: dict) -> int:
    """Hidden Linear blocks in a saved head. The last Linear is the class layer."""
    n_linear = sum(1 for key in head_state if str(key).endswith(".weight"))
    if n_linear < 2:
        raise ValueError(
            f"Classification head state has {n_linear} Linear layers; expected at least 2"
        )
    return n_linear - 1


class MeanPoolClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        n_cls: int,
        head_hidden: int,
        head_dropout: float,
        n_hidden_layers: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")
        layers: List[nn.Module] = []
        in_dim = embed_dim
        for _ in range(n_hidden_layers):
            layers.extend(
                (
                    nn.Linear(in_dim, head_hidden),
                    nn.GELU(),
                    nn.Dropout(head_dropout),
                )
            )
            in_dim = head_hidden
        layers.append(nn.Linear(in_dim, n_cls))
        self.head = nn.Sequential(*layers)

    def forward(self, batch: Dict) -> torch.Tensor:
        z = self.encoder.encode_raw(batch)
        z = z.reshape(z.shape[0], -1)
        return self.head(z)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    metrics_title: str = "Validation metrics",
    print_summary: bool = True,
) -> Tuple[float, float, List[List[int]]]:
    """
    Run inference on validation, aggregate loss / accuracy / confusion matrix
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = [0] * NUM_CLASSES
    per_class_total = [0] * NUM_CLASSES
    cm: List[List[int]] = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = to_device(batch, device)
            y = batch.pop("y").to(device)
            logits = model(batch)
            loss = criterion(logits, y)
            bs = int(y.size(0))
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += bs
            for t, p in zip(y.view(-1), preds.view(-1)):
                t_i = int(t.item())
                p_i = int(p.item())
                if 0 <= t_i < NUM_CLASSES and 0 <= p_i < NUM_CLASSES:
                    cm[t_i][p_i] += 1
                    per_class_total[t_i] += 1
                    if t_i == p_i:
                        per_class_correct[t_i] += 1

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    per_class_acc = {
        CLASS_NAMES[c]: (per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0)
        for c in range(NUM_CLASSES)
    }

    if print_summary:
        print(f"\n{metrics_title}:", flush=True)
        print(f"  Loss: {avg_loss:.4f}", flush=True)
        print(f"  Accuracy: {accuracy:.4f}", flush=True)

        print("\n  Confusion matrix (rows = true, cols = pred):", flush=True)
        header = " " * 10 + "".join(f"{name:>10}" for name in CLASS_NAMES)
        print(header, flush=True)
        for i, row in enumerate(cm):
            row_str = "".join(f"{val:>10d}" for val in row)
            print(f"{CLASS_NAMES[i]:>10}{row_str}", flush=True)

        print("\n  Per-class metrics:", flush=True)
        print(f"{'Class':>10}  {'Acc':>6}", flush=True)
        for name in CLASS_NAMES:
            acc_c = per_class_acc.get(name, 0.0)
            print(f"{name:>10}  {acc_c:6.3f}", flush=True)

    return avg_loss, accuracy, cm


def build_performance_json(
    best_epoch: int,
    val_loss: float,
    val_acc: float,
    cm: List[List[int]],
    *,
    loss_by_epoch: Optional[List[List[Any]]] = None,
) -> Dict[str, Any]:
    """
    Build model_performance.json
    """
    total_count = sum(cm[i][j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES))
    correct_count = sum(cm[c][c] for c in range(NUM_CLASSES))

    per_class: Dict[str, Any] = {}
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        support = sum(cm[c][j] for j in range(NUM_CLASSES))
        correct_c = cm[c][c]
        acc_pct = round(100.0 * (correct_c / support), 2) if support > 0 else 0.0
        per_class[name] = {
            "count": support,
            "correct_count": correct_c,
            "accuracy_pct": acc_pct,
        }

    out: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "cumulative": {
            "total_count": total_count,
            "correct_count": correct_count,
            "accuracy_pct": round(100.0 * val_acc, 2),
            "loss": round(val_loss, 6),
        },
        "per_class": per_class,
        "confusion_matrix_raw": cm,
        "confusion_matrix_labels": list(CLASS_NAMES),
    }
    if loss_by_epoch is not None:
        out["loss_by_epoch"] = loss_by_epoch
    return out


def _checkpoint_payload(model: MeanPoolClassifier, cfg: dict) -> Dict[str, Any]:
    return {
        "encoder_state_dict": model.encoder.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "cfg": cfg,
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "label_map": LABEL_MAP,
        "label_column": LABEL_COLUMN,
    }


def train_classifier(
    model: MeanPoolClassifier,
    tr_load: DataLoader,
    va_load: DataLoader,
    te_load: DataLoader,
    device: torch.device,
    out_dir: pathlib.Path,
    cfg: dict,
    *,
    class_weights: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
) -> pathlib.Path:
    """
    Train with AdamW + weighted CE
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_ckpt_path = out_dir / "classifier_best.pt"
    final_ckpt_path = out_dir / "classifier.pt"
    perf_path = out_dir / "model_performance.json"

    print(
        f"[train] epochs={epochs}  early_stop_patience={early_stopping_patience}  lr={lr}  "
        f"wd={weight_decay}  out_dir={out_dir}  (tqdm on train batches)",
        flush=True,
    )

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    loss_by_epoch: List[List[Any]] = []
    best_metric = (float("inf"), -1.0)
    best_epoch = 0
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        tl, ta = [], []
        pbar = tqdm(
            tr_load,
            desc=f"Epoch {ep}/{epochs} train",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            batch = to_device(batch, device)
            y = batch.pop("y").to(device)
            logits = model(batch)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl.append(loss.item())
            ta.append((logits.argmax(-1) == y).float().mean().item())
            pbar.set_postfix(loss=f"{np.mean(tl):.4f}", acc=f"{np.mean(ta):.4f}")

        val_loss, val_acc, val_cm = evaluate(
            model,
            va_load,
            loss_fn,
            device,
            metrics_title="Validation metrics",
            print_summary=True,
        )
        test_loss, test_acc, _ = evaluate(
            model,
            te_load,
            loss_fn,
            device,
            metrics_title="Test metrics",
            print_summary=False,
        )

        train_loss_ep = _mean(tl)
        loss_by_epoch.append(
            [
                ep,
                float(train_loss_ep) if np.isfinite(train_loss_ep) else None,
                float(val_loss),
            ]
        )

        dt = time.perf_counter() - t_ep
        print(
            f"epoch {ep}/{epochs} done in {dt:.1f}s  "
            f"train loss {_mean(tl):.4f} acc {_mean(ta):.4f}  |  "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}  |  "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}",
            flush=True,
        )

        metric = (val_loss, -val_acc)
        if metric < best_metric:
            epochs_no_improve = 0
            best_metric = metric
            best_epoch = ep
            perf = build_performance_json(
                best_epoch=best_epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                cm=val_cm,
                loss_by_epoch=[list(r) for r in loss_by_epoch],
            )
            perf_path.write_text(json.dumps(perf, indent=2), encoding="utf-8")

            ckpt = _checkpoint_payload(model, cfg)
            ckpt["best_epoch"] = best_epoch
            ckpt["val_loss"] = val_loss
            ckpt["val_acc"] = val_acc
            torch.save(ckpt, best_ckpt_path)
            print(
                f"[best] epoch={best_epoch} val_loss={val_loss:.6f} val_acc={val_acc:.4f} -> "
                f"{perf_path.name} + {best_ckpt_path.name}",
                flush=True,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"[early stop] no val improvement for {early_stopping_patience} epochs "
                    f"(best epoch {best_epoch})",
                    flush=True,
                )
                break

    print("\n[saving] last-epoch checkpoint …", flush=True)
    torch.save(_checkpoint_payload(model, cfg), final_ckpt_path)
    print("[done] wrote", final_ckpt_path.resolve(), flush=True)
    return best_ckpt_path


def run_one(
    use_redshift: bool,
    iter_idx: int,
    *,
    data_dir: pathlib.Path | None = None,
    out_dir: pathlib.Path | None = None,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    split_source: str | None = None,
    train_seed: int | None = None,
):
    print(f"=== run start | USE_REDSHIFT={use_redshift} ITER={iter_idx} ===", flush=True)

    print("=== TwinsClassifier_Wiserep start ===", flush=True)
    print("[init] repo_import_setup", REPO.resolve(), flush=True)
    repo_import_setup(str(REPO))
    seed_used = SEED + iter_idx if train_seed is None else int(train_seed)
    set_seeds(seed_used)
    print("[init] seed", seed_used, flush=True)

    data_dir = (data_dir if data_dir is not None else data_dir_for(use_redshift)).resolve()
    out_dir = (out_dir if out_dir is not None else out_dir_for(use_redshift, iter_idx)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_from_str(DEVICE)
    print("[init] device:", device, flush=True)

    cfg = build_daep_cfg()
    cfg["iau_split_seed"] = IAU_SPLIT_SEED
    cfg["head_dropout"] = float(cfg.get("head_dropout", HEAD_DROPOUT))
    cfg["n_hidden_layers"] = int(cfg.get("n_hidden_layers", 1))
    cfg["train_seed"] = seed_used

    meta, wave, flux, mask = load_preprocessed_for_classifier(data_dir)
    from train_latent import normalize_latent_meta

    meta = normalize_latent_meta(meta)
    if train_idx is None or val_idx is None or test_idx is None:
        tr_idx, va_idx, te_idx = split_row_indices_by_iau_train_val_test(
            meta,
            IAU_COLUMN,
            IAU_TRAIN_FRAC,
            IAU_VAL_FRAC,
            IAU_TEST_FRAC,
            IAU_SPLIT_SEED,
        )
        split_source = (
            f"iau fracs={IAU_TRAIN_FRAC}/{IAU_VAL_FRAC}/{IAU_TEST_FRAC} "
            f"iau_split_seed={IAU_SPLIT_SEED}"
        )
    else:
        tr_idx = np.asarray(train_idx, dtype=np.int64)
        va_idx = np.asarray(val_idx, dtype=np.int64)
        te_idx = np.asarray(test_idx, dtype=np.int64)
        if split_source is None:
            split_source = "provided_indices"
    cfg["split_source"] = split_source
    iau = meta[IAU_COLUMN].astype(str).str.strip()
    n_iau_tr = pd.unique(iau.iloc[tr_idx]).size
    n_iau_va = pd.unique(iau.iloc[va_idx]).size
    n_iau_te = pd.unique(iau.iloc[te_idx]).size
    print(
        f"[split] IAU (unique): train={n_iau_tr}  val={n_iau_va}  test={n_iau_te}  |  "
        f"spectra: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}  "
        f"({split_source} train_rng_seed={seed_used})",
        flush=True,
    )

    tr_idx = filter_indices_mapped(meta, tr_idx, LABEL_COLUMN)
    va_idx = filter_indices_mapped(meta, va_idx, LABEL_COLUMN)
    te_idx = filter_indices_mapped(meta, te_idx, LABEL_COLUMN)
    if min(len(tr_idx), len(va_idx), len(te_idx)) == 0:
        raise RuntimeError(
            f"Empty split after label filter ({split_source}): "
            f"train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}"
        )

    report_split_label_counts(meta, tr_idx, LABEL_COLUMN, "train")
    report_split_label_counts(meta, va_idx, LABEL_COLUMN, "val")
    report_split_label_counts(meta, te_idx, LABEL_COLUMN, "test")

    n_cls = NUM_CLASSES
    train_ds = LabeledSpectra(meta, wave, flux, mask, tr_idx, LABEL_COLUMN)
    val_ds = LabeledSpectra(meta, wave, flux, mask, va_idx, LABEL_COLUMN)
    test_ds = LabeledSpectra(meta, wave, flux, mask, te_idx, LABEL_COLUMN)

    tr_load = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_labeled, num_workers=0)
    va_load = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_labeled, num_workers=0)
    te_load = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_labeled, num_workers=0)

    print("[model] building encoder + head ", flush=True)
    embed_dim = int(cfg["bottleneck_dim"]) * int(cfg["bottleneck_length"])
    model = MeanPoolClassifier(
        build_daep(cfg).to(device).encoder,
        embed_dim,
        n_cls,
        head_hidden=int(cfg["ff_dim"]),
        head_dropout=float(cfg["head_dropout"]),
        n_hidden_layers=int(cfg["n_hidden_layers"]),
    ).to(device)
    print(
        f"[model] embed_dim={embed_dim}  hidden={int(cfg['ff_dim'])}  "
        f"n_hidden_layers={int(cfg['n_hidden_layers'])}  "
        f"encoder_dropout={cfg.get('dropout')}  head_dropout={cfg['head_dropout']}",
        flush=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] parameters total={n_params:,}  trainable={n_trainable:,}", flush=True)

    ce_weights = inverse_frequency_class_weights(meta, tr_idx, LABEL_COLUMN).to(device)

    best_path = train_classifier(
        model,
        tr_load,
        va_load,
        te_load,
        device,
        out_dir,
        cfg,
        class_weights=ce_weights,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )
    (out_dir / "cfg_used.json").write_text(json.dumps(cfg, indent=2))
    print("[done] best checkpoint:", best_path.resolve(), flush=True)
    print("[done] wrote", (out_dir / "cfg_used.json").resolve(), flush=True)
    print(f"=== run finished | USE_REDSHIFT={use_redshift} ITER={iter_idx} ===", flush=True)

def use_redshift_for_assignment_dir(latent_dir: pathlib.Path) -> bool:
    name = latent_dir.name.lower()
    parent = latent_dir.parent.name.lower()
    if name.startswith("nodered") or parent.endswith("_noz"):
        return False
    if name.startswith("dered"):
        return True
    raise ValueError(
        f"Cannot tell redshift variant from {latent_dir}. "
        "Use a Dered* or Nodered* assignment directory."
    )


def map_assignment_to_flux_rows(
    latent_dir: pathlib.Path,
    flux_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Assignment indices are rows of meta_universal.csv. Map them onto flux rows.

    The deredshifted bundle matches meta_universal row for row. The no-redshift
    bundle has a few extra spectra, so those indices are matched on spec_id.
    """
    from train_latent import (
        load_assignment_indices_from_dir,
        normalize_latent_meta,
        resolve_latent_meta_csv,
    )

    universal = normalize_latent_meta(
        pd.read_csv(resolve_latent_meta_csv(latent_dir), low_memory=False)
    )
    train_idx, val_idx, test_idx, split_source = load_assignment_indices_from_dir(
        latent_dir,
        universal,
    )
    flux = normalize_latent_meta(flux_meta)
    if "spec_id" not in flux.columns or "spec_id" not in universal.columns:
        if len(flux) != len(universal):
            raise ValueError(
                f"Flux meta has {len(flux)} rows but {latent_dir} assignment "
                f"has {len(universal)}, and neither table has spec_id"
            )
        return train_idx, val_idx, test_idx, split_source

    flux_ids = flux["spec_id"].astype(str).str.strip()
    uni_ids = universal["spec_id"].astype(str).str.strip()
    if len(flux_ids) == len(uni_ids) and flux_ids.equals(uni_ids):
        return train_idx, val_idx, test_idx, split_source
    if bool(flux_ids.duplicated().any()) or bool(uni_ids.duplicated().any()):
        raise ValueError(f"Duplicate spec_id while aligning {latent_dir} to the flux meta")

    pos = {sid: i for i, sid in enumerate(flux_ids)}
    missing = [sid for sid in uni_ids if sid not in pos]
    if missing:
        raise ValueError(
            f"{len(missing)} spectra in {latent_dir} are missing from the flux meta"
        )
    mapper = np.fromiter((pos[sid] for sid in uni_ids), dtype=np.int64, count=len(uni_ids))
    print(
        f"[split] mapped {len(uni_ids)} assignment rows onto {len(flux_ids)} flux rows "
        f"({len(flux_ids) - len(uni_ids)} flux rows unused)",
        flush=True,
    )
    return mapper[train_idx], mapper[val_idx], mapper[test_idx], split_source


def assignment_dir_for_init_run(run_dir: pathlib.Path) -> pathlib.Path:
    """``daep_init/try_5/split_36`` -> ``wiserep_henna/try_5/Dered36_5``."""
    run_dir = pathlib.Path(run_dir)
    match = re.fullmatch(r"split_(\d+)", run_dir.name)
    if match is None:
        raise ValueError(f"Expected a split_<seed> directory, got {run_dir}")
    seed = match.group(1)
    try_name = run_dir.parent.name
    noz = try_name.endswith("_noz")
    prefix = "Nodered" if noz else "Dered"
    ver = try_name[: -len("_noz")] if noz else try_name
    if not ver.startswith("try_"):
        raise ValueError(f"Cannot infer the try number from {try_name}")
    try_num = ver.split("_", 1)[1]
    henna_try = HENNA_ROOT / try_name
    candidate = henna_try / f"{prefix}{seed}_{try_num}"
    if candidate.is_dir():
        return candidate
    matches = sorted(henna_try.glob(f"{prefix}{seed}_*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"No assignment directory for {run_dir} (looked for {candidate})")


def flux_test_indices_for_init_run(
    run_dir: pathlib.Path,
    flux_meta: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    """Test-row indices into the flux table for one ``split_<seed>`` init run."""
    from train_latent import normalize_latent_meta

    latent_dir = assignment_dir_for_init_run(run_dir)
    _, _, test_idx, split_source = map_assignment_to_flux_rows(latent_dir, flux_meta)
    meta = normalize_latent_meta(flux_meta)
    return filter_indices_mapped(meta, test_idx, LABEL_COLUMN), split_source


def run_assignment_dirs(
    latent_dirs: List[pathlib.Path],
    *,
    out_root: pathlib.Path,
    train_seed: int,
) -> None:
    from train_latent import split_run_name, try_run_name

    out_root = out_root.resolve()
    for latent_dir in latent_dirs:
        latent_dir = latent_dir.expanduser().resolve()
        use_redshift = use_redshift_for_assignment_dir(latent_dir)
        data_dir = data_dir_for(use_redshift)
        meta_path = data_dir / "meta.csv"
        if not meta_path.is_file():
            meta_path = data_dir / "wiserep_metadata_processed.csv"
        flux_meta = pd.read_csv(meta_path, low_memory=False)
        train_idx, val_idx, test_idx, split_source = map_assignment_to_flux_rows(
            latent_dir,
            flux_meta,
        )
        out_dir = out_root / try_run_name(latent_dir) / split_run_name(latent_dir)
        print(
            f"=== assignment job | dir={latent_dir} out={out_dir} "
            f"redshift={use_redshift} train_seed={train_seed} ===",
            flush=True,
        )
        run_one(
            use_redshift,
            0,
            data_dir=data_dir,
            out_dir=out_dir,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            split_source=split_source,
            train_seed=train_seed,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the no-diffusion DAEP classifier. "
            "Encoder and MLP match the diffusion latent models."
        )
    )
    parser.add_argument(
        "--latent-dirs",
        nargs="+",
        type=pathlib.Path,
        default=None,
        help=(
            "Assignment directories (each with split_assignment*.json), "
            "e.g. data/wiserep_henna/try_5/Dered36_5. "
            "One model per directory, on that directory's train/val/test rows. "
            "Flux is deredshifted/ or noderedshift/."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=pathlib.Path,
        default=DEFAULT_INIT_OUT_ROOT,
        help=(
            f"Parent output directory for --latent-dirs "
            f"(default: {DEFAULT_INIT_OUT_ROOT}). "
            "Each job writes <out-root>/<try_N>/split_<seed>/."
        ),
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=SEED,
        help=f"Training RNG seed for --latent-dirs (default: {SEED}).",
    )
    args = parser.parse_args()

    if args.latent_dirs is not None:
        run_assignment_dirs(
            args.latent_dirs,
            out_root=args.out_root,
            train_seed=args.train_seed,
        )
        return

    for use_redshift in USE_REDSHIFT_VALUES:
        for iter_idx in ITER_VALUES:
            out = out_dir_for(use_redshift, iter_idx)
            if (out / "classifier_best.pt").is_file():
                print(f"[skip] {out} already has classifier_best.pt", flush=True)
                continue
            run_one(use_redshift, iter_idx)

if __name__ == "__main__":
    main()
