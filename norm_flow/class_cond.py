"""5-class features for class-conditioned redshift models.

Latent: concat the 5-d vector onto the flattened embedding (MLP input).
DASH: concat the 5-d vector after the CNN trunk (head input).

``--cls-cond`` uses frozen classifier softmax. ``--cls-true`` uses the mapped
true-label one-hot (oracle; label leakage by design). Unmapped rows/files are
dropped in both modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from paths import PROJECT_ROOT

N_CLASSES = 5
_LATENT_CKPT_NAMES = ("classifier_best.pt", "classifier.pt")
_DASH_CKPT_NAMES = ("model.pth",)
_CFG_NAMES = ("cfg_used.json",)


def default_latent_cls_dir(latent_dir: Path) -> Path:
    from train_latent import split_run_name, try_run_name

    latent_dir = Path(latent_dir).resolve()
    return (
        PROJECT_ROOT
        / "data"
        / "pre_trained_models"
        / "daep_latent"
        / try_run_name(latent_dir)
        / split_run_name(latent_dir)
    )


def default_dash_cls_dir(split_seed: Any) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "pre_trained_models"
        / "henna_matched_split_noz"
        / f"split_{split_seed}"
    )


def assert_observed_frame_cls_dir(path: Path, *, kind: str) -> None:
    text = str(Path(path).resolve()).lower()
    if "noz" not in text and "nodered" not in text:
        raise RuntimeError(
            f"{kind} classifier must be observed-frame / no-z (path should contain 'noz'). "
            f"Got: {path}"
        )


def _first_file(directory: Path, names: Sequence[str]) -> Path:
    directory = Path(directory)
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    if directory.is_dir():
        lower = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
        for name in names:
            hit = lower.get(name.lower())
            if hit is not None:
                return hit
    raise FileNotFoundError(f"No checkpoint among {list(names)} in {directory}")


def _load_blob(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _state_dict(blob: Any) -> Dict[str, torch.Tensor]:
    if isinstance(blob, dict):
        for key in ("model_state_dict", "model_state_dict", "state_dict", "model"):
            val = blob.get(key)
            if isinstance(val, dict) and val and all(torch.is_tensor(v) for v in val.values()):
                return val
        if blob and all(torch.is_tensor(v) for v in blob.values()):
            return blob
    raise RuntimeError(f"Unrecognized classifier checkpoint payload: {type(blob)}")


def load_latent_classifier(
    ckpt_dir: Path,
    embed_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    from train_latent import LatentClassifier, NUM_CLASSES

    ckpt_dir = Path(ckpt_dir).resolve()
    assert_observed_frame_cls_dir(ckpt_dir, kind="latent")
    cfg: Dict[str, Any] = {}
    for name in _CFG_NAMES:
        cfg_path = ckpt_dir / name
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            break
    hidden = int(cfg.get("ff_dim") or 256)
    dropout = float(cfg.get("dropout") or 0.25)
    n_cls = int(cfg.get("n_classes") or NUM_CLASSES)
    if n_cls != N_CLASSES:
        raise RuntimeError(f"Expected {N_CLASSES}-class latent classifier, got n_classes={n_cls}")
    model = LatentClassifier(
        embed_dim=int(embed_dim),
        n_cls=n_cls,
        head_hidden=hidden,
        head_dropout=dropout,
    )
    blob = _load_blob(_first_file(ckpt_dir, _LATENT_CKPT_NAMES), device)
    if isinstance(blob, dict) and isinstance(blob.get("cfg"), dict) and not cfg:
        cfg = blob["cfg"]
        hidden = int(cfg.get("ff_dim") or hidden)
        dropout = float(cfg.get("dropout") or dropout)
        model = LatentClassifier(
            embed_dim=int(embed_dim),
            n_cls=n_cls,
            head_hidden=hidden,
            head_dropout=dropout,
        )
    model.load_state_dict(_state_dict(blob))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(
        f"[cls] latent classifier {ckpt_dir}  hidden={hidden} dropout={dropout} n_cls={n_cls}",
        flush=True,
    )
    return model


@torch.no_grad()
def latent_class_probs(
    model: torch.nn.Module,
    z: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    idx = np.asarray(indices, dtype=np.int64)
    out = torch.zeros(len(idx), N_CLASSES, dtype=torch.float32)
    model.eval()
    for start in range(0, len(idx), int(batch_size)):
        sl = idx[start : start + int(batch_size)]
        zb = torch.from_numpy(np.asarray(z[sl], dtype=np.float32)).to(device)
        logits = model({"z": zb})
        out[start : start + len(sl)] = F.softmax(logits, dim=-1).cpu()
    return out


def load_dash_classifier(ckpt_dir: Path, device: torch.device) -> torch.nn.Module:
    import constants as const
    from dash_retrain import DashCNN1D

    ckpt_dir = Path(ckpt_dir).resolve()
    assert_observed_frame_cls_dir(ckpt_dir, kind="DASH")
    n_cls = int(const.NUM_CLASSES)
    if n_cls != N_CLASSES:
        raise RuntimeError(f"Expected {N_CLASSES}-class DASH classifier, got NUM_CLASSES={n_cls}")
    model = DashCNN1D(input_length=const.TARGET_LENGTH, num_classes=n_cls)
    blob = _load_blob(_first_file(ckpt_dir, _DASH_CKPT_NAMES), device)
    model.load_state_dict(_state_dict(blob))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[cls] DASH classifier {ckpt_dir}  n_cls={n_cls}", flush=True)
    return model


@torch.no_grad()
def dash_class_probs_by_id(
    model: torch.nn.Module,
    inner_ds: Any,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    model.eval()
    probs: Dict[str, torch.Tensor] = {}
    xs: List[torch.Tensor] = []
    ids: List[str] = []

    def _flush() -> None:
        if not xs:
            return
        xb = torch.stack(xs, dim=0).to(device)
        p = F.softmax(model(xb), dim=-1).cpu()
        for i, fid in enumerate(ids):
            probs[fid] = p[i].clone()
        xs.clear()
        ids.clear()

    for i in range(len(inner_ds)):
        item = inner_ds[i]
        if item is None:
            continue
        x, _z = item
        fname = inner_ds.samples[i][0]
        xs.append(x.float())
        ids.append(str(fname))
        if len(xs) >= int(batch_size):
            _flush()
    _flush()
    print(f"[cls] DASH softmax cached for {len(probs)} spectra", flush=True)
    return probs


def _latent_label_col(meta: Any) -> str:
    from train_latent import LABEL_COLUMN

    if LABEL_COLUMN in meta.columns:
        return LABEL_COLUMN
    for alt in ("type", "Type", "Obj. Type"):
        if alt in meta.columns:
            return alt
    raise KeyError(
        f"Need a 5-class label column for class-conditioning; have {list(meta.columns)[:20]}"
    )


def filter_mapped_indices(meta: Any, indices: np.ndarray) -> np.ndarray:
    from train_latent import filter_indices_mapped

    col = _latent_label_col(meta)
    before = int(len(indices))
    kept = filter_indices_mapped(meta, indices, col)
    dropped = before - int(len(kept))
    if dropped:
        print(f"[cls] dropped {dropped} / {before} rows without a 5-class label ({col})", flush=True)
    return kept


def latent_class_onehot(meta: Any, indices: np.ndarray) -> torch.Tensor:
    """True 5-class one-hot in meta-row order of ``indices`` (already mapped)."""
    from TwinsClassifier_Wiserep import row_class_idx

    col = _latent_label_col(meta)
    idx = np.asarray(indices, dtype=np.int64)
    cls = np.array([row_class_idx(meta, int(i), col) for i in idx], dtype=np.int64)
    if (cls < 0).any():
        n_bad = int((cls < 0).sum())
        raise RuntimeError(f"{n_bad} rows still unmapped after filter; refuse to one-hot")
    out = torch.zeros(len(idx), N_CLASSES, dtype=torch.float32)
    out[torch.arange(len(idx)), torch.from_numpy(cls)] = 1.0
    return out


def dash_class_onehot_by_id(
    files: Iterable[str],
    metadata: Mapping[str, Mapping[str, str]],
) -> Dict[str, torch.Tensor]:
    """True 5-class one-hot keyed by ascii filename (already mapped)."""
    import constants as const
    import helpers

    out: Dict[str, torch.Tensor] = {}
    for fname in files:
        meta = metadata.get(fname) or {}
        raw = str(meta.get("type") or meta.get("Type") or "")
        name = helpers.normalize_label(raw)
        if name is None or name not in const.CLASS_TO_IDX:
            raise RuntimeError(f"unmapped ascii file after filter: {fname} type={raw!r}")
        vec = torch.zeros(N_CLASSES, dtype=torch.float32)
        vec[int(const.CLASS_TO_IDX[name])] = 1.0
        out[str(fname)] = vec
    return out


def filter_mapped_files(
    files: Iterable[str],
    metadata: Mapping[str, Mapping[str, str]],
) -> List[str]:
    import helpers

    keep: List[str] = []
    dropped = 0
    for fname in files:
        meta = metadata.get(fname) or {}
        raw = str(meta.get("type") or meta.get("Type") or "")
        if helpers.normalize_label(raw) is None:
            dropped += 1
            continue
        keep.append(fname)
    if dropped:
        print(f"[cls] dropped {dropped} ascii files without a 5-class label", flush=True)
    return keep
