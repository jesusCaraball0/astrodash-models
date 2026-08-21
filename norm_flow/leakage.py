"""Observed-frame / no-z and split-disjointness checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

_NODERED_DIR = re.compile(r"^nodered\d+", re.IGNORECASE)
_DERED_DIR = re.compile(r"^dered\d+", re.IGNORECASE)
_TRY_NOZ = re.compile(r"^try_\d+_noz$", re.IGNORECASE)
_TRY_Z = re.compile(r"^try_\d+$", re.IGNORECASE)


def _parts(path: Path) -> list[str]:
    return [p.lower() for p in path.resolve().parts]


def is_observed_frame_latent_dir(path: Path) -> bool:
    """True for ``try_*_noz``, ``noderedshift``, or ``Nodered{{seed}}_*`` dirs."""
    path = Path(path)
    parts = _parts(path)
    name = path.name
    if any(p == "noderedshift" or p.endswith("_noz") or _TRY_NOZ.match(p) for p in parts):
        return True
    if _NODERED_DIR.match(name):
        return True
    return False


def is_deredshifted_latent_dir(path: Path) -> bool:
    path = Path(path)
    parts = _parts(path)
    if "deredshifted" in parts:
        return True
    if _DERED_DIR.match(path.name):
        return True
    if any(_TRY_Z.match(p) for p in parts) and not any(_TRY_NOZ.match(p) for p in parts):
        if _DERED_DIR.match(path.name) or "deredshifted" in parts:
            return True
    return False


def assert_observed_frame_latent_dir(path: Path) -> None:
    path = Path(path).resolve()
    if is_deredshifted_latent_dir(path) or not is_observed_frame_latent_dir(path):
        raise RuntimeError(
            "Redshift models must use observed-frame / no-z latents "
            f"(try_*_noz or Nodered*), not deredshifted embeddings. Got: {path}"
        )


def assert_observed_frame_splits(splits_json: Path) -> None:
    text = str(Path(splits_json).resolve()).lower()
    if "noz" not in text and "nodered" not in text:
        raise RuntimeError(
            "DASH redshift splits must be observed-frame / no-z "
            f"(filename should contain 'noz'). Got: {splits_json}"
        )


def assert_disjoint_ids(
    parts: Mapping[str, Iterable[str]],
    *,
    kind: str,
) -> None:
    sets = {k: set(str(x).strip() for x in v if str(x).strip()) for k, v in parts.items()}
    leaks = {
        f"{a}&{b}": sorted(sets[a] & sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        if a in sets and b in sets and sets[a] & sets[b]
    }
    if leaks:
        preview = {
            k: v[:8] + ([f"...({len(v)} total)"] if len(v) > 8 else [])
            for k, v in leaks.items()
        }
        raise RuntimeError(f"{kind} leakage across splits: {preview}")


def assert_iau_indices_disjoint(
    meta: pd.DataFrame,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    iau_col: str = "IAU name",
) -> None:
    if iau_col not in meta.columns:
        raise KeyError(f"Need {iau_col!r} for leakage check; have {list(meta.columns)[:20]}")
    iau = meta[iau_col].astype(str).str.strip()
    assert_disjoint_ids(
        {
            "train": iau.iloc[list(train_idx)].tolist(),
            "val": iau.iloc[list(val_idx)].tolist(),
            "test": iau.iloc[list(test_idx)].tolist(),
        },
        kind="IAU name",
    )


def document_z_handling(*, z_floor: float, z_max: float, target: str = "logz") -> str:
    if target == "z":
        y_msg = (
            f"Training target is physical z (clamped at {z_floor} for numerics only)."
        )
    else:
        y_msg = (
            f"Training target is y=ln(max(z, {z_floor})); the floor only affects "
            f"(0, {z_floor}] and matches train_latent_redshift / dash_retrain_redshift."
        )
    return (
        f"Rows with non-finite z or z<=0 or z>{z_max} are dropped (not clipped). {y_msg}"
    )
