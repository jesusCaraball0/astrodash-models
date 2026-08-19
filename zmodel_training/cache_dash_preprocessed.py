#!/usr/bin/env python3
"""
Precompute DASH 1024-bin spectra for ASCII under data/wiserep/wiserep_data_noSEDM.

This is the Dash 1D CNN preprocessor (helpers.preprocess_spectrum), not Henna
``noderedshift/flux.npy`` / ``deredshifted/flux.npy``.

  noz  — process_no_redshift (observed frame). Spectroscopic z is stored but not
         used in the vector; trainers append 0.0 as the extra input feature.
  z    — process(..., z) with metadata redshift (same as dash_retrain.py).

Writes mmap-friendly arrays under data/wiserep/dash_preprocessed/{noz,z}/.

Usage:
  python zmodel_training/cache_dash_preprocessed.py
  python zmodel_training/cache_dash_preprocessed.py --variant noz
  python zmodel_training/cache_dash_preprocessed.py --variant z --workers 8
  python zmodel_training/cache_dash_preprocessed.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

import constants as const
import helpers as helpers

for _name in ("spectrum_io", "dash_preprocess"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_ROOT_DEFAULT = const.WISEREP_DIR / "dash_preprocessed"
VARIANT_NOZ = "noz"
VARIANT_Z = "z"
VARIANTS_ALL = (VARIANT_NOZ, VARIANT_Z)

_WORKER_SPECTRA_DIR = ""
_WORKER_VARIANTS: Tuple[str, ...] = ()


def cache_dir_for(variant: str, out_root: Path) -> Path:
    if variant not in VARIANTS_ALL:
        raise ValueError(f"unknown variant {variant!r}")
    return out_root / variant


def parse_metadata_redshift(raw: object) -> float:
    """Same default as dash_retrain.WISeREPDataset: 0.0 on missing/invalid."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def list_ascii_jobs(
    metadata: Dict[str, Dict[str, str]],
    spectra_dir: Path,
) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
    jobs: List[Tuple[str, float]] = []
    n_missing = 0
    for fname, meta in metadata.items():
        path = spectra_dir / fname
        if not path.is_file():
            n_missing += 1
            continue
        jobs.append((fname, parse_metadata_redshift(meta.get("redshift"))))
    return jobs, {"n_metadata": len(metadata), "n_missing_file": n_missing, "n_jobs": len(jobs)}


def _init_worker(spectra_dir: str, variants: Tuple[str, ...]) -> None:
    global _WORKER_SPECTRA_DIR, _WORKER_VARIANTS
    _WORKER_SPECTRA_DIR = spectra_dir
    _WORKER_VARIANTS = variants
    for name in ("spectrum_io", "dash_preprocess"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _process_one(job: Tuple[str, float]) -> Dict[str, Any]:
    fname, z_val = job
    result = helpers.load_spectrum(Path(_WORKER_SPECTRA_DIR) / fname)
    if result is None:
        return {"name": fname, "redshift": float(z_val), "error": "load"}
    wave, flux = result
    out: Dict[str, Any] = {"name": fname, "redshift": float(z_val), "error": None}
    failed: List[str] = []
    if VARIANT_NOZ in _WORKER_VARIANTS:
        vec = helpers.preprocess_spectrum(wave, flux, None, const.TARGET_LENGTH)
        if vec is None:
            failed.append(VARIANT_NOZ)
        else:
            out[VARIANT_NOZ] = np.asarray(vec, dtype=np.float32)
    if VARIANT_Z in _WORKER_VARIANTS:
        vec = helpers.preprocess_spectrum(wave, flux, float(z_val), const.TARGET_LENGTH)
        if vec is None:
            failed.append(VARIANT_Z)
        else:
            out[VARIANT_Z] = np.asarray(vec, dtype=np.float32)
    if failed:
        out["error"] = "preprocess:" + ",".join(failed)
    return out


def _run_jobs(
    jobs: Sequence[Tuple[str, float]],
    *,
    spectra_dir: Path,
    variants: Sequence[str],
    workers: int,
) -> List[Dict[str, Any]]:
    variant_t = tuple(variants)
    if workers <= 1:
        _init_worker(str(spectra_dir), variant_t)
        return [_process_one(job) for job in tqdm(jobs, desc="preprocess", dynamic_ncols=True)]
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(spectra_dir), variant_t),
    ) as pool:
        for row in tqdm(
            pool.map(_process_one, jobs, chunksize=16),
            total=len(jobs),
            desc=f"preprocess x{workers}",
            dynamic_ncols=True,
        ):
            rows.append(row)
    return rows


def _refuse_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite {path}\n"
            "Pass --overwrite, or choose a different --out-root."
        )


def _write_variant(
    out_dir: Path,
    *,
    variant: str,
    names: List[str],
    flux: np.ndarray,
    redshift: np.ndarray,
    failed: List[Dict[str, str]],
    counts: Dict[str, int],
    spectra_dir: Path,
    overwrite: bool,
) -> None:
    _refuse_existing(out_dir / "flux.npy", overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    # np.save appends .npy unless the path already ends with it.
    tmp_flux = out_dir / "flux.tmp.npy"
    tmp_z = out_dir / "redshift.tmp.npy"
    np.save(tmp_flux, flux)
    np.save(tmp_z, redshift)
    tmp_flux.replace(out_dir / "flux.npy")
    tmp_z.replace(out_dir / "redshift.npy")
    (out_dir / "names.json").write_text(json.dumps(names, indent=0), encoding="utf-8")
    (out_dir / "failed.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in failed),
        encoding="utf-8",
    )
    manifest = {
        "variant": variant,
        "has_redshift": variant == VARIANT_Z,
        "n_ok": int(flux.shape[0]),
        "n_failed": len(failed),
        "flux_shape": [int(flux.shape[0]), int(flux.shape[1])] if flux.ndim == 2 else list(flux.shape),
        "nw": int(const.NW),
        "wave_min": float(const.WAVE_MIN),
        "wave_max": float(const.WAVE_MAX),
        "spectra_dir": str(spectra_dir.resolve()),
        "metadata_csv": str(const.METADATA_CSV.resolve()),
        "note": (
            "flux is DASH nw bins only; append redshift (z cache) or 0.0 (noz cache) "
            "to match dash_retrain / dash_retrain_redshift model input."
        ),
        **counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"wrote {out_dir}: n_ok={manifest['n_ok']} n_failed={manifest['n_failed']} "
        f"flux={flux.shape} ({flux.nbytes / 1e6:.1f} MB)",
        flush=True,
    )


def load_dash_preprocessed_cache(cache_dir: Path) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    """Load a cache dir. Returns (filename -> row, flux mmap (N, nw), redshift (N,))."""
    cache_dir = cache_dir.resolve()
    names = json.loads((cache_dir / "names.json").read_text(encoding="utf-8"))
    index = {str(name): i for i, name in enumerate(names)}
    flux = np.load(cache_dir / "flux.npy", mmap_mode="r")
    redshift = np.load(cache_dir / "redshift.npy", mmap_mode="r")
    if len(index) != int(flux.shape[0]) or len(index) != int(redshift.shape[0]):
        raise ValueError(f"Cache length mismatch under {cache_dir}")
    return index, flux, redshift


class DashFluxCache:
    """Filename lookup into a dash_preprocessed/{noz,z} directory."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir.resolve()
        self.index, self.flux, self.redshift = load_dash_preprocessed_cache(self.cache_dir)

    def __contains__(self, fname: object) -> bool:
        return str(fname) in self.index

    def __len__(self) -> int:
        return len(self.index)

    def flux_row(self, fname: str) -> np.ndarray:
        return np.asarray(self.flux[self.index[str(fname)]], dtype=np.float32)


def cache_dir_is_ready(cache_dir: Path) -> bool:
    cache_dir = cache_dir.expanduser()
    return (cache_dir / "flux.npy").is_file() and (cache_dir / "names.json").is_file()


def resolve_flux_cache(
    has_redshift: bool,
    *,
    cache_dir: Path | None = None,
    disable: bool = False,
) -> DashFluxCache | None:
    """Load the default ±z cache if present. Explicit --cache-dir must exist."""
    if disable:
        return None
    variant = VARIANT_Z if has_redshift else VARIANT_NOZ
    path = (
        cache_dir.expanduser().resolve()
        if cache_dir is not None
        else cache_dir_for(variant, OUT_ROOT_DEFAULT)
    )
    if not cache_dir_is_ready(path):
        if cache_dir is not None:
            raise FileNotFoundError(f"DASH cache not found under {path}")
        return None
    cache = DashFluxCache(path)
    print(f"[cache] {variant} {cache.cache_dir}  n={len(cache)}", flush=True)
    return cache


def model_input_from_cache(
    flux_row: np.ndarray,
    redshift: float,
    *,
    has_redshift: bool,
) -> np.ndarray:
    """1024 DASH bins + extra feature, same concat as dash_retrain.WISeREPDataset."""
    z_feat = float(redshift) if has_redshift else 0.0
    return np.concatenate([np.asarray(flux_row, dtype=np.float32), [z_feat]]).astype(np.float32)


def collect_variant_arrays(
    rows: Iterable[Dict[str, Any]],
    variant: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[Dict[str, str]]]:
    names: List[str] = []
    flux_rows: List[np.ndarray] = []
    z_rows: List[float] = []
    failed: List[Dict[str, str]] = []
    for row in rows:
        vec = row.get(variant)
        if vec is None:
            failed.append(
                {
                    "name": str(row["name"]),
                    "error": str(row.get("error") or f"missing:{variant}"),
                }
            )
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size != const.NW:
            failed.append(
                {
                    "name": str(row["name"]),
                    "error": f"bad_nw:{arr.size}",
                }
            )
            continue
        names.append(str(row["name"]))
        flux_rows.append(arr)
        z_rows.append(float(row["redshift"]))
    if flux_rows:
        flux = np.stack(flux_rows, axis=0)
        redshift = np.asarray(z_rows, dtype=np.float32)
    else:
        flux = np.zeros((0, const.NW), dtype=np.float32)
        redshift = np.zeros((0,), dtype=np.float32)
    return names, flux, redshift, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache DASH-preprocessed ASCII spectra (noz and/or z)."
    )
    parser.add_argument(
        "--variant",
        choices=("both", VARIANT_NOZ, VARIANT_Z),
        default="both",
        help="Which preprocessor to run (default: both, one ASCII load per file).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT_DEFAULT,
        help=f"Output parent (default: {OUT_ROOT_DEFAULT}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Process workers (default: min(8, CPU count)). Use 1 to disable pooling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing flux.npy / names.json in the variant dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on files (debug).",
    )
    args = parser.parse_args()
    variants = list(VARIANTS_ALL) if args.variant == "both" else [str(args.variant)]
    out_root = args.out_root.expanduser().resolve()
    workers = max(1, int(args.workers))
    spectra_dir = const.SPECTRA_DIR

    if not const.METADATA_CSV.is_file():
        raise SystemExit(f"Missing metadata: {const.METADATA_CSV}")
    if not spectra_dir.is_dir():
        raise SystemExit(f"Missing spectra dir: {spectra_dir}")

    for variant in variants:
        _refuse_existing(cache_dir_for(variant, out_root) / "flux.npy", bool(args.overwrite))

    metadata = helpers.load_metadata(const.METADATA_CSV)
    jobs, counts = list_ascii_jobs(metadata, spectra_dir)
    if args.limit is not None:
        jobs = jobs[: max(0, int(args.limit))]
        counts["n_jobs_limited"] = len(jobs)
    print(
        f"metadata={counts['n_metadata']}  missing_file={counts['n_missing_file']}  "
        f"jobs={len(jobs)}  variants={variants}  workers={workers}",
        flush=True,
    )
    if not jobs:
        raise SystemExit("No ascii files to preprocess.")

    rows = _run_jobs(jobs, spectra_dir=spectra_dir, variants=variants, workers=workers)
    for variant in variants:
        names, flux, redshift, failed = collect_variant_arrays(rows, variant)
        _write_variant(
            cache_dir_for(variant, out_root),
            variant=variant,
            names=names,
            flux=flux,
            redshift=redshift,
            failed=failed,
            counts=counts,
            spectra_dir=spectra_dir,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
