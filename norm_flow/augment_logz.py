"""Observed-frame redshift augmentation for frozen Universal / DAEP latents.

Stretch native flux on the Henna noderedshift grid

    λ_src = λ * (1 + z_old) / (1 + z_new)

with z_new from noise on ln(1+z), encode with the seed's ``best_ckpt.pt``, and
cache the new train embeddings. Val / test are never augmented.

  ln(1 + z_new) = ln(1 + z_old) + ε,  ε ~ Unif(eps_lo, eps_hi)
  z_new = (1 + z_old) exp(ε) - 1

ε is drawn on the truncation of that interval where z_new stays in [z_floor, z_max]
(equivalent to rejection sampling from the original uniform).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from paths import PROJECT_ROOT, WISEREP_DIR
from train_utils import Z_FLOOR, Z_MAX

HENNA_NOZ = PROJECT_ROOT / "data" / "wiserep_henna" / "noderedshift"
DAEP_REPO = WISEREP_DIR / "Perceiver-diffusion-autoencoder"
DEFAULT_CKPT_NAME = "best_ckpt.pt"

AUG_K = 5
AUG_EPS_LO = -0.08
AUG_EPS_HI = 0.1
AUG_ENCODE_BATCH = 32
AUG_CACHE_NAME = "latent_aug.npz"
MIN_VALID_BINS = 2


class TensorLatentDataset(Dataset):
    """Flattened embeddings + physical-z labels (augmented train rows)."""

    def __init__(self, embeddings: np.ndarray, z_phys: np.ndarray, ids: list[str]):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.z_phys = np.asarray(z_phys, dtype=np.float32).reshape(-1)
        self.ids = list(ids)
        if self.embeddings.shape[0] != self.z_phys.shape[0] or self.embeddings.shape[0] != len(self.ids):
            raise ValueError(
                f"aug dataset size mismatch: z={self.embeddings.shape} "
                f"y={self.z_phys.shape} ids={len(self.ids)}"
            )

    def __len__(self) -> int:
        return int(self.z_phys.shape[0])

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return {
            "x": torch.from_numpy(np.ascontiguousarray(self.embeddings[i].reshape(-1))),
            "y": torch.tensor(float(self.z_phys[i]), dtype=torch.float32),
            "id": self.ids[i],
        }


def cache_dir_for(latent_dir: Path, k: int) -> Path:
    return Path(latent_dir) / f"aug_ln1pz_k{int(k)}"


def cache_path_for(latent_dir: Path, k: int) -> Path:
    return cache_dir_for(latent_dir, k) / AUG_CACHE_NAME


def _npz_temp_path(out_path: Path) -> Path:
    """Temp path that already ends in .npz so numpy will not append another suffix."""
    return out_path.with_name(out_path.stem + ".tmp.npz")


def _recover_partial_cache(out_path: Path) -> None:
    """Move a leftover numpy-mangled temp onto ``out_path`` if the real cache is missing."""
    if out_path.is_file():
        return
    candidates = [
        out_path.with_suffix(".npz.tmp.npz"),  # latent_aug.npz.tmp.npz (old bug)
        _npz_temp_path(out_path),
    ]
    for src in candidates:
        if src.is_file() and src.resolve() != out_path.resolve():
            src.replace(out_path)
            print(f"[aug] recovered {src.name} -> {out_path.name}", flush=True)
            return


def eps_valid_bounds(
    z_old: np.ndarray,
    *,
    eps_lo: float,
    eps_hi: float,
    z_floor: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-spectrum [lo, hi] for ε so z_new stays in [z_floor, z_max]."""
    z_old = np.asarray(z_old, dtype=np.float64).reshape(-1)
    one = 1.0 + z_old
    lo = np.maximum(float(eps_lo), np.log((1.0 + float(z_floor)) / one))
    hi = np.minimum(float(eps_hi), np.log((1.0 + float(z_max)) / one))
    return lo, hi


def sample_z_new(
    z_old: np.ndarray,
    rng: np.random.Generator,
    *,
    k: int,
    eps_lo: float = AUG_EPS_LO,
    eps_hi: float = AUG_EPS_HI,
    z_floor: float = Z_FLOOR,
    z_max: float = Z_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``z_new`` and ``eps`` with shape (n, k)."""
    z_old = np.asarray(z_old, dtype=np.float64).reshape(-1)
    n = int(z_old.size)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    lo, hi = eps_valid_bounds(
        z_old, eps_lo=eps_lo, eps_hi=eps_hi, z_floor=z_floor, z_max=z_max
    )
    if np.any(hi <= lo):
        bad = int(np.sum(hi <= lo))
        raise RuntimeError(
            f"{bad} spectra have empty ε interval for Unif({eps_lo}, {eps_hi}) "
            f"inside z in [{z_floor}, {z_max}]"
        )
    u = rng.random((n, k))
    eps = lo[:, None] + (hi - lo)[:, None] * u
    z_new = (1.0 + z_old)[:, None] * np.exp(eps) - 1.0
    return z_new, eps


def warp_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    mask_valid: np.ndarray,
    z_old: float,
    z_new: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebin a stretched copy onto the original wavelength grid.

    ``F'(λ) = F(λ (1+z_old)/(1+z_new))``. Bins whose source λ is off-grid or
    originally masked are invalid (DAEP padding).
    """
    wave = np.asarray(wave, dtype=np.float64).reshape(-1)
    flux = np.asarray(flux, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask_valid, dtype=bool).reshape(-1)
    if wave.size != flux.size or flux.size != valid.size:
        raise ValueError("wave / flux / mask length mismatch")
    z_old = float(z_old)
    z_new = float(z_new)
    if z_old <= -1.0 or z_new <= -1.0:
        raise ValueError(f"z_old={z_old} z_new={z_new} would make 1+z non-positive")
    lam_src = wave * (1.0 + z_old) / (1.0 + z_new)
    in_window = (lam_src >= wave[0]) & (lam_src <= wave[-1])
    idx = np.clip(np.searchsorted(wave, lam_src), 1, wave.size - 1)
    left = idx - 1
    nearest = np.where(np.abs(wave[idx] - lam_src) < np.abs(wave[left] - lam_src), idx, left)
    src_ok = valid[nearest] & in_window
    ok = valid
    flux_new = np.zeros_like(flux, dtype=np.float64)
    if int(ok.sum()) >= 2:
        interp = np.interp(lam_src, wave[ok], flux[ok], left=np.nan, right=np.nan)
        good = src_ok & np.isfinite(interp)
        flux_new[good] = interp[good]
        mask_new = good
    else:
        mask_new = np.zeros_like(valid)
    return flux_new.astype(np.float32), mask_new.astype(bool)


def resolve_flux_rows(latent_meta: pd.DataFrame, flux_meta: pd.DataFrame) -> np.ndarray:
    """Map each latent-meta row to a row of ``noderedshift/flux.npy``."""
    if "spec_id" not in latent_meta.columns or "spec_id" not in flux_meta.columns:
        raise KeyError("Need spec_id on both latent meta and noderedshift meta")
    pos = {str(sid): i for i, sid in enumerate(flux_meta["spec_id"].tolist())}
    rows = np.empty(len(latent_meta), dtype=np.int64)
    missing = 0
    for i, sid in enumerate(latent_meta["spec_id"].tolist()):
        key = str(sid)
        if key not in pos:
            missing += 1
            rows[i] = -1
        else:
            rows[i] = int(pos[key])
    if missing:
        raise RuntimeError(
            f"{missing} latent rows have spec_id not in noderedshift/meta.csv"
        )
    return rows


def _cache_matches(
    data: np.lib.npyio.NpzFile,
    *,
    parent_idx: np.ndarray,
    k: int,
    eps_lo: float,
    eps_hi: float,
    seed: int,
    z_floor: float,
    z_max: float,
) -> bool:
    try:
        if not np.array_equal(np.asarray(data["parent_idx"], dtype=np.int64), parent_idx):
            return False
        if int(np.asarray(data["k"]).reshape(-1)[0]) != int(k):
            return False
        if abs(float(np.asarray(data["eps_lo"]).reshape(-1)[0]) - float(eps_lo)) > 1e-12:
            return False
        if abs(float(np.asarray(data["eps_hi"]).reshape(-1)[0]) - float(eps_hi)) > 1e-12:
            return False
        if int(np.asarray(data["seed"]).reshape(-1)[0]) != int(seed):
            return False
        if abs(float(np.asarray(data["z_floor"]).reshape(-1)[0]) - float(z_floor)) > 1e-12:
            return False
        if abs(float(np.asarray(data["z_max"]).reshape(-1)[0]) - float(z_max)) > 1e-12:
            return False
    except KeyError:
        return False
    return True


def _load_frozen_encoder(ckpt_path: Path, device: torch.device):
    from TwinsModel_Wiserep import repo_import_setup, build_daep

    if not DAEP_REPO.is_dir():
        raise FileNotFoundError(f"DAEP repo not found: {DAEP_REPO}")
    repo_import_setup(str(DAEP_REPO.resolve()))
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if "cfg" not in ckpt or "state_dict" not in ckpt:
        raise KeyError(f"{ckpt_path} is not a TwinsTrain best_ckpt (need cfg + state_dict)")
    model = build_daep(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _encode_flux_batch(
    model,
    wave: np.ndarray,
    flux: np.ndarray,
    mask_valid: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """``flux`` / ``mask_valid`` are (B, W); returns encoder z (B, L, D) on CPU."""
    b, w = flux.shape
    wave_t = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    batch = {
        "idx": torch.arange(b, dtype=torch.long),
        "flux": torch.from_numpy(np.asarray(flux, dtype=np.float32)),
        "mask": torch.from_numpy((~np.asarray(mask_valid, dtype=bool))),
        "phase": torch.zeros(b, dtype=torch.float32),
        "wavelength": wave_t.unsqueeze(0).expand(b, w).contiguous(),
    }
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        z = model.encoder.encode_raw(batch)
    return z.detach().float().cpu().numpy().astype(np.float32)


def ensure_aug_latents(
    *,
    latent_dir: Path,
    meta: pd.DataFrame,
    z_old: np.ndarray,
    train_idx: np.ndarray,
    embed_shape: tuple[int, ...],
    device: torch.device,
    k: int = AUG_K,
    eps_lo: float = AUG_EPS_LO,
    eps_hi: float = AUG_EPS_HI,
    z_floor: float = Z_FLOOR,
    z_max: float = Z_MAX,
    seed: int = 0,
    flux_dir: Path | None = None,
    ckpt_path: Path | None = None,
    encode_batch: int = AUG_ENCODE_BATCH,
    force: bool = False,
) -> Dict[str, np.ndarray]:
    """Load or build the (n_kept, L, D) augmented train embeddings."""
    latent_dir = Path(latent_dir).resolve()
    train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    z_old = np.asarray(z_old, dtype=np.float64).reshape(-1)
    if z_old.shape[0] != len(meta):
        raise ValueError("z_old must be aligned with meta rows")
    out_path = cache_path_for(latent_dir, k)
    _recover_partial_cache(out_path)
    if out_path.is_file() and not force:
        with np.load(out_path, allow_pickle=False) as cached:
            if _cache_matches(
                cached,
                parent_idx=train_idx,
                k=k,
                eps_lo=eps_lo,
                eps_hi=eps_hi,
                seed=seed,
                z_floor=z_floor,
                z_max=z_max,
            ):
                payload = {key: np.array(cached[key]) for key in cached.files}
                print(f"[aug] cache hit {out_path}  n={payload['z'].shape[0]}", flush=True)
                return payload
        print(f"[aug] cache params mismatch; rebuilding {out_path}", flush=True)

    flux_dir = Path(flux_dir or HENNA_NOZ).resolve()
    ckpt_path = Path(ckpt_path or (latent_dir / DEFAULT_CKPT_NAME)).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"--aug needs the Universal encoder checkpoint at {ckpt_path}"
        )

    flux_meta = pd.read_csv(flux_dir / "meta.csv", low_memory=False)
    wave = np.load(flux_dir / "wavelength.npy").astype(np.float32)
    flux = np.load(flux_dir / "flux.npy", mmap_mode="r")
    mask_valid = np.load(flux_dir / "mask.npy", mmap_mode="r")
    flux_rows = resolve_flux_rows(meta, flux_meta)
    z_train = z_old[train_idx]
    rng = np.random.default_rng(int(seed))
    z_new, eps = sample_z_new(
        z_train,
        rng,
        k=k,
        eps_lo=eps_lo,
        eps_hi=eps_hi,
        z_floor=z_floor,
        z_max=z_max,
    )

    n = int(train_idx.size)
    n_slots = n * int(k)
    print(
        f"[aug] stretching {n} train spectra × {k}  "
        f"ε~Unif({eps_lo:g}, {eps_hi:g}) on ln(1+z)  encode_batch={encode_batch}",
        flush=True,
    )

    model = _load_frozen_encoder(ckpt_path, device)
    bs = max(1, int(encode_batch))
    flux_batch: list[np.ndarray] = []
    mask_batch: list[np.ndarray] = []
    parent_buf: list[int] = []
    copy_buf: list[int] = []
    znew_buf: list[float] = []
    zold_buf: list[float] = []
    eps_buf: list[float] = []
    z_list: list[np.ndarray] = []
    skipped = 0

    def _flush() -> None:
        if not flux_batch:
            return
        z_list.append(
            _encode_flux_batch(
                model,
                wave,
                np.stack(flux_batch, axis=0),
                np.stack(mask_batch, axis=0),
                device,
            )
        )
        flux_batch.clear()
        mask_batch.clear()

    try:
        pbar = tqdm(total=n_slots, desc="aug encode", leave=False)
        for i, src in enumerate(train_idx):
            frow = int(flux_rows[int(src)])
            for j in range(int(k)):
                f_new, m_new = warp_spectrum(
                    wave,
                    np.asarray(flux[frow]),
                    np.asarray(mask_valid[frow]),
                    float(z_train[i]),
                    float(z_new[i, j]),
                )
                pbar.update(1)
                if int(m_new.sum()) < MIN_VALID_BINS:
                    skipped += 1
                    continue
                flux_batch.append(f_new)
                mask_batch.append(m_new)
                parent_buf.append(int(src))
                copy_buf.append(int(j))
                znew_buf.append(float(z_new[i, j]))
                zold_buf.append(float(z_train[i]))
                eps_buf.append(float(eps[i, j]))
                if len(flux_batch) >= bs:
                    _flush()
        _flush()
        pbar.close()
    finally:
        del model
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()

    n_keep = len(parent_buf)
    if n_keep == 0 or not z_list:
        raise RuntimeError("All augmented spectra were dropped (too few valid bins)")
    print(
        f"[aug] kept {n_keep} / {n_slots} warped spectra  skipped={skipped}",
        flush=True,
    )

    z_aug = np.concatenate(z_list, axis=0)
    expect = (n_keep,) + tuple(int(x) for x in embed_shape)
    if z_aug.shape != expect:
        raise RuntimeError(f"aug latent shape {z_aug.shape} != expected {expect}")

    payload = {
        "z": z_aug.astype(np.float32),
        "z_new": np.asarray(znew_buf, dtype=np.float32),
        "z_old": np.asarray(zold_buf, dtype=np.float32),
        "eps": np.asarray(eps_buf, dtype=np.float32),
        "aug_parent": np.asarray(parent_buf, dtype=np.int64),
        "aug_copy": np.asarray(copy_buf, dtype=np.int32),
        "parent_idx": train_idx.astype(np.int64),
        "k": np.int32(k),
        "eps_lo": np.float64(eps_lo),
        "eps_hi": np.float64(eps_hi),
        "seed": np.int64(seed),
        "z_floor": np.float64(z_floor),
        "z_max": np.float64(z_max),
        "skipped": np.int64(skipped),
        "ckpt": np.asarray(str(ckpt_path)),
        "flux_dir": np.asarray(str(flux_dir)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _npz_temp_path(out_path)
    np.savez_compressed(tmp, **payload)
    tmp.replace(out_path)
    meta_json = {
        "n_train": n,
        "k": int(k),
        "n_kept": n_keep,
        "skipped": skipped,
        "eps_lo": float(eps_lo),
        "eps_hi": float(eps_hi),
        "seed": int(seed),
        "ckpt": str(ckpt_path),
        "flux_dir": str(flux_dir),
    }
    (out_path.parent / "aug_cfg.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
    print(f"[aug] wrote {out_path}  z={tuple(z_aug.shape)}", flush=True)
    return payload


def dataset_from_cache(cache: Dict[str, np.ndarray]) -> TensorLatentDataset:
    parents = np.asarray(cache["aug_parent"], dtype=np.int64)
    copies = np.asarray(cache["aug_copy"], dtype=np.int32)
    ids = [f"{int(p)}:aug{int(c)}" for p, c in zip(parents.tolist(), copies.tolist())]
    return TensorLatentDataset(cache["z"], cache["z_new"], ids)
