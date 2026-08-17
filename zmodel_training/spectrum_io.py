"""Load WISeREP spectrum files for Dash training (formerly FileSpectrumRepository)."""
from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from dash_preprocess import validate_spectrum

logger = logging.getLogger(__name__)

WAVE_FILTER_MIN = 4000.0
WAVE_FILTER_MAX = 9000.0
_TEXT_SUFFIXES = (".dat", ".txt", ".ascii", ".asci", ".flm")


def load_spectrum_file(filepath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Parse a spectrum file and return (wave, flux) filtered to 4000–9000 Å,
    matching the old FileSpectrumRepository loaders.
    """
    path = Path(filepath)
    name = path.name.lower()
    try:
        if name.endswith(".lnw"):
            xy = _read_lnw_file(path)
        elif name.endswith(_TEXT_SUFFIXES):
            xy = _read_text_file(path)
        elif name.endswith(".fits"):
            xy = _read_fits_file(path)
        elif name.endswith(".csv"):
            xy = _read_csv_file(path)
        else:
            logger.error(f"Unsupported file format: {path.name}")
            return None
    except Exception as e:
        logger.error(f"Error reading file {path.name}: {e}", exc_info=True)
        return None

    if xy is None:
        return None
    wave, flux = xy
    if wave.size == 0 or flux.size == 0:
        return None
    return wave, flux


def _read_text_content(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _pairs_to_arrays(
    spectrum_data: List[Tuple[float, float]], filename: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not spectrum_data:
        logger.error(f"No valid spectrum data in {filename} (after 4000-9000 Å filter)")
        return None
    spectrum_data.sort(key=lambda x: x[0])
    wave = [w for w, _ in spectrum_data]
    flux = [f for _, f in spectrum_data]
    try:
        validate_spectrum(wave, flux, None)
    except Exception as e:
        logger.error(f"Spectrum validation failed for {filename}: {e}")
        return None
    return np.asarray(wave, dtype=float), np.asarray(flux, dtype=float)


def _read_text_file(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    content = _read_text_content(path)
    spectrum_data: List[Tuple[float, float]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            wavelength = float(parts[0])
            flux = float(parts[1])
        except ValueError:
            continue
        if WAVE_FILTER_MIN <= wavelength <= WAVE_FILTER_MAX:
            spectrum_data.append((wavelength, flux))
    return _pairs_to_arrays(spectrum_data, path.name)


def _read_lnw_file(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    content = _read_text_content(path)
    spectrum_data: List[Tuple[float, float]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        try:
            wavelength = float(parts[0])
            flux = float(parts[1])
            if WAVE_FILTER_MIN <= wavelength <= WAVE_FILTER_MAX:
                spectrum_data.append((wavelength, flux))
        except ValueError:
            continue
    return _pairs_to_arrays(spectrum_data, path.name)


def _read_csv_file(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    content = _read_text_content(path)
    for delimiter in (",", "\t"):
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            continue
        header = [c.strip().upper() for c in rows[0]]
        data_rows = rows[1:]

        wave_idx = None
        flux_idx = None
        for i, col in enumerate(header):
            if col in ("WAVE", "WAVELENGTH", "LAMBDA", "WL"):
                wave_idx = i
            if col in ("FLUX", "FLUX_DENSITY", "F"):
                flux_idx = i
        if wave_idx is None or flux_idx is None:
            if len(header) >= 2:
                wave_idx, flux_idx = 0, 1
            else:
                continue

        spectrum_data: List[Tuple[float, float]] = []
        for row in data_rows:
            if len(row) <= max(wave_idx, flux_idx):
                continue
            try:
                w = float(row[wave_idx].strip())
                f = float(row[flux_idx].strip())
                if WAVE_FILTER_MIN <= w <= WAVE_FILTER_MAX:
                    spectrum_data.append((w, f))
            except (ValueError, IndexError):
                continue

        parsed = _pairs_to_arrays(spectrum_data, path.name)
        if parsed is not None:
            return parsed

    logger.error(f"No valid spectrum data in CSV file {path.name}")
    return None


def _read_fits_file(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    from astropy.io import fits

    with fits.open(path) as hdul:
        spectrum_data = None
        wavelength = None
        flux = None

        for ext in ["SPECTRUM", "SPECTRA", "FLUX", "DATA"]:
            if ext in hdul:
                spectrum_data = hdul[ext].data
                break
        if spectrum_data is None and len(hdul) > 1:
            spectrum_data = hdul[1].data

        if spectrum_data is not None:
            if hasattr(spectrum_data, "wavelength") and hasattr(spectrum_data, "flux"):
                wavelength = np.asarray(spectrum_data.wavelength, dtype=float)
                flux = np.asarray(spectrum_data.flux, dtype=float)
            elif hasattr(spectrum_data, "wave") and hasattr(spectrum_data, "flux"):
                wavelength = np.asarray(spectrum_data.wave, dtype=float)
                flux = np.asarray(spectrum_data.flux, dtype=float)
            elif getattr(spectrum_data.dtype, "names", None) and len(spectrum_data.dtype.names) >= 2:
                wavelength = np.asarray(spectrum_data[spectrum_data.dtype.names[0]], dtype=float)
                flux = np.asarray(spectrum_data[spectrum_data.dtype.names[1]], dtype=float)
            else:
                spectrum_data = None

        if spectrum_data is None and wavelength is None and len(hdul) > 0:
            primary = hdul[0]
            if hasattr(primary, "data") and primary.data is not None:
                data = np.asarray(primary.data, dtype=float).flatten()
                if data.ndim == 1 and len(data) > 0:
                    h = primary.header
                    crval1 = h.get("CRVAL1")
                    crpix1 = h.get("CRPIX1", 1)
                    cdel1 = h.get("CDELT1")
                    if crval1 is not None and cdel1 is not None:
                        wavelength = crval1 + (np.arange(len(data), dtype=float) + 1 - crpix1) * cdel1
                        flux = data
                    else:
                        logger.error(f"No spectrum table and no WCS (CRVAL1/CDELT1) in FITS file {path.name}")
                        return None
                else:
                    logger.error(f"No spectrum data found in FITS file {path.name}")
                    return None
            else:
                logger.error(f"No spectrum data found in FITS file {path.name}")
                return None

        if wavelength is None or flux is None:
            logger.error(f"No spectrum data found in FITS file {path.name}")
            return None

        filtered = [
            (w, f)
            for w, f in zip(wavelength.tolist(), flux.tolist())
            if WAVE_FILTER_MIN <= w <= WAVE_FILTER_MAX
        ]
        return _pairs_to_arrays(filtered, path.name)
