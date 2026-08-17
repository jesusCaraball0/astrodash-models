"""DASH spectrum preprocessing used by 1D CNN training (formerly prod_backend)."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when spectrum data or preprocessing is invalid."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


def validate_spectrum_data(x: List[float], y: List[float]) -> None:
    if not x or not y or len(x) != len(y):
        raise ValidationError("Spectrum x and y must be non-empty and of equal length.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValidationError("Spectrum data contains NaN values.")


def validate_redshift(redshift) -> float:
    try:
        z = float(redshift)
        if z < 0:
            raise ValidationError("Redshift must be non-negative.")
        return z
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("Invalid redshift value.")


def validate_spectrum(
    x: List[float],
    y: List[float],
    redshift: Optional[float] = None,
) -> None:
    validate_spectrum_data(x, y)
    if redshift is not None:
        validate_redshift(redshift)


class DashSpectrumProcessor:
    """
    Handles all preprocessing for the Dash (CNN) classifier.
    Includes normalization, wavelength binning, continuum removal, mean zeroing, and apodization.
    """

    DEFAULT_EDGE_WIDTH = 50
    DEFAULT_EDGE_RATIO = 4
    DEFAULT_OUTER_VAL = 0.5
    MIN_FILTER_SIZE = 3

    def __init__(self, w0: float, w1: float, nw: int, num_spline_points: int = 13):
        if w0 <= 0 or w1 <= 0 or w0 >= w1:
            raise ValueError(f"Invalid wavelength range: w0={w0}, w1={w1}")
        if nw <= 0:
            raise ValueError(f"Invalid number of bins: nw={nw}")
        if num_spline_points < 3:
            raise ValueError(f"Invalid spline points: {num_spline_points} (minimum 3)")

        self.w0 = float(w0)
        self.w1 = float(w1)
        self.nw = int(nw)
        self.num_spline_points = int(num_spline_points)

        logger.info(f"DashSpectrumProcessor initialized: w0={w0}, w1={w1}, nw={nw}")

    def process(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        z: float,
        smooth: int = 0,
        min_wave: Optional[float] = None,
        max_wave: Optional[float] = None,
    ) -> Tuple[np.ndarray, int, int, float]:
        try:
            validate_spectrum(wave.tolist(), flux.tolist(), z)

            wave = np.asarray(wave, dtype=float).copy()
            flux = np.asarray(flux, dtype=float).copy()
            if wave.size > 1 and wave[0] > wave[-1]:
                perm = np.argsort(wave)
                wave = wave[perm]
                flux = flux[perm]

            flux_norm = self.normalise_spectrum(flux)
            effective_min = self.w0 if min_wave is None else min_wave
            effective_max = self.w1 if max_wave is None else max_wave
            flux_limited = self.limit_wavelength_range(wave, flux_norm, effective_min, effective_max)

            effective_smooth = smooth if smooth > 0 else 6
            w_density = (self.w1 - self.w0) / self.nw
            wavelength_density = (np.max(wave) - np.min(wave)) / max(len(wave), 1)
            if wavelength_density <= 0:
                filter_size = 1
            else:
                filter_size = int(w_density / wavelength_density * effective_smooth / 2) * 2 + 1
            if filter_size < 1:
                filter_size = 1
            if filter_size % 2 == 0:
                filter_size += 1
            n_flux = len(flux_limited)
            if filter_size > n_flux:
                filter_size = n_flux if n_flux % 2 == 1 else max(1, n_flux - 1)
            flux_smoothed = medfilt(flux_limited, kernel_size=filter_size)

            wave_deredshifted = wave / (1 + z)
            if len(wave_deredshifted) < 2:
                raise ValidationError("Spectrum is out of classification range after deredshifting")

            mask = (wave_deredshifted >= self.w0) & (wave_deredshifted < self.w1)
            wave_dereds = wave_deredshifted[mask]
            flux_dereds = flux_smoothed[mask]
            if wave_dereds.size == 0:
                raise ValidationError(
                    f"Spectrum out of wavelength range [{self.w0}, {self.w1}] after deredshifting"
                )
            flux_dereds = self.normalise_spectrum(flux_dereds)

            binned_wave, binned_flux, min_idx, max_idx = self.log_wavelength_binning(
                wave_dereds, flux_dereds
            )

            if min_idx == max_idx == 0 and not np.any(binned_flux):
                flat = np.full(self.nw, self.DEFAULT_OUTER_VAL, dtype=float)
                return flat, 0, 0, z

            cont_removed, _ = self.continuum_removal(binned_wave, binned_flux, min_idx, max_idx)
            mean_zero_flux = self.mean_zero(cont_removed, min_idx, max_idx)
            apodized_flux = self.apodize(mean_zero_flux, min_idx, max_idx)
            flux_norm_final = self.normalise_spectrum(apodized_flux)
            flux_norm_final = self.zero_non_overlap_part(
                flux_norm_final, min_idx, max_idx, self.DEFAULT_OUTER_VAL
            )

            logger.debug(f"Processing completed: min_idx={min_idx}, max_idx={max_idx}")
            return flux_norm_final, min_idx, max_idx, z

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Spectrum processing failed: {str(e)}")
            raise ValidationError(f"Spectrum processing failed: {str(e)}") from e

    def process_no_redshift(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        smooth: int = 0,
        min_wave: Optional[float] = None,
        max_wave: Optional[float] = None,
    ) -> Tuple[np.ndarray, int, int]:
        try:
            validate_spectrum(wave.tolist(), flux.tolist(), None)

            wave = np.asarray(wave, dtype=float).copy()
            flux = np.asarray(flux, dtype=float).copy()
            if wave.size > 1 and wave[0] > wave[-1]:
                perm = np.argsort(wave)
                wave = wave[perm]
                flux = flux[perm]

            flux_norm = self.normalise_spectrum(flux)
            effective_min = self.w0 if min_wave is None else min_wave
            effective_max = self.w1 if max_wave is None else max_wave
            flux_limited = self.limit_wavelength_range(wave, flux_norm, effective_min, effective_max)

            effective_smooth = smooth if smooth > 0 else 6
            w_density = (self.w1 - self.w0) / self.nw
            wavelength_density = (np.max(wave) - np.min(wave)) / max(len(wave), 1)
            if wavelength_density <= 0:
                filter_size = 1
            else:
                filter_size = int(w_density / wavelength_density * effective_smooth / 2) * 2 + 1
            if filter_size < 1:
                filter_size = 1
            if filter_size % 2 == 0:
                filter_size += 1
            n_flux = len(flux_limited)
            if filter_size > n_flux:
                filter_size = n_flux if n_flux % 2 == 1 else max(1, n_flux - 1)
            flux_smoothed = medfilt(flux_limited, kernel_size=filter_size)

            mask = (wave >= self.w0) & (wave < self.w1)
            wave_restricted = wave[mask]
            flux_restricted = flux_smoothed[mask]
            if wave_restricted.size < 2:
                raise ValidationError(
                    "Spectrum is out of classification range (fewer than 2 points in [w0, w1])"
                )
            if flux_restricted.size == 0:
                raise ValidationError(
                    f"Spectrum out of wavelength range [{self.w0}, {self.w1}]"
                )
            flux_dereds = self.normalise_spectrum(flux_restricted)

            binned_wave, binned_flux, min_idx, max_idx = self.log_wavelength_binning(
                wave_restricted, flux_dereds
            )

            if min_idx == max_idx == 0 and not np.any(binned_flux):
                flat = np.full(self.nw, self.DEFAULT_OUTER_VAL, dtype=float)
                return flat, 0, 0

            cont_removed, _ = self.continuum_removal(binned_wave, binned_flux, min_idx, max_idx)
            mean_zero_flux = self.mean_zero(cont_removed, min_idx, max_idx)
            apodized_flux = self.apodize(mean_zero_flux, min_idx, max_idx)
            flux_norm_final = self.normalise_spectrum(apodized_flux)
            flux_norm_final = self.zero_non_overlap_part(
                flux_norm_final, min_idx, max_idx, self.DEFAULT_OUTER_VAL
            )

            logger.debug(f"Processing (no redshift) completed: min_idx={min_idx}, max_idx={max_idx}")
            return flux_norm_final, min_idx, max_idx

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Spectrum processing failed: {str(e)}")
            raise ValidationError(f"Spectrum processing failed: {str(e)}") from e

    def _apply_smoothing(self, wave: np.ndarray, flux: np.ndarray, smooth: int) -> np.ndarray:
        try:
            wavelength_density = (np.max(wave) - np.min(wave)) / len(wave)
            w_density = (self.w1 - self.w0) / self.nw
            filter_size = int(w_density / wavelength_density * smooth / 2) * 2 + 1

            if filter_size >= self.MIN_FILTER_SIZE:
                flux_smoothed = medfilt(flux, kernel_size=filter_size)
                logger.debug(f"Applied smoothing with filter size {filter_size}")
                return flux_smoothed
            logger.warning(f"Filter size {filter_size} too small, skipping smoothing")
            return flux
        except Exception as e:
            logger.warning(f"Smoothing failed: {str(e)}, returning original flux")
            return flux

    @staticmethod
    def normalise_spectrum(flux: np.ndarray) -> np.ndarray:
        if len(flux) == 0:
            raise ValidationError("Cannot normalize empty array")

        flux_min, flux_max = np.min(flux), np.max(flux)

        if not np.isfinite(flux_min) or not np.isfinite(flux_max):
            raise ValidationError("Array contains non-finite values")

        if flux_min == flux_max:
            logger.warning("Normalizing spectrum: constant flux array")
            return np.zeros(len(flux))

        if flux_max <= flux_min:
            raise ValidationError(f"Invalid flux range: min={flux_min}, max={flux_max}")

        return (flux - flux_min) / (flux_max - flux_min)

    @staticmethod
    def limit_wavelength_range(
        wave: np.ndarray,
        flux: np.ndarray,
        min_wave: Optional[float],
        max_wave: Optional[float],
    ) -> np.ndarray:
        flux_out = np.copy(flux)

        if min_wave is not None and np.isfinite(min_wave):
            min_idx = int((np.abs(np.asarray(wave) - min_wave)).argmin())
            flux_out[:min_idx] = 0

        if max_wave is not None and np.isfinite(max_wave):
            max_idx = int((np.abs(np.asarray(wave) - max_wave)).argmin())
            flux_out[max_idx:] = 0

        return flux_out

    def log_wavelength_binning(
        self, wave: np.ndarray, flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        try:
            dwlog = np.log(self.w1 / self.w0) / self.nw
            wlog = self.w0 * np.exp(np.arange(0, self.nw) * dwlog)
            binned_flux = np.interp(wlog, wave, flux, left=0, right=0)

            non_zero_indices = np.where(binned_flux != 0)[0]

            if len(non_zero_indices) == 0:
                min_index = max_index = 0
            else:
                min_index = non_zero_indices[0]
                max_index = non_zero_indices[-1]

            return wlog, binned_flux, min_index, max_index

        except Exception as e:
            logger.error(f"Wavelength binning failed: {str(e)}")
            raise ValidationError(f"Wavelength binning failed: {str(e)}") from e

    def continuum_removal(
        self, wave: np.ndarray, flux: np.ndarray, min_idx: int, max_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            min_idx = int(np.clip(min_idx, 0, len(flux) - 1))
            max_idx = int(np.clip(max_idx, min_idx, len(flux) - 1))

            flux_plus = flux + 1.0
            cont_removed = np.copy(flux_plus)

            continuum = np.zeros_like(flux_plus)
            if (max_idx - min_idx) > 5:
                spline = UnivariateSpline(
                    wave[min_idx : max_idx + 1], flux_plus[min_idx : max_idx + 1], k=3
                )
                spline_wave = np.linspace(
                    wave[min_idx], wave[max_idx], num=self.num_spline_points, endpoint=True
                )
                spline_points = spline(spline_wave)
                spline_more = UnivariateSpline(spline_wave, spline_points, k=3)
                spline_points_more = spline_more(wave[min_idx : max_idx + 1])
                continuum[min_idx : max_idx + 1] = spline_points_more
            else:
                continuum[min_idx : max_idx + 1] = 1.0

            valid = continuum[min_idx : max_idx + 1] != 0
            if np.any(valid):
                cont_removed[min_idx : max_idx + 1][valid] = (
                    flux_plus[min_idx : max_idx + 1][valid]
                    / continuum[min_idx : max_idx + 1][valid]
                )

            cont_removed_norm = DashSpectrumProcessor.normalise_spectrum(cont_removed - 1.0)
            cont_removed_norm[:min_idx] = 0.0
            cont_removed_norm[max_idx + 1 :] = 0.0

            return cont_removed_norm, continuum - 1.0

        except Exception as e:
            logger.error(f"Continuum removal failed: {str(e)}")
            raise ValidationError(f"Continuum removal failed: {str(e)}") from e

    @staticmethod
    def mean_zero(flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        if flux.size == 0:
            return flux

        min_idx = int(np.clip(min_idx, 0, len(flux) - 1))
        max_idx = int(np.clip(max_idx, min_idx, len(flux) - 1))

        if max_idx <= min_idx:
            return flux

        out = np.copy(flux)
        mean_flux = np.mean(out[min_idx:max_idx])
        out[min_idx : max_idx + 1] = out[min_idx : max_idx + 1] - mean_flux
        return out

    @staticmethod
    def apodize(flux: np.ndarray, min_idx: int, max_idx: int) -> np.ndarray:
        if flux.size == 0:
            return flux

        out = np.copy(flux)
        nw = len(out)
        min_idx = int(np.clip(min_idx, 0, nw - 1))
        max_idx = int(np.clip(max_idx, min_idx, nw - 1))

        percent = 0.05
        nsquash = int(nw * percent)
        if nsquash <= 1:
            return out

        for i in range(nsquash):
            arg = np.pi * i / (nsquash - 1)
            factor = 0.5 * (1.0 - np.cos(arg))
            if (min_idx + i < nw) and (max_idx - i >= 0):
                out[min_idx + i] = factor * out[min_idx + i]
                out[max_idx - i] = factor * out[max_idx - i]
            else:
                break

        return out

    @staticmethod
    def zero_non_overlap_part(
        array: np.ndarray,
        min_index: int,
        max_index: int,
        outer_val: float = 0.0,
    ) -> np.ndarray:
        sliced_array = np.copy(array)

        min_index = np.clip(min_index, 0, len(sliced_array) - 1)
        max_index = np.clip(max_index, min_index, len(sliced_array) - 1)

        sliced_array[:min_index] = outer_val
        sliced_array[max_index + 1 :] = outer_val

        return sliced_array
