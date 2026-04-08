from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch


BANDS_HZ: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _bandpower_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    band_limits: tuple[float, float],
) -> np.ndarray:
    low_hz, high_hz = band_limits
    band_mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(band_mask):
        return np.zeros(psd.shape[0], dtype=float)
    return np.trapz(psd[:, band_mask], frequencies[band_mask], axis=1)


def _line_length(window: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(window, axis=1)), axis=1)


def _sanitize_channel_name(channel_name: str) -> str:
    sanitized = []
    for character in channel_name:
        sanitized.append(character if character.isalnum() else "_")
    return "".join(sanitized).strip("_")


def _window_feature_row(window: np.ndarray, sfreq: float, channel_names: list[str]) -> dict[str, float]:
    frequencies, psd = welch(window, fs=sfreq, axis=1, nperseg=min(window.shape[1], int(sfreq * 2)))

    row: dict[str, float] = {}

    channel_means = np.mean(window, axis=1)
    channel_variances = np.var(window, axis=1)
    channel_line_lengths = _line_length(window)

    row["signal_mean_mean"] = float(np.mean(channel_means))
    row["signal_mean_std"] = float(np.std(channel_means))
    row["signal_variance_mean"] = float(np.mean(channel_variances))
    row["signal_variance_std"] = float(np.std(channel_variances))
    row["line_length_mean"] = float(np.mean(channel_line_lengths))
    row["line_length_std"] = float(np.std(channel_line_lengths))

    for channel_name, channel_mean, channel_variance, channel_line_length in zip(
        channel_names,
        channel_means,
        channel_variances,
        channel_line_lengths,
    ):
        safe_name = _sanitize_channel_name(channel_name)
        row[f"{safe_name}__mean"] = float(channel_mean)
        row[f"{safe_name}__variance"] = float(channel_variance)
        row[f"{safe_name}__line_length"] = float(channel_line_length)

    for band_name, band_limits in BANDS_HZ.items():
        band_power = _bandpower_from_psd(frequencies=frequencies, psd=psd, band_limits=band_limits)
        row[f"{band_name}_power_mean"] = float(np.mean(band_power))
        row[f"{band_name}_power_std"] = float(np.std(band_power))
        for channel_name, channel_band_power in zip(channel_names, band_power):
            safe_name = _sanitize_channel_name(channel_name)
            row[f"{safe_name}__{band_name}_power"] = float(channel_band_power)

    return row


def extract_features(windows: np.ndarray, sfreq: float, channel_names: list[str]) -> pd.DataFrame:
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")
    if windows.shape[1] != len(channel_names):
        raise ValueError("channel_names length must match the window channel dimension")

    rows = [
        _window_feature_row(window=window, sfreq=sfreq, channel_names=channel_names)
        for window in windows
    ]
    features = pd.DataFrame(rows)
    features.insert(0, "window_id", np.arange(len(features), dtype=int))
    return features
