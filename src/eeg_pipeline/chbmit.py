from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import mne
import numpy as np
import pandas as pd
import requests

PHYSIONET_BASE_URL = "https://physionet.org/files/chbmit/1.0.0"


@dataclass(frozen=True)
class RecordingTarget:
    patient_id: str
    filename: str

    @property
    def url(self) -> str:
        return f"{PHYSIONET_BASE_URL}/{self.patient_id}/{self.filename}"


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)

    return destination


def ensure_subset(
    data_dir: Path,
    patient_id: str,
    filenames: Iterable[str],
) -> dict[str, Path]:
    patient_dir = data_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    summary_name = f"{patient_id}-summary.txt"
    summary_path = download_file(
        url=f"{PHYSIONET_BASE_URL}/{patient_id}/{summary_name}",
        destination=patient_dir / summary_name,
    )
    files[summary_name] = summary_path

    for filename in filenames:
        files[filename] = download_file(
            url=f"{PHYSIONET_BASE_URL}/{patient_id}/{filename}",
            destination=patient_dir / filename,
        )

    return files


def parse_summary(summary_path: Path) -> tuple[float, list[str], pd.DataFrame]:
    text = summary_path.read_text()

    sampling_match = re.search(r"Data Sampling Rate:\s+(\d+)\s+Hz", text)
    if not sampling_match:
        raise ValueError(f"Unable to parse sampling rate from {summary_path}")

    channel_matches = re.findall(r"Channel \d+:\s+(.+)", text)

    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            if current is not None:
                records.append(current)
            current = {
                "filename": line.split(":", 1)[1].strip(),
                "seizure_count": 0,
                "seizure_intervals": [],
                "seizure_start_seconds": None,
                "seizure_end_seconds": None,
            }
        elif current is None:
            continue
        elif line.startswith("File Start Time:"):
            current["file_start_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("File End Time:"):
            current["file_end_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("Number of Seizures in File:"):
            current["seizure_count"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Seizure Start Time:"):
            seizure_start = int(line.split(":", 1)[1].split()[0])
            current["seizure_intervals"].append([seizure_start, None])
            if current["seizure_start_seconds"] is None:
                current["seizure_start_seconds"] = seizure_start
        elif line.startswith("Seizure End Time:"):
            seizure_end = int(line.split(":", 1)[1].split()[0])
            current["seizure_intervals"][-1][1] = seizure_end
            if current["seizure_end_seconds"] is None:
                current["seizure_end_seconds"] = seizure_end

    if current is not None:
        records.append(current)

    summary_df = pd.DataFrame(records)
    return float(sampling_match.group(1)), channel_matches, summary_df


def load_recording(edf_path: Path, pick_channels: list[str] | None = None) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    removable = [name for name in raw.ch_names if name.strip() == "-" or name.upper().startswith("ECG")]
    if removable:
        raw.drop_channels(removable)

    if pick_channels is not None:
        available = [name for name in pick_channels if name in raw.ch_names]
        raw.pick(available)

    return raw


def common_channels(raws: Iterable[mne.io.BaseRaw]) -> list[str]:
    channel_sets = [set(raw.ch_names) for raw in raws]
    shared = set.intersection(*channel_sets)
    return sorted(shared)


def seizure_intervals_for_file(summary_df: pd.DataFrame, filename: str) -> list[tuple[float, float]]:
    match = summary_df.loc[summary_df["filename"] == filename]
    if match.empty:
        raise KeyError(f"{filename} not found in summary data")

    intervals = match.iloc[0]["seizure_intervals"]
    if intervals is None or (isinstance(intervals, float) and np.isnan(intervals)):
        return []

    return [(float(start), float(end)) for start, end in intervals if start is not None and end is not None]
