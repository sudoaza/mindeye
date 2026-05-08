"""
Event-aligned cropper for ZUNA-normalized continuous EEG.

ZUNA output FIFs currently do not preserve stimulus annotations, so the safest
alignment path is:
1. read `stim_on` annotation times from the original raw NOD FIF,
2. convert those onset times to samples in the corresponding ZUNA FIF, and
3. crop a short semantic window from the ZUNA signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import warnings

import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CropConfig:
    """Configuration for semantic EEG crops."""

    tmin: float = -0.25
    tmax: float = 1.0
    expected_sfreq: float = 256.0
    event_name: str = "stim_on"


@dataclass(frozen=True)
class CropResult:
    """Paths and summary stats for one cropped run."""

    run: int
    epochs_saved: int
    dropped_out_of_bounds: int
    shape: tuple[int, int, int]
    sfreq: float
    fif_path: Path
    npz_path: Path
    metadata_path: Path


def _read_raw(path: str | Path, preload: bool = False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.io.read_raw_fif(path, preload=preload, verbose=False)


def stim_onsets_from_raw(raw_fif_path: str | Path, event_name: str = "stim_on") -> np.ndarray:
    """Return event onset times in seconds from a raw NOD FIF annotation stream."""
    raw = _read_raw(raw_fif_path, preload=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    event_code = event_dict.get(event_name)
    if event_code is None:
        raise ValueError(f"Event {event_name!r} not found in annotations for {raw_fif_path}")
    event_samples = events[events[:, 2] == event_code][:, 0]
    return event_samples / float(raw.info["sfreq"])


def events_for_run(
    events_df: pd.DataFrame,
    *,
    session: str | None = None,
    run: int,
) -> pd.DataFrame:
    """Filter the NOD detailed-events CSV to one session/run."""
    mask = events_df["run"].astype(int) == int(run)
    if session is not None and "session" in events_df.columns:
        mask &= events_df["session"].astype(str) == session
    return events_df.loc[mask].reset_index(drop=True).copy()


def build_mne_events_for_zuna(
    onset_seconds: np.ndarray,
    zuna_sfreq: float,
    n_times: int,
    config: CropConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert source onset times to MNE event rows in the ZUNA raw sample space.

    Returns `(events, valid_mask)`, where invalid rows would place the crop
    outside the ZUNA output duration.
    """
    events = np.zeros((len(onset_seconds), 3), dtype=int)
    events[:, 0] = np.rint(onset_seconds * zuna_sfreq).astype(int)
    events[:, 2] = 1

    start = events[:, 0] + int(round(config.tmin * zuna_sfreq))
    stop = events[:, 0] + int(round(config.tmax * zuna_sfreq))
    valid = (start >= 0) & (stop < n_times)
    return events, valid


def crop_zuna_run_to_epochs(
    *,
    raw_fif_path: str | Path,
    zuna_fif_path: str | Path,
    events_df: pd.DataFrame,
    run: int,
    output_dir: str | Path,
    subject: str = "sub-01",
    session: str = "ImageNet01",
    config: CropConfig | None = None,
) -> CropResult:
    """Crop one ZUNA output FIF into event-aligned semantic epochs."""
    config = config or CropConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zuna = _read_raw(zuna_fif_path, preload=True)
    zuna_sfreq = float(zuna.info["sfreq"])
    if abs(zuna_sfreq - config.expected_sfreq) > 1:
        print(f"Warning: expected ~{config.expected_sfreq}Hz ZUNA output, got {zuna_sfreq}Hz")

    onset_seconds = stim_onsets_from_raw(raw_fif_path, config.event_name)
    metadata = events_for_run(events_df, session=session, run=run)
    n = min(len(onset_seconds), len(metadata))
    onset_seconds = onset_seconds[:n]
    metadata = metadata.iloc[:n].copy()
    metadata.insert(0, "stim_onset_sec", onset_seconds)
    metadata.insert(0, "zuna_source_fif", Path(zuna_fif_path).name)

    mne_events, valid = build_mne_events_for_zuna(onset_seconds, zuna_sfreq, zuna.n_times, config)
    metadata = metadata.loc[valid].reset_index(drop=True)
    mne_events = mne_events[valid]

    epochs = mne.Epochs(
        zuna,
        mne_events,
        event_id={config.event_name: 1},
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=None,
        metadata=metadata,
        preload=True,
        verbose=False,
    )
    data = epochs.get_data(copy=True)

    stem = f"{subject}_ses-{session}_run-{run:02d}_zuna_semantic"
    fif_path = output_dir / f"{stem}-epo.fif"
    npz_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{subject}_ses-{session}_run-{run:02d}_metadata.csv"

    epochs.save(fif_path, overwrite=True, verbose=False)
    np.savez_compressed(
        npz_path,
        eeg=data,
        sfreq=zuna_sfreq,
        times=epochs.times,
        ch_names=np.array(epochs.ch_names, dtype=object),
    )
    metadata.to_csv(metadata_path, index=False)

    return CropResult(
        run=int(run),
        epochs_saved=int(len(epochs)),
        dropped_out_of_bounds=int((~valid).sum()),
        shape=tuple(int(x) for x in data.shape),
        sfreq=zuna_sfreq,
        fif_path=fif_path,
        npz_path=npz_path,
        metadata_path=metadata_path,
    )


def crop_zuna_runs(
    *,
    raw_dir: str | Path,
    zuna_dir: str | Path,
    events_csv: str | Path,
    output_dir: str | Path,
    subject: str = "sub-01",
    session: str = "ImageNet01",
    runs: Iterable[int] = range(1, 6),
    config: CropConfig | None = None,
) -> dict:
    """Crop a batch of matching raw/ZUNA FIF runs and write a summary JSON."""
    raw_dir = Path(raw_dir)
    zuna_dir = Path(zuna_dir)
    output_dir = Path(output_dir)
    events_df = pd.read_csv(events_csv)
    config = config or CropConfig()

    all_metadata = []
    summary = {
        "subject": subject,
        "session": session,
        "tmin": config.tmin,
        "tmax": config.tmax,
        "runs": [],
    }

    for run in runs:
        raw_fif = raw_dir / f"{subject}_ses-{session}_task-ImageNet_run-{run:02d}_eeg_clean.fif"
        zuna_fif = zuna_dir / f"{subject}_ses-{session}_task-ImageNet_run-{run:02d}_eeg_clean.fif"
        if not raw_fif.exists() or not zuna_fif.exists():
            summary["runs"].append({
                "run": int(run),
                "status": "missing_input",
                "raw_exists": raw_fif.exists(),
                "zuna_exists": zuna_fif.exists(),
            })
            continue

        result = crop_zuna_run_to_epochs(
            raw_fif_path=raw_fif,
            zuna_fif_path=zuna_fif,
            events_df=events_df,
            run=int(run),
            output_dir=output_dir,
            subject=subject,
            session=session,
            config=config,
        )
        run_meta = pd.read_csv(result.metadata_path)
        run_meta["epoch_file"] = result.fif_path.name
        run_meta["npz_file"] = result.npz_path.name
        all_metadata.append(run_meta)
        summary["runs"].append({
            "run": result.run,
            "status": "ok",
            "epochs_saved": result.epochs_saved,
            "dropped_out_of_bounds": result.dropped_out_of_bounds,
            "shape": list(result.shape),
            "sfreq": result.sfreq,
            "fif": result.fif_path.name,
            "npz": result.npz_path.name,
            "metadata_csv": result.metadata_path.name,
        })

    if all_metadata:
        pd.concat(all_metadata, ignore_index=True).to_csv(output_dir / "all_runs_metadata.csv", index=False)
    summary["total_epochs"] = int(sum(r.get("epochs_saved", 0) for r in summary["runs"]))
    (output_dir / "crop_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
