"""
Event-aligned cropper for ZUNA-normalized continuous EEG.

ZUNA output FIFs currently do not preserve stimulus annotations, so the safest
alignment path is:
1. read `stim_on` annotation times from the original raw NOD FIF,
2. convert those onset times to samples in the corresponding source FIF
   (ZUNA output, raw, or resample-only), and
3. crop a short semantic window from that signal.

Three crop modes are supported so all baseline-matrix conditions can be run:
  - "zuna"     : source FIF = ZUNA output (default)
  - "raw"      : source FIF = the raw preprocessed FIF at its native sfreq
  - "resample" : source FIF = the raw FIF resampled to target_sfreq (default 256 Hz)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal
import json
import warnings

import mne
import numpy as np
import pandas as pd


CropMode = Literal["zuna", "raw", "resample"]
RUNS_PER_SESSION = 8


def global_run_to_session_run(run: int) -> tuple[str, int]:
    """Map global ImageNet run ids to NOD session/local run ids."""
    if run < 1:
        raise ValueError(f"Run ids must be 1-based; got {run}")
    session_idx = ((int(run) - 1) // RUNS_PER_SESSION) + 1
    local_run = ((int(run) - 1) % RUNS_PER_SESSION) + 1
    return f"ImageNet{session_idx:02d}", local_run


@dataclass(frozen=True)
class CropConfig:
    """Configuration for semantic EEG crops."""

    tmin: float = -0.25
    tmax: float = 1.0
    expected_sfreq: float = 256.0
    event_name: str = "stim_on"
    mode: CropMode = "zuna"
    resample_sfreq: float = 256.0   # target sfreq for "resample" mode
    window_mode: str = "crop"
    has_event_marker: bool = False
    # Alignment guardrails (see crop_run_to_epochs). When the CSV carries an
    # `onset` column, onsets are matched by time; otherwise pairing is positional.
    onset_match_tol_s: float = 0.100   # max |csv_onset - fif_onset| to accept a time-join
    strict_align: bool = True          # hard-fail on count/time-alignment mismatch


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
    event_name: str = "stim_on",
) -> pd.DataFrame:
    """Filter the NOD detailed-events CSV to one session/run's stimulus events.

    The rows are returned sorted by the CSV's own ``onset`` column (when present)
    so their order matches the FIF annotation stream. Only stimulus rows are kept
    when a ``trial_type`` column is available, so non-stimulus markers do not shift
    the positional pairing with annotation onsets.
    """
    mask = events_df["run"].astype(int) == int(run)
    if session is not None and "session" in events_df.columns:
        mask &= events_df["session"].astype(str) == session
    # Keep only stimulus events so counts line up with `stim_on` annotations.
    if "trial_type" in events_df.columns:
        trial_type = events_df["trial_type"].astype(str).str.lower()
        stim_like = trial_type.isin({"stimulus", "stim", event_name.lower()})
        # Only apply the filter if it actually matches rows; some CSVs use other labels.
        if bool((mask & stim_like).any()):
            mask &= stim_like
    out = events_df.loc[mask].copy()
    if "onset" in out.columns:
        out = out.sort_values("onset", kind="stable")
    return out.reset_index(drop=True)


def build_mne_events_for_source(
    onset_seconds: np.ndarray,
    source_sfreq: float,
    n_times: int,
    config: CropConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert source onset times to MNE event rows in the source raw sample space.

    Returns `(events, valid_mask)`, where invalid rows would place the crop
    outside the source output duration.
    """
    events = np.zeros((len(onset_seconds), 3), dtype=int)
    events[:, 0] = np.rint(onset_seconds * source_sfreq).astype(int)
    events[:, 2] = 1

    start = events[:, 0] + int(round(config.tmin * source_sfreq))
    stop = events[:, 0] + int(round(config.tmax * source_sfreq))
    valid = (start >= 0) & (stop < n_times)
    return events, valid


def _load_source_raw(
    raw_fif_path: str | Path,
    source_fif_path: str | Path | None,
    config: CropConfig,
) -> mne.io.BaseRaw:
    """Load and optionally resample the source FIF depending on crop mode."""
    if config.mode == "zuna":
        if source_fif_path is None:
            raise ValueError("source_fif_path required for mode='zuna'")
        raw = _read_raw(source_fif_path, preload=True)
        if abs(raw.info["sfreq"] - config.expected_sfreq) > 1:
            print(
                f"  [warn] expected ~{config.expected_sfreq} Hz ZUNA output, "
                f"got {raw.info['sfreq']} Hz"
            )
        return raw

    if config.mode == "raw":
        return _read_raw(raw_fif_path, preload=True)

    if config.mode == "resample":
        raw = _read_raw(raw_fif_path, preload=True)
        if abs(raw.info["sfreq"] - config.resample_sfreq) > 0.5:
            print(
                f"  Resampling {Path(raw_fif_path).name} "
                f"from {raw.info['sfreq']} Hz → {config.resample_sfreq} Hz"
            )
            raw = raw.resample(config.resample_sfreq, npad="auto", verbose=False)
        return raw

    raise ValueError(f"Unknown crop mode: {config.mode!r}")


def crop_run_to_epochs(
    *,
    raw_fif_path: str | Path,
    source_fif_path: str | Path | None,
    events_df: pd.DataFrame,
    run: int,
    output_dir: str | Path,
    subject: str = "sub-01",
    session: str = "ImageNet01",
    config: CropConfig | None = None,
) -> CropResult:
    """Crop one source FIF into event-aligned semantic epochs.

    `source_fif_path` is only used in mode='zuna'; for 'raw' and 'resample'
    the raw_fif_path itself is the signal source.
    """
    config = config or CropConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_raw = _load_source_raw(raw_fif_path, source_fif_path, config)
    source_sfreq = float(source_raw.info["sfreq"])

    onset_seconds = stim_onsets_from_raw(raw_fif_path, config.event_name)
    event_offset_s = abs(config.tmin)
    anchor_sample = int(round(event_offset_s * source_sfreq))

    metadata = events_for_run(
        events_df, session=session, run=run, event_name=config.event_name
    )

    # --- Alignment guardrail (finding H2) -------------------------------------
    # `onset_seconds` come from the FIF annotation stream (sorted by sample).
    # `metadata` rows are sorted by the CSV's own `onset` column (when present),
    # so positional pairing below only holds if counts match and, when a CSV
    # onset column exists, the two onset series agree within tolerance.
    n_onsets, n_meta = len(onset_seconds), len(metadata)
    if n_onsets != n_meta:
        msg = (
            f"[Cropper] Onset/metadata count mismatch for {subject} {session} run {run}: "
            f"{n_onsets} FIF annotation onsets vs {n_meta} events-CSV rows. "
            f"Positional pairing would mislabel this run."
        )
        if config.strict_align:
            raise ValueError(msg)
        print(f"  [WARN] {msg} Proceeding with positional min() pairing.")

    n = min(n_onsets, n_meta)
    onset_seconds = onset_seconds[:n]
    metadata = metadata.iloc[:n].copy()

    # When the CSV exposes its own onset column, verify the time-join rather than
    # trusting positional order alone.
    if n > 0 and "onset" in metadata.columns:
        csv_onsets = metadata["onset"].to_numpy(dtype=float)
        # Align both series to their own run start so absolute-clock offsets
        # (annotation stream vs CSV) do not defeat the comparison.
        fif_rel = onset_seconds - onset_seconds[0]
        csv_rel = csv_onsets - csv_onsets[0]
        max_dev = float(np.max(np.abs(fif_rel - csv_rel))) if n > 1 else 0.0
        if max_dev > config.onset_match_tol_s:
            msg = (
                f"[Cropper] Onset time-join failed for {subject} {session} run {run}: "
                f"max |fif-csv| inter-onset deviation {max_dev:.3f}s "
                f"> tol {config.onset_match_tol_s:.3f}s. Labels may be shifted/misordered."
            )
            if config.strict_align:
                raise ValueError(msg)
            print(f"  [WARN] {msg}")
        else:
            print(f"  [Cropper] Onset time-join OK (max dev {max_dev:.3f}s over {n} events)")
    previous_counts = []
    future_counts = []
    other_counts = []
    
    for onset in onset_seconds:
        window_start = onset + config.tmin
        window_end = onset + config.tmax
        # previous: onset in [tmin, 0)
        prev = ((onset_seconds >= window_start) & (onset_seconds < onset)).sum()
        # future: onset in (0, tmax]
        fut = ((onset_seconds > onset) & (onset_seconds <= window_end)).sum()
        
        previous_counts.append(prev)
        future_counts.append(fut)
        other_counts.append(prev + fut)
    
    avg_prev = np.mean(previous_counts) if previous_counts else 0.0
    avg_fut = np.mean(future_counts) if future_counts else 0.0
    avg_others = np.mean(other_counts) if other_counts else 0.0
    
    print(f"  [Cropper] Window mode: {config.window_mode}")
    print(f"  [Cropper] Window: {config.tmin:.1f}s to +{config.tmax:.1f}s")
    print(f"  [Cropper] Event offset: {event_offset_s:.1f}s / sample {anchor_sample}")
    print(f"  [Cropper] Avg previous stimuli: {avg_prev:.2f}")
    print(f"  [Cropper] Avg future stimuli: {avg_fut:.2f}")
    print(f"  [Cropper] Avg total other stimuli: {avg_others:.2f}")
    
    if avg_others > 1.0:
        print(f"  [WARN] High label noise! windows contain multiple stimuli on average.")

    metadata.insert(0, "has_event_marker", config.has_event_marker)
    metadata.insert(0, "window_mode", config.window_mode)
    metadata.insert(0, "window_tmax", config.tmax)
    metadata.insert(0, "window_tmin", config.tmin)
    metadata.insert(0, "n_other_stimuli", other_counts)
    metadata.insert(0, "n_future_stimuli", future_counts)
    metadata.insert(0, "n_previous_stimuli", previous_counts)
    metadata.insert(0, "anchor_sample", anchor_sample)
    metadata.insert(0, "event_offset_s", event_offset_s)
    metadata.insert(0, "stim_onset_sec", onset_seconds)
    metadata.insert(0, "source_mode", config.mode)

    mne_events, valid = build_mne_events_for_source(
        onset_seconds, source_sfreq, source_raw.n_times, config
    )
    metadata = metadata.loc[valid].reset_index(drop=True)
    mne_events = mne_events[valid]

    epochs = mne.Epochs(
        source_raw,
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

    mode_tag = config.mode
    stem = f"{subject}_ses-{session}_run-{run:02d}_{mode_tag}_semantic"
    fif_path = output_dir / f"{stem}-epo.fif"
    npz_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{subject}_ses-{session}_run-{run:02d}_metadata.csv"

    epochs.save(fif_path, overwrite=True, verbose=False)
    np.savez_compressed(
        npz_path,
        eeg=data,
        sfreq=source_sfreq,
        times=epochs.times,
        ch_names=np.array(epochs.ch_names, dtype=object),
    )
    metadata.to_csv(metadata_path, index=False)

    return CropResult(
        run=int(run),
        epochs_saved=int(len(epochs)),
        dropped_out_of_bounds=int((~valid).sum()),
        shape=tuple(int(x) for x in data.shape),
        sfreq=source_sfreq,
        fif_path=fif_path,
        npz_path=npz_path,
        metadata_path=metadata_path,
    )


# ---------------------------------------------------------------------------
# Legacy alias — kept so existing call-sites using crop_zuna_runs still work
# ---------------------------------------------------------------------------

def crop_zuna_run_to_epochs(
    *,
    raw_fif_path,
    zuna_fif_path,
    events_df,
    run,
    output_dir,
    subject="sub-01",
    session="ImageNet01",
    config=None,
) -> CropResult:
    """Backward-compatible wrapper; delegates to crop_run_to_epochs with mode='zuna'."""
    cfg = config or CropConfig()
    if cfg.mode != "zuna":
        cfg = CropConfig(
            tmin=cfg.tmin, tmax=cfg.tmax,
            expected_sfreq=cfg.expected_sfreq,
            event_name=cfg.event_name,
            mode="zuna",
        )
    return crop_run_to_epochs(
        raw_fif_path=raw_fif_path,
        source_fif_path=zuna_fif_path,
        events_df=events_df,
        run=run,
        output_dir=output_dir,
        subject=subject,
        session=session,
        config=cfg,
    )


def crop_runs(
    *,
    raw_dir: str | Path,
    source_dir: str | Path | None,
    events_csv: str | Path,
    output_dir: str | Path,
    subject: str = "sub-01",
    session: str = "ImageNet01",
    runs: Iterable[int] = range(1, 6),
    config: CropConfig | None = None,
) -> dict:
    """Crop a batch of runs and write a summary JSON.

    For mode='zuna', `source_dir` must contain matching ZUNA FIF files.
    For mode='raw' or 'resample', `source_dir` is ignored (raw_dir is used directly).
    """
    raw_dir = Path(raw_dir)
    source_dir = Path(source_dir) if source_dir else None
    output_dir = Path(output_dir)
    events_df = pd.read_csv(events_csv)
    config = config or CropConfig()

    all_metadata = []
    summary = {
        "subject": subject,
        "session": session,
        "mode": config.mode,
        "tmin": config.tmin,
        "tmax": config.tmax,
        "runs": [],
    }

    for run in runs:
        # Treat requested run ids as global ImageNet runs across sessions.
        # Example: global run 9 is ImageNet02/run-01 on disk and in events.csv.
        actual_session, local_run = global_run_to_session_run(int(run))
        raw_fif = raw_dir / f"{subject}_ses-{actual_session}_task-ImageNet_run-{local_run:02d}_eeg_clean.fif"

        # Locate source FIF (only needed for zuna mode)
        source_fif: Path | None = None
        if config.mode == "zuna":
            if source_dir is None:
                raise ValueError("source_dir required for mode='zuna'")
            # Accept both naming conventions (real vs mock ZUNA output)
            candidates = [
                source_dir / f"{subject}_ses-{actual_session}_task-ImageNet_run-{local_run:02d}_eeg_clean.fif",
                source_dir / f"{subject}_ses-{actual_session}_task-ImageNet_run-{local_run:02d}_eeg_clean_zuna_mock.fif",
            ]
            source_fif = next((p for p in candidates if p.exists()), None)
            if source_fif is None:
                summary["runs"].append({
                    "run": int(run),
                    "session": actual_session,
                    "local_run": int(local_run),
                    "status": "missing_zuna_fif",
                    "raw_exists": raw_fif.exists(),
                })
                continue

        if not raw_fif.exists():
            summary["runs"].append({
                "run": int(run),
                "session": actual_session,
                "local_run": int(local_run),
                "status": "missing_raw_fif",
                "raw_exists": False,
            })
            continue

        result = crop_run_to_epochs(
            raw_fif_path=raw_fif,
            source_fif_path=source_fif,
            events_df=events_df,
            run=int(local_run),
            output_dir=output_dir,
            subject=subject,
            session=actual_session,
            config=config,
        )
        run_meta = pd.read_csv(result.metadata_path)
        run_meta["epoch_file"] = result.fif_path.name
        run_meta["npz_file"] = result.npz_path.name
        run_meta["session"] = actual_session
        run_meta["local_run"] = result.run
        run_meta["global_run"] = int(run)
        # Keep the canonical training split column as the global run id.
        run_meta["run"] = int(run)
        # Record subject identity so downstream caching/training can distinguish
        # subjects (used for unique sample_ids and subject FiLM). Parse "sub-0X" -> X.
        run_meta["subject_label"] = subject
        try:
            run_meta["subject"] = int(str(subject).split("-")[-1])
        except (ValueError, IndexError):
            run_meta["subject"] = 1
        all_metadata.append(run_meta)
        summary["runs"].append({
            "run": int(run),
            "session": actual_session,
            "local_run": result.run,
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
        pd.concat(all_metadata, ignore_index=True).to_csv(
            output_dir / "all_runs_metadata.csv", index=False
        )
    summary["total_epochs"] = int(sum(r.get("epochs_saved", 0) for r in summary["runs"]))
    (output_dir / "crop_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# Backward-compatible alias
def crop_zuna_runs(
    *,
    raw_dir,
    zuna_dir,
    events_csv,
    output_dir,
    subject="sub-01",
    session="ImageNet01",
    runs=range(1, 6),
    config=None,
) -> dict:
    cfg = config or CropConfig()
    return crop_runs(
        raw_dir=raw_dir,
        source_dir=zuna_dir,
        events_csv=events_csv,
        output_dir=output_dir,
        subject=subject,
        session=session,
        runs=runs,
        config=CropConfig(
            tmin=cfg.tmin, tmax=cfg.tmax,
            expected_sfreq=cfg.expected_sfreq,
            event_name=cfg.event_name,
            mode="zuna",
        ),
    )
