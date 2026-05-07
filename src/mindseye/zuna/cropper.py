"""
Event-aligned cropper: extracts stimulus-relevant EEG windows from ZUNA output.
ZUNA produces 5s epochs at 256Hz. This module finds which 5s epoch contains
each stimulus onset, then crops a shorter task-relevant window.
"""
import mne
import numpy as np
import pandas as pd


def crop_stimulus_windows(
    zuna_fif_path,
    events_df,
    run_id,
    crop_tmin=0.0,
    crop_tmax=1.25,
    sfreq=256,
):
    """
    Extract stimulus-aligned crops from a ZUNA-reconstructed .fif file.

    Args:
        zuna_fif_path: path to ZUNA output .fif
        events_df: DataFrame with at least 'onset_time', 'session', 'run' columns
        run_id: which run to filter events for
        crop_tmin: seconds before stimulus onset (0.0 or -0.25)
        crop_tmax: seconds after stimulus onset (1.0 or 1.25)
        sfreq: expected sampling rate (256 for ZUNA output)

    Returns:
        crops: ndarray of shape (n_trials, n_channels, n_samples)
        trial_meta: DataFrame with metadata for each crop
    """
    raw = mne.io.read_raw_fif(zuna_fif_path, preload=True, verbose=False)

    actual_sfreq = raw.info["sfreq"]
    if abs(actual_sfreq - sfreq) > 1:
        print(f"Warning: expected {sfreq}Hz, got {actual_sfreq}Hz")

    # Filter events for this run
    run_events = events_df[events_df["run"] == run_id].copy()
    if len(run_events) == 0:
        print(f"No events found for run {run_id}")
        return np.array([]), pd.DataFrame()

    n_samples = int((crop_tmax - crop_tmin) * actual_sfreq)
    data = raw.get_data()  # (channels, total_samples)

    crops = []
    valid_trials = []

    for _, trial in run_events.iterrows():
        onset_sec = trial.get("onset_time", trial.get("onset", None))
        if onset_sec is None:
            continue

        start_sample = int((onset_sec + crop_tmin) * actual_sfreq)
        end_sample = start_sample + n_samples

        if start_sample < 0 or end_sample > data.shape[1]:
            continue  # skip if window falls outside recording

        crop = data[:, start_sample:end_sample]
        crops.append(crop)
        valid_trials.append(trial)

    if not crops:
        return np.array([]), pd.DataFrame()

    crops = np.stack(crops)  # (n_trials, n_channels, n_samples)
    trial_meta = pd.DataFrame(valid_trials).reset_index(drop=True)

    print(f"Cropped {len(crops)} trials from {zuna_fif_path}")
    print(f"  Shape: {crops.shape} (trials, channels, samples)")

    return crops, trial_meta
