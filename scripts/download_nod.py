"""
Download a minimal subset of NOD-EEG (ds005811) from OpenNeuro.
Downloads one subject's epoch .fif, events CSV, and stimulus metadata.

Usage: venv/bin/python scripts/download_nod.py
"""
import os
import openneuro


DATASET_ID = "ds005811"
SUBJECT = "sub-01"
OUTPUT_DIR = "data/raw/nod"

# Each pattern is downloaded separately. openneuro-py does not support globs
# in the middle of filenames, but does support directory-level wildcards.
INCLUDE_PATTERNS = [
    # Continuous preprocessed run (one run for testing)
    f"derivatives/preprocessed/raw/{SUBJECT}_ses-ImageNet01_task-ImageNet_run-01_eeg_clean.fif",
    # Concatenated epoch file (all trials for this subject)
    f"derivatives/preprocessed/epochs/{SUBJECT}_eeg_epo.fif",
    # Detailed events CSV
    f"derivatives/detailed_events/{SUBJECT}_events.csv",
    # Stimulus metadata tables
    "stimuli/metadata/*",
    # Dataset description
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
]


def download():
    """Download NOD-EEG subset using openneuro Python API."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for pattern in INCLUDE_PATTERNS:
        print(f"\n=== Downloading: {pattern} ===")
        try:
            openneuro.download(
                dataset=DATASET_ID,
                target_dir=OUTPUT_DIR,
                include=[pattern],
            )
        except Exception as e:
            print(f"  Warning: {e}")
            print("  Continuing with next pattern...")

    print(f"\nDone! NOD-EEG subset downloaded to {OUTPUT_DIR}/")


if __name__ == "__main__":
    download()
