"""
NOD-EEG dataset loader.
Loads preprocessed .fif continuous EEG, detailed events CSV, and stimulus image paths.
"""
import os
import glob
import mne
import pandas as pd


class NODLoader:
    """Loads NOD-EEG data for a single subject from the downloaded OpenNeuro layout."""

    def __init__(self, data_root, subject="sub-01"):
        self.data_root = data_root
        self.subject = subject

    # --- paths ---

    def _deriv_dir(self, kind):
        return os.path.join(self.data_root, "derivatives", "preprocessed", kind)

    def _raw_fif_pattern(self, session=None):
        sess = f"_{session}" if session else "_ses-*"
        return os.path.join(
            self._deriv_dir("raw"),
            f"{self.subject}{sess}_task-ImageNet_run-*_eeg_clean.fif",
        )

    def _epoch_path(self):
        return os.path.join(self._deriv_dir("epochs"), f"{self.subject}_eeg_epo.fif")

    def _events_csv_path(self):
        return os.path.join(
            self.data_root, "derivatives", "detailed_events",
            f"{self.subject}_events.csv",
        )

    def _stimuli_dir(self):
        return os.path.join(self.data_root, "stimuli", "ImageNet")

    # --- loaders ---

    def list_runs(self, session=None):
        """List available preprocessed .fif run files."""
        return sorted(glob.glob(self._raw_fif_pattern(session)))

    def load_raw(self, fif_path, preload=True):
        """Load a single preprocessed continuous .fif file."""
        return mne.io.read_raw_fif(fif_path, preload=preload, verbose=False)

    def load_epochs(self):
        """Load the concatenated epoch file for this subject."""
        path = self._epoch_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Epoch file missing: {path}")
        return mne.read_epochs(path, preload=True, verbose=False)

    def load_events(self):
        """Load the detailed events CSV."""
        path = self._events_csv_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Events CSV missing: {path}")
        return pd.read_csv(path)

    def get_image_path(self, synset_id, image_id):
        """Resolve a stimulus image path from synset and image IDs."""
        return os.path.join(self._stimuli_dir(), f"{synset_id}_{image_id}.JPEG")

    def summary(self):
        """Print a quick summary of available data."""
        runs = self.list_runs()
        print(f"Subject: {self.subject}")
        print(f"  Preprocessed runs: {len(runs)}")
        for r in runs:
            print(f"    {os.path.basename(r)}")

        epoch_path = self._epoch_path()
        print(f"  Epoch file: {'EXISTS' if os.path.exists(epoch_path) else 'MISSING'}")

        events_path = self._events_csv_path()
        if os.path.exists(events_path):
            df = pd.read_csv(events_path)
            print(f"  Events CSV: {len(df)} trials, columns: {list(df.columns)}")
        else:
            print(f"  Events CSV: MISSING")

        stim_dir = self._stimuli_dir()
        if os.path.exists(stim_dir):
            imgs = glob.glob(os.path.join(stim_dir, "*.JPEG"))
            print(f"  Stimulus images: {len(imgs)}")
        else:
            print(f"  Stimulus images dir: MISSING")
