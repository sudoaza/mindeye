"""
Channel montage utilities for ZUNA compatibility.
ZUNA requires .fif files to have a channel montage with 3D electrode positions.
"""
import mne


CANONICAL_32CH = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "O2",
]


def ensure_montage(raw, montage_name="standard_1005"):
    """
    Ensure a Raw object has a channel montage with 3D positions.
    If missing, set the standard 10-05 montage.
    """
    if raw.get_montage() is not None:
        return raw

    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing="warn")
    return raw


def get_channel_positions(raw):
    """Extract 3D channel positions from a Raw object's montage."""
    montage = raw.get_montage()
    if montage is None:
        return None
    pos = montage.get_positions()
    ch_pos = pos.get("ch_pos", {})
    return {name: tuple(xyz) for name, xyz in ch_pos.items()}
