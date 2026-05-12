import numpy as np

# Typical 10-20 or 10-10 channel namings
EPOC_X_14 = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

OCCIPITAL_14 = [
    "O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "POz",
    "P1", "P2", "P3", "P4", "P5", "P6"
]

def mask_eeg_channels(raw_eeg_data, current_channel_names, target_subset):
    """
    Masks out channels that are not in the target subset (zeroes them out) 
    or extracts only the target subset, depending on ZUNA's requirement.
    For ZUNA upsampling, it's typical to provide the full shape but with 
    unmeasured channels zeroed, or provide a reduced shape with coordinates.
    
    Returns:
        masked_data: EEG data with non-target channels zeroed.
        mask_indices: The indices of the kept channels.
    """
    # Normalize naming for matching (case-insensitive usually preferred)
    current_lower = [c.lower() for c in current_channel_names]
    target_lower = [t.lower() for t in target_subset]
    
    keep_indices = []
    for i, ch in enumerate(current_lower):
        if ch in target_lower:
            keep_indices.append(i)
            
    masked_data = np.zeros_like(raw_eeg_data)
    if len(keep_indices) > 0:
        masked_data[keep_indices, :] = raw_eeg_data[keep_indices, :]
        
    return masked_data, keep_indices

def simulate_low_channel_recording(raw_fif_path, target_channels, output_path):
    """
    Loads a raw FIF, applies the channel mask to simulate a low-density headset,
    and saves the simulated FIF.
    """
    import mne
    raw = mne.io.read_raw_fif(raw_fif_path, preload=True)
    
    current_names = raw.ch_names
    masked_data, kept = mask_eeg_channels(raw.get_data(), current_names, target_channels)
    
    print(f"Kept {len(kept)} channels out of {len(current_names)}.")
    
    # Create new raw object with masked data
    simulated_raw = mne.io.RawArray(masked_data, raw.info)
    
    # Save
    simulated_raw.save(output_path, overwrite=True)
    return simulated_raw
