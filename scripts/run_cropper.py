"""
Extracts event-aligned semantic windows (1.25s) from ZUNA-processed continuous EEG.
Usage: venv/bin/python scripts/run_cropper.py
"""
import os
import sys
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mindseye.datasets.nod import NODLoader

# Input data
DATA_ROOT = "data/raw/nod"
ZUNA_OUTPUT_DIR = "data/processed/zuna_output/4_fif_output"

# Output for semantic embeddings phase
FINAL_EPOCHS_OUT = "data/processed/semantic_epochs"

def main():
    if not os.path.exists(ZUNA_OUTPUT_DIR):
        print(f"Error: No ZUNA output found at {ZUNA_OUTPUT_DIR}")
        sys.exit(1)

    os.makedirs(FINAL_EPOCHS_OUT, exist_ok=True)
    loader = NODLoader(DATA_ROOT, subject="sub-01")
    
    print("Loading events CSV...")
    events_df = loader.load_events()
    
    # Filter events for Run 01
    run1_events = events_df[
        (events_df["session"] == "ImageNet01") & 
        (events_df["run"] == 1)
    ].reset_index(drop=True)
    
    print(f"Found {len(run1_events)} image presentation events in CSV for this run.")

    mock_filename = "sub-01_ses-ImageNet01_task-ImageNet_run-01_eeg_clean_zuna_mock.fif"
    zuna_fif_path = os.path.join(ZUNA_OUTPUT_DIR, mock_filename)
    
    if not os.path.exists(zuna_fif_path):
        print(f"Error: Could not find ZUNA output file: {zuna_fif_path}")
        sys.exit(1)

    print(f"\nExtracting semantic windows from: {mock_filename}")
    
    # 1. Load the continuous ZUNA data
    raw = mne.io.read_raw_fif(zuna_fif_path, preload=True, verbose=False)
    
    # 2. Find stimulus triggers in the continuous EEG
    try:
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        print(f"Found {len(events)} total events from annotations.")
        print(f"Event types found: {event_dict}")
        
        # We need exactly `len(run1_events)` triggers. Let's find which event code matches!
        target_count = len(run1_events)
        event_counts = {name: sum(events[:, 2] == code) for name, code in event_dict.items()}
        
        target_event_code = None
        for name, count in event_counts.items():
            if count == target_count:
                print(f"Match found! Event '{name}' occurs exactly {target_count} times.")
                target_event_code = event_dict[name]
                break
                
        if target_event_code is not None:
            # Filter the events array to only include the image presentations
            events = events[events[:, 2] == target_event_code]
        else:
            print(f"Warning: Could not find an event type that occurs exactly {target_count} times.")
            print(f"Event counts: {event_counts}")
            # Fallback: Just take the first ones if we must
            min_len = min(len(events), len(run1_events))
            events = events[:min_len]
            run1_events = run1_events.iloc[:min_len]
            
    except Exception as e:
        print(f"Could not read annotations: {e}")
        events = []
        
    if len(events) == 0:
        print("Warning: No annotations found.")
        sys.exit(1)
    
    # Check if counts match
    if len(events) != len(run1_events):
        print("Warning: Event count mismatch between CSV and EEG triggers!")
        # We will truncate to the shortest to proceed safely
        min_len = min(len(events), len(run1_events))
        events = events[:min_len]
        run1_events = run1_events.iloc[:min_len]
        
    # 3. Create the Epochs object directly (-0.25s to 1.0s)
    print("Slicing out 1.25s epochs around the triggers...")
    epochs = mne.Epochs(
        raw, 
        events, 
        tmin=-0.25, 
        tmax=1.0, 
        baseline=None, 
        metadata=run1_events,  # Attach all the ImageNet labels!
        preload=True,
        verbose=False
    )
    
    # 4. Save
    out_path = os.path.join(FINAL_EPOCHS_OUT, "sub-01_ses-ImageNet01_run-01_semantic-epo.fif")
    epochs.save(out_path, overwrite=True, verbose=False)
    
    print(f"\nSuccess! Saved cropped semantic epochs to: {out_path}")
    print(f"Shape: {epochs.get_data().shape} (trials, channels, samples)")
    print(f"Metadata rows attached: {len(epochs.metadata)}")

if __name__ == "__main__":
    main()
