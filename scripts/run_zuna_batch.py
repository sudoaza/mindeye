"""
Run the ZUNA batch offline pipeline on downloaded NOD-EEG continuous runs.
Usage: venv/bin/python scripts/run_zuna_batch.py
"""
import os
import sys
import shutil
import glob
import mne

mne.set_log_level('ERROR')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mindseye.zuna.offline_pipeline import run_zuna_offline

INPUT_DIR = "data/raw/nod/derivatives/preprocessed/raw"
OUTPUT_DIR = "data/processed/zuna_output"
TARGET_CHANNELS = None 

def mock_zuna_offline(input_fif_dir, working_dir):
    """
    Mocks ZUNA locally. It resamples the raw data to 256Hz (like ZUNA does)
    and saves it to the final output directory so the rest of the pipeline
    can proceed.
    """
    print("\n=== [MOCK] ZUNA Offline Pipeline ===")
    fif_output_dir = os.path.join(working_dir, "4_fif_output")
    os.makedirs(fif_output_dir, exist_ok=True)
    
    fif_files = glob.glob(os.path.join(input_fif_dir, "*.fif"))
    for fif_path in fif_files:
        print(f"Mocking ZUNA for {os.path.basename(fif_path)}")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        
        # ZUNA operates at 256Hz
        if raw.info['sfreq'] != 256:
            print("  Resampling to 256Hz...")
            raw.resample(256.0)
            
        out_name = os.path.basename(fif_path).replace(".fif", "_zuna_mock.fif")
        out_path = os.path.join(fif_output_dir, out_name)
        raw.save(out_path, overwrite=True, verbose=False)
        
    print(f"\n[MOCK] Finished. Output in: {fif_output_dir}")
    return fif_output_dir


def main():
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"Error: No continuous runs found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Starting ZUNA batch pipeline on {INPUT_DIR}")
    
    # Swapped to MOCK for local laptop development to avoid 35GB RAM OOM
    out_dir = mock_zuna_offline(
        input_fif_dir=INPUT_DIR,
        working_dir=OUTPUT_DIR,
    )
    
    # REAL ZUNA CALL (Uncomment when running on GPU / RunPod)
    # out_dir = run_zuna_offline(
    #     input_fif_dir=INPUT_DIR,
    #     working_dir=OUTPUT_DIR,
    #     target_channels=TARGET_CHANNELS,
    #     gpu_device=0,
    #     diffusion_steps=15
    # )
    
if __name__ == "__main__":
    main()
