import os
import argparse

# Adjust path to import channel mask logic
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mindseye.zuna.channel_simulation import simulate_low_channel_recording, EPOC_X_14

def main():
    parser = argparse.ArgumentParser(description="Simulate low channel (EPOC) recordings.")
    parser.add_argument("--subject", type=str, default="sub-01")
    parser.add_argument("--runs", type=int, nargs="+", default=[1], help="List of runs to process")
    parser.add_argument("--raw-dir", type=str, default="data/raw/nod")
    parser.add_argument("--out-dir", type=str, default="data/processed/simulated_epoc14")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    for run in args.runs:
        # Map global run (1-32) to session and local run
        session_idx = ((int(run) - 1) // 8) + 1
        local_run = ((int(run) - 1) % 8) + 1
        session_str = f"ImageNet{session_idx:02d}"
        run_str = f"run-{local_run:02d}"
        
        raw_path = os.path.join(
            args.raw_dir, "derivatives", "preprocessed", "raw",
            f"{args.subject}_ses-{session_str}_task-ImageNet_{run_str}_eeg_clean.fif"
        )
        
        if not os.path.exists(raw_path):
            print(f"Warning: Raw file not found for global run {run} at {raw_path}")
            continue
            
        out_name = os.path.basename(raw_path)
        out_path = os.path.join(args.out_dir, out_name)
        
        print(f"Simulating EPOC-14 headset for global run {run} ({session_str}/{run_str})...")
        simulate_low_channel_recording(raw_path, EPOC_X_14, out_path)
        print(f"Saved simulated EPOC-14 recording to: {out_path}")
        
    print("Next step: Run ZUNA inference on this simulated recording to assess recovery.")

if __name__ == "__main__":
    main()
