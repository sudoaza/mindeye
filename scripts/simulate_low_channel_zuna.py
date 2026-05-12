import os
import argparse

# Adjust path to import channel mask logic
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mindseye.zuna.channel_simulation import simulate_low_channel_recording, EPOC_X_14

def main():
    parser = argparse.ArgumentParser(description="Simulate low channel (EPOC) recordings.")
    parser.add_argument("--subject", type=str, default="sub-01")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--raw-dir", type=str, default="data/raw/nod")
    parser.add_argument("--out-dir", type=str, default="data/processed/simulated_epoc14")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    run_str = f"run-{args.run:02d}"
    raw_path = os.path.join(args.raw_dir, "derivatives", "preprocessed", "raw", f"{args.subject}_ses-ImageNet01_task-ImageNet_{run_str}_eeg_clean.fif")
    out_path = os.path.join(args.out_dir, f"{args.subject}_task-ImageNet_{run_str}_epoc14_sim_eeg.fif")
    
    if not os.path.exists(raw_path):
        print(f"Error: Raw file not found at {raw_path}")
        print("This is a scaffold. Ensure you have downloaded the NOD dataset using make nod.")
        sys.exit(1)
        
    print(f"Simulating EPOC-14 headset for {args.subject} {run_str}...")
    simulate_low_channel_recording(raw_path, EPOC_X_14, out_path)
    
    print(f"Saved simulated EPOC-14 recording to: {out_path}")
    print("Next step: Run ZUNA inference on this simulated recording to assess recovery.")

if __name__ == "__main__":
    main()
