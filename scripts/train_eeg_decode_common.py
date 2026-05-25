#!/usr/bin/env python3
import sys
import os

def main():
    # Pass along any additional arguments, but set the defaults for Phase 12A
    cmd = [
        sys.executable, "scripts/train_eeg_clip.py",
        "--common-embeddings", "data/processed/clip_embeddings/decode_common_embeddings.pt",
        "--loss", "contrastive",
        "--metadata", "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv",
        "--epochs-dir", "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40",
        "--window-mode", "tight1s"
    ] + sys.argv[1:]
    
    print("Running:", " ".join(cmd))
    os.execvp(sys.executable, cmd)

if __name__ == "__main__":
    main()
