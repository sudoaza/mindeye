#!/usr/bin/env python3
"""Wrapper for training EEG→decode_unit encoder (Phase 12B canonical entry point).

Mode controls the training objective:
  contrastive_only   (default) — single-head InfoNCE on decode_unit. Best retrieval.
  dual_fixed_norm    — dual-head: InfoNCE + raw MSE with fixed mean-norm scaling.
  dual_learned_norm  — dual-head: InfoNCE + raw MSE + learned norm head.

Any additional flags are passed through to train_eeg_clip.py.
"""
import sys
import os
import argparse


def main():
    # Parse only our --mode flag; pass everything else to train_eeg_clip.py
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--mode",
        choices=("contrastive_only", "dual_fixed_norm", "dual_learned_norm"),
        default="contrastive_only",
        help="Training objective (default: contrastive_only)",
    )
    known, rest = p.parse_known_args()

    # Fixed Phase 12 defaults
    cmd = [
        sys.executable, "scripts/train_eeg_clip.py",
        "--common-embeddings", "data/processed/clip_embeddings/decode_common_embeddings.pt",
        "--loss", "contrastive",
        "--model", "temporal_attn_small",
        "--target-space", "decode_unit",
        "--add-event-marker",
        "--augment-eeg",
        "--metadata", "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv",
        "--epochs-dir", "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40",
        "--window-mode", "tight1s",
    ]

    # Mode → flags
    if known.mode == "dual_fixed_norm":
        cmd += ["--dual-head", "--use-fixed-mean-norm"]
    elif known.mode == "dual_learned_norm":
        cmd += ["--dual-head"]
    # contrastive_only: no extra flags

    cmd += rest
    print("Running:", " ".join(cmd))
    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
