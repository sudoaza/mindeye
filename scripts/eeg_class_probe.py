#!/usr/bin/env python3
"""EEG signal sanity probe: can ZUNA latents linearly decode the stimulus class?

Before investing more in EEG->vision retrieval, verify the ZUNA latents carry
*any* stimulus-discriminative information. This fits a linear (logistic-regression)
probe from mean-pooled ZUNA latents to the ImageNet class_id (1000-way), trained on
train runs and evaluated on held-out val runs, and compares:
  - real labels  vs  shuffled-label control (the honest chance floor)
  - top-1 / top-5 accuracy vs analytic chance (1/1000, ~5/1000)

If real >> shuffled/chance, the EEG carries decodable stimulus info and the
retrieval bottleneck is the bridge/target. If real ~= chance, the signal itself
is the ceiling and no bridge will work.

Usage:
    PYTHONPATH=src python scripts/eeg_class_probe.py \
        --latents-dir data/processed/zuna_latents/cohort9_runs01_32 \
        --layer-name post_mmd \
        --train-runs 1-6 --val-runs 7-8 \
        --latent-tc-start 15 --latent-tc-end 31
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_runs_spec(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--layer-name", default="post_mmd")
    p.add_argument("--train-runs", default="1-6")
    p.add_argument("--val-runs", default="7-8")
    p.add_argument("--n-channels", type=int, default=62)
    p.add_argument("--tc", type=int, default=40)
    p.add_argument("--latent-tc-start", type=int, default=15)
    p.add_argument("--latent-tc-end", type=int, default=31)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cache_dir = args.latents_dir
    meta = torch.load(os.path.join(cache_dir, "metadata.pt"), map_location="cpu")
    layer_path = os.path.join(cache_dir, f"latents_{args.layer_name}.pt")
    layer_dict = torch.load(layer_path, map_location="cpu")
    print(f"Loaded {len(meta)} records; layer '{args.layer_name}' with {len(layer_dict)} latents")

    train_runs = parse_runs_spec(args.train_runs)
    val_runs = parse_runs_spec(args.val_runs)

    def pool(latent: torch.Tensor) -> np.ndarray:
        # latent: [seq, latent_dim] flattened as [n_channels*tc, D]. Crop the time
        # window (same convention as the QFormer) then mean-pool over channels+time.
        lat = latent.float()
        D = lat.shape[-1]
        lat = lat.reshape(args.n_channels, args.tc, D)[:, args.latent_tc_start:args.latent_tc_end, :]
        return lat.reshape(-1, D).mean(dim=0).numpy()

    Xtr, ytr, Xva, yva = [], [], [], []
    for r in meta:
        s_id = r["sample_id"]
        if s_id not in layer_dict:
            continue
        run = int(r["run_id"])
        cls = r["class_id"]
        if run in train_runs:
            Xtr.append(pool(layer_dict[s_id])); ytr.append(cls)
        elif run in val_runs:
            Xva.append(pool(layer_dict[s_id])); yva.append(cls)

    Xtr = np.stack(Xtr); Xva = np.stack(Xva)
    ytr = np.array(ytr); yva = np.array(yva)
    n_classes = len(set(ytr.tolist()) | set(yva.tolist()))
    print(f"Train {Xtr.shape} / Val {Xva.shape} | feat_dim={Xtr.shape[1]} | n_classes={n_classes}")
    print(f"Analytic chance: top-1={1.0/n_classes:.4%}  top-5={min(5,n_classes)/n_classes:.4%}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr); Xva_s = scaler.transform(Xva)

    def fit_eval(y_train, tag: str):
        clf = LogisticRegression(C=args.C, max_iter=args.max_iter, n_jobs=-1)
        clf.fit(Xtr_s, y_train)
        proba = clf.predict_proba(Xva_s)
        classes = clf.classes_
        # map true val labels to column indices (unseen-in-train labels -> never correct)
        col = {c: i for i, c in enumerate(classes)}
        top1 = top5 = 0
        for i, true in enumerate(yva):
            order = np.argsort(proba[i])[::-1]
            top_classes = classes[order[:5]]
            if len(top_classes) and top_classes[0] == true:
                top1 += 1
            if true in set(top_classes.tolist()):
                top5 += 1
        n = len(yva)
        print(f"  [{tag}] top-1={top1/n:.4%}  top-5={top5/n:.4%}  (n_val={n})")
        return top1 / n, top5 / n

    print("\n=== Linear probe: ZUNA latent -> class_id ===")
    real1, real5 = fit_eval(ytr, "REAL labels")

    rng = np.random.default_rng(args.seed)
    y_shuf = ytr.copy(); rng.shuffle(y_shuf)
    shuf1, shuf5 = fit_eval(y_shuf, "SHUFFLED labels (control)")

    print("\n=== Verdict ===")
    print(f"  top-1: real={real1:.4%}  shuffled={shuf1:.4%}  delta={real1-shuf1:+.4%}")
    print(f"  top-5: real={real5:.4%}  shuffled={shuf5:.4%}  delta={real5-shuf5:+.4%}")
    if real5 > 2 * max(shuf5, 5.0 / n_classes):
        print("  -> EEG carries decodable class signal; retrieval bottleneck is the bridge/target.")
    else:
        print("  -> EEG shows ~chance class decoding; the signal itself is the ceiling.")


if __name__ == "__main__":
    main()
