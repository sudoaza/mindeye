import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm


# Coarse ImageNet-synset -> high-level category via WordNet hypernym lexnames.
# EEG is expected to carry coarse semantics (animal/vehicle/food...) more than fine
# identity, so a category probe is the sensitive test for "any semantic signal".
_CAT_CACHE: dict[str, str] = {}


def synset_category(class_id: str) -> str:
    """Map an ImageNet WordNet id (e.g. 'n01983481') to its lexicographer file
    name (e.g. 'noun.animal'), a ~20-way coarse semantic category."""
    if class_id in _CAT_CACHE:
        return _CAT_CACHE[class_id]
    try:
        from nltk.corpus import wordnet as wn
        offset = int(class_id[1:])
        ss = wn.synset_from_pos_and_offset("n", offset)
        cat = ss.lexname()
    except Exception:
        cat = "unknown"
    _CAT_CACHE[class_id] = cat
    return cat


def category_probe(X: torch.Tensor, cats: list[str], kf) -> tuple[float, float, int]:
    """5-fold logistic-regression accuracy predicting coarse category from X.
    Returns (mean_acc, majority_baseline, n_categories)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import collections

    Xn = X.cpu().numpy()
    y = np.array(cats)
    counts = collections.Counter(y.tolist())
    majority = max(counts.values()) / len(y)
    accs = []
    for tr, te in kf.split(Xn):
        sc = StandardScaler().fit(Xn[tr])
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(sc.transform(Xn[tr]), y[tr])
        accs.append(float((clf.predict(sc.transform(Xn[te])) == y[te]).mean()))
    return float(np.mean(accs)), float(majority), len(counts)


def ridge_regression_fit_predict(X_train, Y_train, X_test, alpha=1.0):
    # Closed-form Ridge Regression: W = (X_train^T X_train + alpha I)^(-1) X_train^T Y_train
    # X_train: [N, D_in], Y_train: [N, D_out]
    N, D_in = X_train.shape
    device = X_train.device
    
    # Add bias term by appending a column of ones
    ones_train = torch.ones(N, 1, device=device)
    X_train_bias = torch.cat([X_train, ones_train], dim=1)
    
    N_test = X_test.shape[0]
    ones_test = torch.ones(N_test, 1, device=device)
    X_test_bias = torch.cat([X_test, ones_test], dim=1)
    
    # Solve W
    XTX = X_train_bias.T @ X_train_bias
    I = torch.eye(D_in + 1, device=device)
    W = torch.linalg.solve(XTX + alpha * I, X_train_bias.T @ Y_train)
    
    # Predict
    Y_pred = X_test_bias @ W
    return Y_pred

def compute_cosine_sim(Y_pred, Y_test):
    # Y_pred: [N, D], Y_test: [N, D]
    pred_norm = Y_pred / (Y_pred.norm(dim=-1, keepdim=True) + 1e-8)
    test_norm = Y_test / (Y_test.norm(dim=-1, keepdim=True) + 1e-8)
    return (pred_norm * test_norm).sum(dim=-1).mean().item()

def compute_top10_retrieval(Y_pred, Y_test):
    # Predict over test fold using test fold targets as the candidate bank
    # Y_pred: [N, D], Y_test: [N, D]
    pred_norm = Y_pred / (Y_pred.norm(dim=-1, keepdim=True) + 1e-8)
    test_norm = Y_test / (Y_test.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Cosine matrix: [N, N]
    similarity_matrix = pred_norm @ test_norm.T
    
    # Top 10 indices
    top10_indices = similarity_matrix.topk(10, dim=-1).indices
    
    # Check if correct index (diagonal) is in top 10
    correct = 0
    N = Y_pred.shape[0]
    for i in range(N):
        if i in top10_indices[i]:
            correct += 1
    return correct / N

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ridge-probe each ZUNA layer -> visual targets (finds which layer carries retrievable signal).")
    ap.add_argument("--cache-dir", default="/workspace/mindeye/data/processed/zuna_latents/sub01_layersweep",
                    help="Dir with split-file cache: latents_<layer>.pt + metadata.pt")
    ap.add_argument("--rae-path", default="/workspace/mindeye/data/processed/rae_embeddings/rae_dinov2_base_all.pt")
    ap.add_argument("--n-channels", type=int, default=62)
    ap.add_argument("--tc", type=int, default=40)
    ap.add_argument("--onset-lo", type=int, default=15, help="onset-window start tc (post-onset crop)")
    ap.add_argument("--onset-hi", type=int, default=31, help="onset-window end tc")
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    import os
    meta = torch.load(os.path.join(args.cache_dir, "metadata.pt"), map_location="cpu")
    print(f"Loaded {len(meta)} metadata records from {args.cache_dir}")

    print(f"Loading DINO/RAE embeddings from {args.rae_path}...")
    rae_data = torch.load(args.rae_path, map_location="cpu")
    rae_global = rae_data.get("image_id_to_rae_global") or rae_data.get("rae_global")
    rae_unit = rae_data.get("image_id_to_rae_unit") or rae_data.get("rae_unit")
    dino_cls = rae_data.get("image_id_to_dino_cls") or rae_data.get("dino_cls")

    target_spaces = {"DINO-Unit": rae_unit, "DINO-Global": rae_global}
    if dino_cls is not None:
        target_spaces["DINO-CLS"] = dino_cls

    layers = ["layer_4", "layer_8", "layer_12", "layer_16", "pre_mmd", "post_mmd"]
    pooling_modes = ["all", "onset"]

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Records with targets available
    valid_records = [r for r in meta if r["image_id"] in rae_unit]
    print(f"Found {len(valid_records)} records with target embeddings.")

    for layer in layers:
        layer_path = os.path.join(args.cache_dir, f"latents_{layer}.pt")
        if not os.path.exists(layer_path):
            print(f"[skip] {layer}: {layer_path} missing")
            continue
        layer_dict = torch.load(layer_path, map_location="cpu")
        for pool_mode in pooling_modes:
            X_list = []
            for r in valid_records:
                latent = layer_dict[r["sample_id"]].float()  # [2480, dim]
                dim = latent.shape[1]
                latent_reshaped = latent.view(args.n_channels, args.tc, dim)
                if pool_mode == "all":
                    x = latent_reshaped.mean(dim=(0, 1))  # [dim]
                else:
                    x = latent_reshaped[:, args.onset_lo:args.onset_hi, :].mean(dim=(0, 1))
                X_list.append(x)
            X = torch.stack(X_list).to(device)

            # Coarse-category probe (semantic signal detector) — once per pooled X.
            cats = [synset_category(r["class_id"]) for r in valid_records]
            cat_acc, cat_base, n_cat = category_probe(X, cats, kf)
            print(f"Layer: {layer:8s} | Pooling: {pool_mode:5s} | Category({n_cat}-way) "
                  f"acc={cat_acc:.4f} (majority baseline={cat_base:.4f})")
            results.append({
                "Layer": layer, "Pooling": pool_mode, "Target": f"Category-{n_cat}way",
                "Test Cosine": float("nan"), "Top-10 Acc": cat_acc,
            })

            for target_name, target_dict in target_spaces.items():
                if target_dict is None:
                    continue
                Y_list = [target_dict[r["image_id"]].float() for r in valid_records]
                Y = torch.stack(Y_list).to(device)

                fold_cos, fold_top10 = [], []
                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    Y_train, Y_test = Y[train_idx], Y[test_idx]
                    Y_pred = ridge_regression_fit_predict(X_train, Y_train, X_test, alpha=args.alpha)
                    fold_cos.append(compute_cosine_sim(Y_pred, Y_test))
                    fold_top10.append(compute_top10_retrieval(Y_pred, Y_test))

                mean_cos = float(np.mean(fold_cos))
                mean_top10 = float(np.mean(fold_top10))
                results.append({
                    "Layer": layer, "Pooling": pool_mode, "Target": target_name,
                    "Test Cosine": mean_cos, "Top-10 Acc": mean_top10,
                })
                # Within-fold chance for top-10 ~ 10 / fold_size
                fold_n = len(next(iter(kf.split(X)))[1])
                chance10 = 10.0 / max(fold_n, 1)
                print(f"Layer: {layer:8s} | Pooling: {pool_mode:5s} | Target: {target_name:12s} | "
                      f"Cosine: {mean_cos:.4f} | Top-10: {mean_top10:.4f} (chance~{chance10:.4f})")

    df_res = pd.DataFrame(results).sort_values(by="Top-10 Acc", ascending=False)
    print("\n\n### ZUNA Layer Sweep Results (Ridge Probe) ###")
    print(df_res.to_markdown(index=False))
    out_csv = os.path.join(args.cache_dir, "sweep_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

if __name__ == "__main__":
    main()
