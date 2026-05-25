#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    runs_dir = Path("outputs/runs")
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    # Find all run folders containing metrics.json
    run_paths = []
    for p in runs_dir.iterdir():
        if p.is_dir() and (p / "metrics.json").exists():
            run_paths.append(p)

    # Sort run paths by name/timestamp
    run_paths = sorted(run_paths, key=lambda x: x.name)

    print("| Run Directory | Target Mode | Slug / Config | Best Epoch | Loss | Val Top-1 | Val Top-10 | Val MRR | Collapse Score |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    
    for path in run_paths:
        with open(path / "metrics.json") as f:
            m = json.load(f)
        
        # Load config to get details
        config = {}
        if (path / "config.json").exists():
            with open(path / "config.json") as cf:
                config = json.load(cf)
        
        slug = config.get("slug", "N/A")
        model = config.get("model", "N/A")
        dual_head = config.get("dual_head", False)
        use_fixed = config.get("use_fixed_mean_norm", False)
        
        config_desc = f"{model}"
        if dual_head:
            config_desc += " + Dual"
            if use_fixed:
                config_desc += " (Fixed Norm)"
            else:
                config_desc += " (Learned Norm)"
        else:
            config_desc += " (Contrastive Only)"
            
        if slug != "N/A":
            config_desc = f"{slug} / {config_desc}"

        print(f"| `{path.name}` | {m.get('target_mode', 'N/A')} | {config_desc} | {m.get('best_epoch', 'N/A')} | {m.get('loss', 0.0):.4f} | {m.get('top1', 0.0):.4f} | {m.get('top10', 0.0):.4f} | {m.get('mrr', 0.0):.4f} | {m.get('collapse_score', 0.0):.4f} |")

if __name__ == "__main__":
    main()
