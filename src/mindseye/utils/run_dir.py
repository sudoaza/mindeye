import os
import sys
import json
import subprocess
from datetime import datetime
import torch

def setup_run_dir(base_dir="outputs/runs", config=None, slug=""):
    """
    Creates a standardized run directory with timestamp and slug.
    Saves environment info, config, and git commit.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{slug}" if slug else timestamp
    run_dir = os.path.join(base_dir, run_name)
    
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    if config is not None:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            if isinstance(config, dict):
                import yaml
                yaml.dump(config, f)
            else:
                f.write(str(config))
                
    # Save environment details
    with open(os.path.join(run_dir, "environment.txt"), "w") as f:
        f.write(f"Hostname: {os.uname().nodename}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")
        
    # Save pip freeze
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        with open(os.path.join(run_dir, "requirements.txt"), "w") as f:
            f.write(pip_freeze)
    except Exception as e:
        print(f"Warning: Could not save pip freeze: {e}")
        
    # Save git commit
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        with open(os.path.join(run_dir, "git_commit.txt"), "w") as f:
            f.write(git_commit)
    except Exception as e:
        print(f"Warning: Could not get git commit: {e}")
        
    return run_dir

def save_metrics(run_dir, metrics_dict):
    """
    Saves metrics dictionary to metrics.json
    """
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)
        
    # Optionally append to a CSV
    import csv
    csv_path = os.path.join(run_dir, "metrics.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(metrics_dict)
