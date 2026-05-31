import torch
import sys
import os
import json
from huggingface_hub import hf_hub_download

# Ensure import path includes src/
sys.path.append("/workspace/mindeye/src")

def main():
    hf_repo = "Zyphra/ZUNA"
    config_path = hf_hub_download(repo_id=hf_repo, filename="config.json")
    with open(config_path, "r") as f:
        cfig = json.load(f)
    print("HF config model section:")
    print(json.dumps(cfig["model"], indent=2))

if __name__ == "__main__":
    main()
