import boto3
import os
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config

def sync_targeted_stimuli(bucket_name, include_list_path, local_root):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    local_root = Path(local_root)
    
    with open(include_list_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Filter only ImageNet stimuli
    stimuli_paths = [p for p in paths if p.startswith('stimuli/ImageNet/')]
    
    print(f"Found {len(stimuli_paths)} targeted stimuli to download from S3.")
    
    downloaded = 0
    skipped = 0
    for key in stimuli_paths:
        local_path = local_root / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # OpenNeuro S3 bucket prefix is 'ds005811/'
        s3_key = f"ds005811/{key}"
        
        if not local_path.exists():
            try:
                s3.download_file(bucket_name, s3_key, str(local_path))
                downloaded += 1
                if downloaded % 100 == 0:
                    print(f"Downloaded {downloaded} files...", flush=True)
            except Exception as e:
                print(f"Error downloading {s3_key}: {e}", flush=True)
        else:
            skipped += 1
            
    print(f"Finished. Downloaded: {downloaded}, Skipped (already exist): {skipped}", flush=True)

if __name__ == "__main__":
    sync_targeted_stimuli('openneuro.org', 'data/raw/nod/stimuli_include.txt', 'data/raw/nod')
