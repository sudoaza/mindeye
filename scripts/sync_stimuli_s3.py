import boto3
import os
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

def sync_s3_folder(bucket_name, s3_folder, local_dir):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):
                continue
            
            relative_path = key[len(s3_folder):].lstrip('/')
            local_path = local_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not local_path.exists():
                print(f"Downloading {key}...")
                s3.download_file(bucket_name, key, str(local_path))

if __name__ == "__main__":
    sync_s3_folder('openneuro.org', 'ds005811/stimuli/ImageNet', 'data/raw/nod/stimuli/ImageNet')
