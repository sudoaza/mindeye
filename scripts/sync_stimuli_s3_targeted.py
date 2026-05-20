import boto3
import os
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
import threading

def download_one(bucket_name, key, local_path):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    try:
        s3.download_file(bucket_name, f"ds005811/{key}", str(local_path))
        return True, None
    except Exception as e:
        return False, str(e)

def sync_targeted_stimuli(bucket_name, include_list_path, local_root, max_workers=32):
    local_root = Path(local_root)
    
    with open(include_list_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    stimuli_paths = [p for p in paths if p.startswith('stimuli/ImageNet/')]
    print(f"Found {len(stimuli_paths)} targeted stimuli to check.")
    
    tasks = []
    skipped = 0
    for key in stimuli_paths:
        local_path = local_root / key
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((key, local_path))
        else:
            skipped += 1
            
    print(f"Already exist: {skipped}. Need to download: {len(tasks)}")
    if not tasks:
        print("All files exist. Nothing to download.")
        return
        
    downloaded = 0
    failed = 0
    
    lock = threading.Lock()
    
    def worker(task):
        key, local_path = task
        success, err = download_one(bucket_name, key, local_path)
        with lock:
            nonlocal downloaded, failed
            if success:
                downloaded += 1
                if downloaded % 100 == 0:
                    print(f"Downloaded {downloaded}/{len(tasks)} files...", flush=True)
            else:
                failed += 1
                print(f"Error downloading {key}: {err}", flush=True)
                
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(worker, tasks))
        
    print(f"Finished. Downloaded: {downloaded}, Failed: {failed}, Skipped: {skipped}", flush=True)

if __name__ == "__main__":
    sync_targeted_stimuli('openneuro.org', 'data/raw/nod/stimuli_include.txt', 'data/raw/nod')
