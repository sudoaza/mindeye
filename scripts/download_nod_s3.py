#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

def download_file(s3_client, bucket, key, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(bucket, key, str(local_path))
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    include_file = Path("data/processed/clip_embeddings/missing_image_includes.txt")
    if not include_file.exists():
        print(f"Error: {include_file} does not exist.")
        sys.exit(1)

    lines = []
    with open(include_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)

    print(f"Total files in include list: {len(lines)}")
    
    # Filter out files that already exist on disk
    missing_files = []
    for line in lines:
        local_path = Path("data/raw/nod") / line
        if not local_path.exists():
            missing_files.append((line, local_path))

    print(f"Total missing files to download: {len(missing_files)}")
    if not missing_files:
        print("No files to download.")
        return

    # Set up S3 client with UNSIGNED config (no credentials needed)
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'openneuro.org'
    
    max_workers = 32
    print(f"Starting downloads with {max_workers} threads...")
    
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for line, local_path in missing_files:
            key = f"ds005811/{line}"
            future = executor.submit(download_file, s3_client, bucket, key, local_path)
            futures[future] = line
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            line = futures[future]
            success, err = future.result()
            if not success:
                failures.append((line, err))
                
    if failures:
        print(f"\nCompleted with {len(failures)} failures.")
        # Write failures to a log
        with open("s3_download_failures.log", "w") as f:
            for line, err in failures:
                f.write(f"{line} : {err}\n")
        print("Failures logged to s3_download_failures.log")
    else:
        print("\nAll downloads completed successfully!")

if __name__ == "__main__":
    main()
