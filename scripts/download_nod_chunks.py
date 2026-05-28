#!/usr/bin/env python3
import sys
from pathlib import Path
import openneuro

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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
    missing_lines = []
    for line in lines:
        local_path = Path("data/raw/nod") / line
        if not local_path.exists():
            missing_lines.append(line)

    print(f"Total missing files to download: {len(missing_lines)}")
    if not missing_lines:
        print("No files to download.")
        return

    chunk_size = 300
    chunks = list(chunk_list(missing_lines, chunk_size))
    print(f"Divided into {len(chunks)} chunks of size {chunk_size}.")

    for idx, chunk in enumerate(chunks):
        print(f"\n==========================================")
        print(f"Downloading chunk {idx+1}/{len(chunks)} ({len(chunk)} files)...")
        print(f"==========================================")
        
        # We try up to 3 times for each chunk in case of network hiccups
        success = False
        for attempt in range(3):
            try:
                openneuro.download(
                    dataset="ds005811",
                    target_dir="data/raw/nod",
                    include=chunk,
                )
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    print("Max attempts reached. Skipping this chunk (we can rerun later).")
                    
        if success:
            print(f"Successfully downloaded chunk {idx+1}/{len(chunks)}.")

    print("\nChunk download run completed.")

if __name__ == "__main__":
    main()
