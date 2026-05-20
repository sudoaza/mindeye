#!/usr/bin/env python3
"""Build a FAISS retrieval index from a compiled PyTorch embedding table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mindseye.embeddings.faiss_index import FAISSIndex

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embeddings",
        required=True,
        help="Path to .pt containing target image/common embeddings table",
    )
    p.add_argument(
        "--embedding-key",
        default=None,
        help="Sub-dictionary key to index (e.g. 'image_id_to_common' or 'image_id_to_image')",
    )
    p.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix path for saving the FAISS index and metadata files",
    )
    p.add_argument(
        "--metric",
        choices=("cosine", "l2"),
        default="cosine",
        help="Retrieval similarity metric (default: cosine)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings_dict = torch.load(args.embeddings, map_location="cpu")
    
    target_dict = None
    if args.embedding_key:
        if args.embedding_key in embeddings_dict:
            target_dict = embeddings_dict[args.embedding_key]
        else:
            raise KeyError(f"Key '{args.embedding_key}' not found in the embeddings file.")
    else:
        # Auto-detect structure
        if isinstance(embeddings_dict, dict):
            first_val = next(iter(embeddings_dict.values()))
            if isinstance(first_val, torch.Tensor) and first_val.ndim == 1:
                target_dict = embeddings_dict
            elif "image_id_to_common" in embeddings_dict:
                print("Auto-detected 'image_id_to_common' key in embeddings file.")
                target_dict = embeddings_dict["image_id_to_common"]
            elif "image_id_to_image" in embeddings_dict:
                print("Auto-detected 'image_id_to_image' key in embeddings file.")
                target_dict = embeddings_dict["image_id_to_image"]
            else:
                raise ValueError("Could not auto-detect embedding sub-dictionary. Please specify --embedding-key.")
        else:
            raise ValueError("Embeddings table is not a dictionary.")
            
    # Extract IDs and embeddings
    ids = sorted(target_dict.keys())
    if not ids:
        raise ValueError("No embeddings found in the target dictionary.")
        
    sample_emb = target_dict[ids[0]]
    dimension = sample_emb.shape[0]
    print(f"Indexing {len(ids)} unique items in {dimension}-dimensional space using {args.metric} metric...")
    
    # Compile embeddings matrix
    embeddings_list = [target_dict[i].float() for i in ids]
    embeddings_matrix = torch.stack(embeddings_list)
    
    # Build FAISS Index
    index = FAISSIndex(dimension=dimension, metric=args.metric)
    index.add(embeddings_matrix, ids)
    
    # Save index
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    index.save(output_prefix)
    
    print(f"FAISS index successfully saved to: {output_prefix}.index")
    print(f"Index metadata successfully saved to: {output_prefix}_meta.json")


if __name__ == "__main__":
    main()
