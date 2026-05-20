import json
from pathlib import Path
import numpy as np
import torch
import faiss

class FAISSIndex:
    """Production-grade wrapper around FAISS for visual and multimodal retrieval.
    
    Uses IndexFlatIP (Inner Product) for cosine similarity on L2-normalized embeddings,
    and IndexFlatL2 for Euclidean distance. Strictly relies on native faiss.
    """
    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric.lower()
        self.ids = []
        
        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Must be 'cosine' or 'l2'")

    def add(self, embeddings: np.ndarray | torch.Tensor, ids: list[str]) -> None:
        """Add embeddings to the index with corresponding IDs.
        
        Args:
            embeddings: [N, D] array or tensor of embeddings.
            ids: list of N identifiers (e.g., image IDs).
        """
        if len(ids) == 0:
            return
            
        assert len(embeddings) == len(ids), f"Mismatch: {len(embeddings)} embeddings and {len(ids)} IDs."
        
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = np.asarray(embeddings, dtype=np.float32)
            
        embeddings_np = embeddings_np.astype(np.float32)
        
        if self.metric == "cosine":
            # L2 normalize vectors to perform cosine similarity via Inner Product
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            embeddings_np = embeddings_np / np.maximum(norms, 1e-8)
            
        self.ids.extend(ids)
        self.index.add(embeddings_np)

    def search(self, query: np.ndarray | torch.Tensor, k: int = 5) -> tuple[np.ndarray, list[list[str]]]:
        """Search the index for the top-k nearest neighbors.
        
        Args:
            query: [Q, D] query embeddings.
            k: number of neighbors to retrieve.
            
        Returns:
            distances: [Q, k] distances or similarities.
            retrieved_ids: list of list of IDs of shape [Q, k].
        """
        if len(self.ids) == 0:
            return np.zeros((len(query), 0)), [[] for _ in range(len(query))]
            
        if isinstance(query, torch.Tensor):
            query_np = query.detach().cpu().numpy()
        else:
            query_np = np.asarray(query, dtype=np.float32)
            
        query_np = query_np.astype(np.float32)
        
        if self.metric == "cosine":
            # L2 normalize query vectors
            norms = np.linalg.norm(query_np, axis=1, keepdims=True)
            query_np = query_np / np.maximum(norms, 1e-8)
            
        # FAISS search
        distances, indices = self.index.search(query_np, k)
        
        retrieved_ids = []
        for i in range(len(query_np)):
            row_ids = [self.ids[idx] for idx in indices[i] if idx != -1]
            # Pad with empty string if fewer than k neighbors found
            while len(row_ids) < k:
                row_ids.append("")
            retrieved_ids.append(row_ids)
            
        return distances, retrieved_ids

    def save(self, filepath_prefix: str | Path) -> None:
        """Save the FAISS index and metadata files to disk.
        
        Saves the FAISS index to filepath_prefix + '.index' and the metadata JSON mapping
        to filepath_prefix + '_meta.json'.
        """
        prefix = Path(filepath_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(prefix) + ".index")
        
        # Save metadata JSON
        meta = {
            "dimension": self.dimension,
            "metric": self.metric,
            "ids": self.ids
        }
        with open(str(prefix) + "_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, filepath_prefix: str | Path) -> "FAISSIndex":
        """Load the FAISS index and metadata files from disk."""
        prefix = Path(filepath_prefix)
        meta_path = Path(str(prefix) + "_meta.json")
        index_path = Path(str(prefix) + ".index")
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        obj = cls(dimension=meta["dimension"], metric=meta["metric"])
        obj.ids = meta["ids"]
        obj.index = faiss.read_index(str(index_path))
        
        return obj
