"""
FAISS Indexing Module
Build and manage vector index for similarity search
Intermediate Level: Vector databases, approximate nearest neighbor search, index optimization
"""

import numpy as np
import faiss
from pathlib import Path
import time
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INDEX_TYPE, N_CLUSTERS, N_PROBE, MODEL_DIR, EMBEDDING_DIM
)


class FAISSIndexing:
    """
    Build and manage FAISS index for efficient similarity search
    
    Key Concepts:
    - FAISS: Facebook AI Similarity Search
    - ANN: Approximate Nearest Neighbor (trades accuracy for speed)
    - Index Types:
      * FLAT: Exhaustive search (slow but accurate)
      * IVF: Inverted File index (fast with clustering)
      * HNSW: Hierarchical Navigable Small World (very fast)
    
    Use Cases:
    - Store millions of embeddings
    - Query in milliseconds
    - Scalable to production workloads
    """
    
    def __init__(self, embedding_dim=EMBEDDING_DIM, index_type=INDEX_TYPE):
        """
        Initialize FAISS indexing
        
        Args:
            embedding_dim: dimension of embeddings (2048 for ResNet50)
            index_type: 'FLAT', 'IVF', or 'HNSW'
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.model_dir = MODEL_DIR
        self.index = None
        self.metadata = {
            'embedding_dim': embedding_dim,
            'index_type': index_type,
            'n_vectors': 0,
            'created_at': str(Path(__file__).parent.parent)
        }
        
        print(f"✅ FAISSIndexing initialized")
        print(f"   Embedding Dim: {embedding_dim}")
        print(f"   Index Type: {index_type}")
    
    def create_index(self, embeddings, normalize=False):
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: numpy array (n_samples, embedding_dim)
            normalize: whether to L2-normalize embeddings for cosine similarity
            
        Returns:
            faiss.Index: FAISS index object
        """
        print(f"🔨 Building {self.index_type} FAISS index...")
        print(f"   Total vectors: {len(embeddings)}")
        print(f"   Dimension: {embeddings.shape[1]}")
        print(f"   Normalize embeddings: {normalize}")
        
        start_time = time.time()
        
        # Ensure embeddings are float32 and C-contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # existing code: normalize embeddings before indexing if cosine similarity is used
        if normalize:
            print("   Normalizing embeddings for cosine similarity...")
            faiss.normalize_L2(embeddings)

        print(f"   Embedding shape: {embeddings.shape}")

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D. Got shape {embeddings.shape}"
            )
        
        if self.index_type == 'FLAT':
            # Exhaustive search - most accurate
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings)
            
        elif self.index_type == 'IVF':
            # # Inverted File with clustering
            # quantizer = faiss.IndexFlatL2(self.embedding_dim)
            # n_clusters = min(N_CLUSTERS, len(embeddings) // 100)
            # self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            
            # print(f"   Clustering embeddings into {n_clusters} clusters...")
            # self.index.train(embeddings)
            # self.index.add(embeddings)
            # self.index.nprobe = N_PROBE

            # Validate embeddings
            if embeddings is None or len(embeddings) == 0:
                raise ValueError(
                    "Embeddings array is empty. "
                    "Feature extraction may have failed."
                )

            if embeddings.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch. "
                    f"Expected {self.embedding_dim}, "
                    f"got {embeddings.shape[1]}"
                )

            # Inverted File with clustering
            quantizer = faiss.IndexFlatL2(self.embedding_dim)

            # Safe cluster calculation
            n_vectors = len(embeddings)

            if n_vectors < 5000:
                print("⚠️ Small dataset detected. Switching IVF -> FLAT")

                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.index.add(embeddings)

                return self.index
            # Recommended IVF rule:
            # clusters should be sqrt(number of vectors)
            # n_clusters = min(N_CLUSTERS, max(1, int(np.sqrt(n_vectors))))
            # FAISS requires enough training samples per cluster

            # Inverted File with clustering
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            
            max_clusters = max(1, n_vectors // 40)

            n_clusters = min(
                N_CLUSTERS,
                max_clusters
            )

            # Absolute safety
            n_clusters = max(1, n_clusters)

            print(f"   Clustering embeddings into {n_clusters} clusters...")

            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                n_clusters
            )

            # Train only if enough vectors exist
            if n_vectors < n_clusters:
                raise ValueError(
                    f"Not enough vectors ({n_vectors}) "
                    f"for {n_clusters} clusters"
                )

            self.index.train(embeddings)
            self.index.add(embeddings)

            # nprobe cannot exceed n_clusters
            self.index.nprobe = min(N_PROBE, n_clusters)
            
        elif self.index_type == 'HNSW':
            # Hierarchical Navigable Small World
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.add(embeddings)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        build_time = time.time() - start_time
        
        self.metadata['n_vectors'] = len(embeddings)
        self.metadata['build_time_sec'] = build_time
        self.metadata['normalized'] = normalize
        
        print(f"✅ Index built successfully!")
        print(f"   Total vectors indexed: {self.index.ntotal}")
        print(f"   Build time: {build_time:.2f}s")
        
        return self.index
    
    def search(self, query_embeddings, k=10):
        """
        Search for top-k nearest neighbors
        
        Args:
            query_embeddings: numpy array (n_queries, embedding_dim)
            k: number of nearest neighbors to return
            
        Returns:
            tuple: (distances, indices)
                distances: (n_queries, k) - L2 distances
                indices: (n_queries, k) - indices of nearest neighbors
        """
        if self.index is None:
            raise ValueError("Index not built! Call create_index() first")
        
        # Ensure query is float32
        query_embeddings = np.ascontiguousarray(
            query_embeddings.astype(np.float32)
        )
        
        print(f"🔍 Searching for {k} nearest neighbors...")
        start_time = time.time()
        
        distances, indices = self.index.search(query_embeddings, k)
        
        search_time = time.time() - start_time
        print(f"✅ Search completed in {search_time*1000:.2f}ms")
        
        return distances, indices
    
    def save_index(self, save_path=None):
        """
        Save index to disk
        
        Args:
            save_path: path to save index (default: models/faiss_index.bin)
        """
        if self.index is None:
            raise ValueError("No index to save!")
        
        if save_path is None:
            save_path = self.model_dir / "faiss_index.bin"
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving FAISS index to {save_path}...")
        faiss.write_index(self.index, str(save_path))
        
        # Save metadata
        metadata_path = save_path.parent / "faiss_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Index saved! Size: {save_path.stat().st_size / (1024**2):.2f} MB")
    
    def load_index(self, load_path=None):
        """
        Load index from disk
        
        Args:
            load_path: path to load index from
        """
        if load_path is None:
            load_path = self.model_dir / "faiss_index.bin"
        else:
            load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index not found at {load_path}")
        
        print(f"📂 Loading FAISS index from {load_path}...")
        self.index = faiss.read_index(str(load_path))
        
        # Load metadata if available
        metadata_path = load_path.parent / "faiss_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        print(f"✅ Index loaded! Total vectors: {self.index.ntotal}")
        return self.index
    
    def get_index_info(self):
        """Return index information"""
        if self.index is None:
            return None
        
        return {
            'index_type': self.index_type,
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_class': self.index.__class__.__name__,
            'metadata': self.metadata
        }
    
    def add_vectors(self, vectors):
        """
        Add new vectors to existing index
        
        Args:
            vectors: numpy array (n_new_vectors, embedding_dim)
        """
        if self.index is None:
            raise ValueError("Index not built! Call create_index() first")
        
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        print(f"➕ Adding {len(vectors)} new vectors to index...")
        self.index.add(vectors)
        self.metadata['n_vectors'] = self.index.ntotal
        
        print(f"✅ Vectors added! Total: {self.index.ntotal}")


# Example usage
if __name__ == "__main__":
    # Create dummy embeddings
    n_samples = 1000
    embedding_dim = 2048
    dummy_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Normalize embeddings (optional but recommended)
    faiss.normalize_L2(dummy_embeddings)
    
    # Build index
    indexing = FAISSIndexing(embedding_dim=embedding_dim, index_type='IVF')
    index = indexing.create_index(dummy_embeddings)
    
    # Search
    query = dummy_embeddings[:5]  # Use first 5 as queries
    distances, indices = indexing.search(query, k=10)
    
    print(f"\n📊 Search Results:")
    print(f"   Distances shape: {distances.shape}")
    print(f"   Indices shape: {indices.shape}")
    print(f"   Top-10 nearest: {indices[0][:10]}")
    
    # Save index
    indexing.save_index()
    
    # Index info
    info = indexing.get_index_info()
    print(f"\n🔧 Index Info:")
    for key, value in info.items():
        if key != 'metadata':
            print(f"   {key}: {value}")
