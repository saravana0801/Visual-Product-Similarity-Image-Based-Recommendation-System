"""
Retrieval Module
Perform similarity search and retrieve similar products
Intermediate Level: Ranking, filtering, similarity scoring
"""

import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import K_RETRIEVAL, SIMILARITY_METRIC


class SimilaritySearch:
    """
    Retrieve visually similar products using FAISS index
    
    Key Features:
    - Query image embedding extraction
    - Similarity search via FAISS
    - Result ranking and filtering
    - Similarity score computation
    """
    
    def __init__(self, index, embeddings, product_ids, similarity_metric=SIMILARITY_METRIC):
        """
        Initialize similarity search
        
        Args:
            index: FAISS index object
            embeddings: numpy array of all embeddings
            product_ids: array of product IDs corresponding to embeddings
            similarity_metric: 'cosine' or 'l2'
        """
        self.index = index
        self.embeddings = embeddings
        self.product_ids = product_ids
        self.similarity_metric = similarity_metric
        self.n_products = len(product_ids)
        
        print(f"✅ SimilaritySearch initialized")
        print(f"   Total products: {self.n_products}")
        print(f"   Similarity metric: {similarity_metric}")
    
    def compute_similarity(self, query_embedding, target_embedding):
        """
        Compute similarity between two embeddings
        
        Args:
            query_embedding: numpy array (dim,)
            target_embedding: numpy array (dim,)
            
        Returns:
            float: similarity score (higher is more similar)
        """
        if self.similarity_metric == 'cosine':
            # Cosine similarity: dot product of normalized vectors
            norm_query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            norm_target = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)
            similarity = np.dot(norm_query, norm_target)
            
        elif self.similarity_metric == 'l2':
            # L2 distance: convert to similarity (inverse distance)
            distance = np.linalg.norm(query_embedding - target_embedding)
            similarity = 1.0 / (1.0 + distance)
        
        else:
            raise ValueError(f"Unknown metric: {self.similarity_metric}")
        
        return similarity
    
    def search_similar_products(self, query_embedding, k=K_RETRIEVAL, dedupe=True):
        """
        Find top-k similar products for query image
        
        Args:
            query_embedding: numpy array (embedding_dim,)
            k: number of results to return
            dedupe: whether to skip repeated product IDs
            
        Returns:
            dict: Contains indices, product_ids, similarities, distances
        """
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # existing code: search in FAISS using raw query embeddings
        # distances, indices = self.index.search(query_embedding, k)

        # new code: normalize query vector for cosine similarity when index uses normalized vectors
        if self.similarity_metric == 'cosine':
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
            query_embedding = query_embedding / norm

        search_k = min(max(k * 4, k + 10), self.n_products)
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Extract results for first (only) query
        distances = distances[0]
        indices = indices[0]
        
        # Convert distances to similarities
        similarities = self._distances_to_similarities(distances)
        
        # Get product IDs
        result_product_ids = self.product_ids[indices]

        if dedupe:
            indices, distances, similarities, result_product_ids = self._dedupe_results(
                indices, distances, similarities, result_product_ids, k
            )

        results = {
            'rank': np.arange(1, len(result_product_ids) + 1),
            'indices': indices,
            'product_ids': result_product_ids,
            'similarities': similarities,
            'distances': distances,
            'metric': self.similarity_metric
        }
        
        return results
    
    def batch_search_similar_products(self, query_embeddings, k=K_RETRIEVAL):
        """
        Find top-k similar products for multiple queries
        
        Args:
            query_embeddings: numpy array (n_queries, embedding_dim)
            k: number of results per query
            
        Returns:
            list: List of result dicts, one per query
        """
        query_embeddings = query_embeddings.astype(np.float32)

        # existing code: search in FAISS using raw query embeddings
        # distances, indices = self.index.search(query_embeddings, k)

        # new code: normalize batch queries for cosine similarity
        if self.similarity_metric == 'cosine':
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
            query_embeddings = query_embeddings / norms

        search_k = min(max(k * 4, k + 10), self.n_products)
        distances, indices = self.index.search(query_embeddings, search_k)
        
        all_results = []
        
        for i in range(len(query_embeddings)):
            distances_i = distances[i]
            indices_i = indices[i]
            similarities = self._distances_to_similarities(distances_i)
            product_ids_i = self.product_ids[indices_i]

            indices_i, distances_i, similarities, product_ids_i = self._dedupe_results(
                indices_i, distances_i, similarities, product_ids_i, k
            )
            
            results = {
                'query_id': i,
                'rank': np.arange(1, len(product_ids_i) + 1),
                'indices': indices_i,
                'product_ids': product_ids_i,
                'similarities': similarities,
                'distances': distances_i
            }
            all_results.append(results)
        
        return all_results

    def _dedupe_results(self, indices, distances, similarities, product_ids, k):
        """
        Remove duplicate product IDs from result lists.

        Args:
            indices: numpy array of FAISS indices
            distances: numpy array of distances
            similarities: numpy array of similarity scores
            product_ids: numpy array of product IDs
            k: number of unique results to keep

        Returns:
            tuple: deduped (indices, distances, similarities, product_ids)
        """
        seen = set()
        unique_indices = []
        unique_distances = []
        unique_similarities = []
        unique_product_ids = []

        for idx, dist, sim, pid in zip(indices, distances, similarities, product_ids):
            if pid in seen:
                continue
            seen.add(pid)
            unique_indices.append(idx)
            unique_distances.append(dist)
            unique_similarities.append(sim)
            unique_product_ids.append(pid)
            if len(unique_product_ids) == k:
                break

        return (
            np.array(unique_indices, dtype=np.int64),
            np.array(unique_distances, dtype=distances.dtype),
            np.array(unique_similarities, dtype=similarities.dtype),
            np.array(unique_product_ids, dtype=product_ids.dtype)
        )
    
    def _distances_to_similarities(self, distances):
        """
        Convert FAISS distances to similarity scores
        
        Args:
            distances: numpy array of L2 distances
            
        Returns:
            numpy array: similarity scores in [0, 1]
        """
        if self.similarity_metric == 'cosine':
            # FAISS returns L2 distances, convert back to cosine similarity
            # If d(a,b) = ||a-b||^2 = 2 - 2*cos_sim(a,b) for normalized vectors
            similarities = 1.0 - (distances / 2.0)
            similarities = np.clip(similarities, 0, 1)
        else:
            # L2: convert distance to similarity
            similarities = 1.0 / (1.0 + distances)
        
        return similarities
    
    def rank_results(self, results, sort_by='similarity'):
        """
        Rank results by similarity score
        
        Args:
            results: dict from search_similar_products()
            sort_by: 'similarity' or 'distance'
            
        Returns:
            dict: ranked results
        """
        if sort_by == 'similarity':
            indices = np.argsort(-results['similarities'])  # descending
        elif sort_by == 'distance':
            indices = np.argsort(results['distances'])  # ascending
        else:
            raise ValueError(f"Unknown sort criterion: {sort_by}")
        
        # Reorder all fields
        results_ranked = {
            'rank': results['rank'][indices],
            'indices': results['indices'][indices],
            'product_ids': results['product_ids'][indices],
            'similarities': results['similarities'][indices],
            'distances': results['distances'][indices],
            'metric': results['metric']
        }
        
        return results_ranked
    
    def filter_by_threshold(self, results, threshold=0.5):
        """
        Filter results by similarity threshold
        
        Args:
            results: dict from search_similar_products()
            threshold: minimum similarity score
            
        Returns:
            dict: filtered results
        """
        mask = results['similarities'] >= threshold
        
        results_filtered = {
            'rank': results['rank'][mask],
            'indices': results['indices'][mask],
            'product_ids': results['product_ids'][mask],
            'similarities': results['similarities'][mask],
            'distances': results['distances'][mask],
            'metric': results['metric'],
            'threshold': threshold,
            'n_original': len(results['similarities']),
            'n_filtered': np.sum(mask)
        }
        
        return results_filtered
    
    def format_results(self, results, include_metadata=False):
        """
        Format results for display/export
        
        Args:
            results: dict from search functions
            include_metadata: whether to include metadata
            
        Returns:
            list: formatted results (rank, product_id, similarity)
        """
        formatted = []
        
        for i in range(len(results['rank'])):
            entry = {
                'rank': int(results['rank'][i]),
                'product_id': int(results['product_ids'][i]),
                'similarity_score': float(results['similarities'][i]),
                'distance': float(results['distances'][i])
            }
            formatted.append(entry)
        
        return formatted
    
    def save_results(self, results, save_path, format='json'):
        """
        Save search results to file
        
        Args:
            results: dict from search functions
            save_path: path to save results
            format: 'json' or 'csv'
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        formatted = self.format_results(results)
        
        if format == 'json':
            with open(save_path, 'w') as f:
                json.dump(formatted, f, indent=2)
        
        print(f"✅ Results saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Create dummy data
    n_products = 1000
    embedding_dim = 2048
    embeddings = np.random.randn(n_products, embedding_dim).astype(np.float32)
    product_ids = np.arange(n_products)
    
    # Create dummy index (for demo)
    import faiss
    quantizer = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
    index.train(embeddings)
    index.add(embeddings)
    
    # Initialize search
    search = SimilaritySearch(index, embeddings, product_ids)
    
    # Search for similar products
    query = embeddings[0]
    results = search.search_similar_products(query, k=10)
    
    print(f"\n📊 Search Results:")
    print(f"   Top 5 similar products:")
    for i in range(min(5, len(results['product_ids']))):
        print(f"   #{i+1}: Product {results['product_ids'][i]}, "
              f"Similarity: {results['similarities'][i]:.4f}")
    
    # Format results
    formatted = search.format_results(results)
    print(f"\n📋 Formatted Results:")
    for entry in formatted[:3]:
        print(f"   {entry}")
