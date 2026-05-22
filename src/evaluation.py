"""
Evaluation Metrics Module
Compute evaluation metrics for similarity search
Intermediate Level: Precision@K, Recall@K, evaluation strategies
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class EvaluationMetrics:
    """
    Compute evaluation metrics for image similarity search
    
    Key Metrics:
    - Precision@K: % of retrieved items that are relevant
    - Recall@K: % of relevant items that are retrieved
    - Mean Average Precision (MAP)
    - Normalized Discounted Cumulative Gain (NDCG)
    """
    
    def __init__(self, ground_truth=None):
        """
        Initialize evaluation metrics
        
        Args:
            ground_truth: dict mapping product_id to list of similar product_ids
        """
        self.ground_truth = ground_truth or {}
        self.results = defaultdict(list)
    
    def precision_at_k(self, retrieved, relevant, k):
        """
        Compute Precision@K
        
        Formula: Precision@K = (# relevant items in top-k) / k
        
        Args:
            retrieved: list of retrieved product indices/ids
            relevant: set/list of relevant product indices/ids
            k: number of top results to consider
            
        Returns:
            float: precision@k score (0-1)
        """
        if k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        # Count how many of top-k are in relevant set
        hits = sum(1 for item in top_k if item in relevant_set)
        
        precision = hits / k if k > 0 else 0.0
        return precision
    
    def recall_at_k(self, retrieved, relevant, k):
        """
        Compute Recall@K
        
        Formula: Recall@K = (# relevant items in top-k) / (total # relevant items)
        
        Args:
            retrieved: list of retrieved product indices/ids
            relevant: set/list of relevant product indices/ids
            k: number of top results to consider
            
        Returns:
            float: recall@k score (0-1)
        """
        if len(relevant) == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        # Count how many of top-k are in relevant set
        hits = sum(1 for item in top_k if item in relevant_set)
        
        recall = hits / len(relevant_set)
        return recall
    
    def average_precision(self, retrieved, relevant, k):
        """
        Compute Average Precision@K
        
        Average of precision values at each relevant item
        
        Args:
            retrieved: list of retrieved product indices/ids
            relevant: set/list of relevant product indices/ids
            k: number of top results to consider
            
        Returns:
            float: average precision score
        """
        if len(relevant) == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        ap = 0.0
        n_hits = 0
        
        for i, item in enumerate(top_k):
            if item in relevant_set:
                n_hits += 1
                ap += n_hits / (i + 1)  # Precision at position i+1
        
        ap = ap / min(len(relevant_set), k)
        return ap
    
    def f1_at_k(self, retrieved, relevant, k):
        """
        Compute F1@K score
        
        Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            retrieved: list of retrieved product indices/ids
            relevant: set/list of relevant product indices/ids
            k: number of top results
            
        Returns:
            float: F1 score
        """
        prec = self.precision_at_k(retrieved, relevant, k)
        rec = self.recall_at_k(retrieved, relevant, k)
        
        if prec + rec == 0:
            return 0.0
        
        f1 = 2 * (prec * rec) / (prec + rec)
        return f1
    
    def mean_reciprocal_rank(self, retrieved, relevant, k):
        """
        Compute Mean Reciprocal Rank@K
        
        MRR = 1 / rank of first relevant item
        
        Args:
            retrieved: list of retrieved product indices/ids
            relevant: set/list of relevant product indices/ids
            k: number of top results
            
        Returns:
            float: MRR score
        """
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        for i, item in enumerate(top_k):
            if item in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def compute_metrics_for_query(self, query_id, retrieved, relevant, k_values=[1, 5, 10]):
        """
        Compute all metrics for a single query
        
        Args:
            query_id: identifier for the query
            retrieved: list of retrieved item indices
            relevant: list/set of relevant items
            k_values: list of k values to compute metrics for
            
        Returns:
            dict: computed metrics
        """
        metrics = {
            'query_id': query_id,
            'n_relevant': len(relevant),
            'n_retrieved': len(retrieved)
        }
        
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved, relevant, k)
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved, relevant, k)
            metrics[f'f1@{k}'] = self.f1_at_k(retrieved, relevant, k)
            metrics[f'map@{k}'] = self.average_precision(retrieved, relevant, k)
            metrics[f'mrr@{k}'] = self.mean_reciprocal_rank(retrieved, relevant, k)
        
        return metrics
    
    def aggregate_metrics(self, all_query_metrics, k_values=[1, 5, 10]):
        """
        Aggregate metrics across multiple queries
        
        Args:
            all_query_metrics: list of metric dicts from compute_metrics_for_query()
            k_values: list of k values
            
        Returns:
            dict: aggregated metrics (mean, std)
        """
        aggregated = {}
        
        for k in k_values:
            precision_scores = [m[f'precision@{k}'] for m in all_query_metrics]
            recall_scores = [m[f'recall@{k}'] for m in all_query_metrics]
            f1_scores = [m[f'f1@{k}'] for m in all_query_metrics]
            map_scores = [m[f'map@{k}'] for m in all_query_metrics]
            mrr_scores = [m[f'mrr@{k}'] for m in all_query_metrics]
            
            aggregated[f'precision@{k}'] = {
                'mean': np.mean(precision_scores),
                'std': np.std(precision_scores)
            }
            aggregated[f'recall@{k}'] = {
                'mean': np.mean(recall_scores),
                'std': np.std(recall_scores)
            }
            aggregated[f'f1@{k}'] = {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores)
            }
            aggregated[f'map@{k}'] = {
                'mean': np.mean(map_scores),
                'std': np.std(map_scores)
            }
            aggregated[f'mrr@{k}'] = {
                'mean': np.mean(mrr_scores),
                'std': np.std(mrr_scores)
            }
        
        aggregated['n_queries'] = len(all_query_metrics)
        
        return aggregated
    
    def print_results(self, metrics_dict):
        """Print metrics in readable format"""
        print("\n" + "="*60)
        print("📊 EVALUATION METRICS")
        print("="*60)
        
        for key, value in metrics_dict.items():
            if isinstance(value, dict) and 'mean' in value:
                print(f"{key:20} -> Mean: {value['mean']:.4f} "
                      f"(±{value['std']:.4f})")
            elif not isinstance(value, dict):
                print(f"{key:20} -> {value}")
    
    def save_metrics(self, metrics_dict, save_path):
        """
        Save metrics to JSON file
        
        Args:
            metrics_dict: dict of metrics
            save_path: path to save
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            return obj
        
        metrics_serializable = convert_types(metrics_dict)
        
        with open(save_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"✅ Metrics saved to {save_path}")


# Example usage
if __name__ == "__main__":
    evaluator = EvaluationMetrics()
    
    # Dummy data: query returns [1, 3, 5, 7, 2, 9, 4, 8, 6, 10]
    # Relevant items: [1, 3, 5, 7]
    retrieved = [1, 3, 5, 7, 2, 9, 4, 8, 6, 10]
    relevant = [1, 3, 5, 7]
    
    # Compute metrics
    metrics = evaluator.compute_metrics_for_query(
        query_id=0,
        retrieved=retrieved,
        relevant=relevant,
        k_values=[1, 5, 10]
    )
    
    print("\n📋 Single Query Metrics:")
    for key, value in metrics.items():
        if not isinstance(value, list):
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    # Simulate multiple queries
    all_metrics = [metrics]
    for i in range(5):
        all_metrics.append(evaluator.compute_metrics_for_query(
            query_id=i+1,
            retrieved=np.random.permutation(retrieved).tolist(),
            relevant=relevant,
            k_values=[1, 5, 10]
        ))
    
    # Aggregate
    aggregated = evaluator.aggregate_metrics(all_metrics)
    evaluator.print_results(aggregated)
