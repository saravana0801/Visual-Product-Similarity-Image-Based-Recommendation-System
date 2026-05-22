"""
Main Application Pipeline
Orchestrate the entire workflow: data loading -> feature extraction -> indexing -> retrieval
"""

import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.indexing import FAISSIndexing
from src.retrieval import SimilaritySearch
from src.evaluation import EvaluationMetrics
from config import RESULTS_DIR, MODEL_DIR, DATA_DIR, SIMILARITY_METRIC


class ImageSimilarityPipeline:
    """
    End-to-end pipeline for visual product similarity
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        self.data_loader = None
        self.feature_extractor = None
        self.indexing = None
        self.search = None
        self.evaluator = None
        
        print("\n" + "="*70)
        print("🛍️  VISUAL PRODUCT SIMILARITY & RECOMMENDATION SYSTEM")
        print("="*70)
    
    def step1_load_data(self, sample_size=None):
        """
        Step 1: Load and preprocess Stanford Online Products dataset
        
        Args:
            sample_size: number of samples to load (None = all)
        """
        print("\n" + "─"*70)
        print("STEP 1: DATA LOADING & PREPROCESSING")
        print("─"*70)
        
        self.data_loader = DataLoader()
        
        # Try loading processed data first
        train_data = self.data_loader.load_processed_dataset(split='train')
        
        if train_data is None:
            # Download and process if not available
            train_data = self.data_loader.load_and_process_dataset(
                split='train',
                sample_size=sample_size
            )
            
            if train_data:
                self.data_loader.save_processed_dataset(train_data, split='train')
        
        return train_data
    
    def step2_extract_features(self, images, model_name='resnet50'):
        """
        Step 2: Extract embeddings using pretrained CNN
        
        Args:
            images: numpy array of preprocessed images
            model_name: 'resnet50' or 'efficientnet_b0'
            
        Returns:
            numpy array of embeddings
        """
        print("\n" + "─"*70)
        print("STEP 2: FEATURE EXTRACTION (EMBEDDINGS)")
        print("─"*70)
        
        self.feature_extractor = FeatureExtractor(model_name=model_name)
        
        # Extract embeddings
        embeddings = self.feature_extractor.extract_embeddings_from_dataset(images)
        
        # Save embeddings
        self.feature_extractor.save_embeddings(embeddings)
        
        return embeddings
    
    def step3_build_index(self, embeddings, index_type='IVF'):
        """
        Step 3: Build FAISS index for similarity search
        
        Args:
            embeddings: numpy array of embeddings
            index_type: 'FLAT', 'IVF', or 'HNSW'
            
        Returns:
            FAISS index object
        """
        print("\n" + "─"*70)
        print("STEP 3: FAISS INDEXING")
        print("─"*70)
        
        self.indexing = FAISSIndexing(
            embedding_dim=embeddings.shape[1],
            index_type=index_type
        )
        
        # Create and save index
        index = self.indexing.create_index(
            embeddings,
            normalize=(SIMILARITY_METRIC == 'cosine')
        )
        self.indexing.save_index()
        
        return index
    
    def step4_search_products(self, query_image, embeddings, product_ids, k=10):
        """
        Step 4: Search for similar products
        
        Args:
            query_image: numpy array of query image
            embeddings: all product embeddings
            product_ids: product IDs
            k: number of results
            
        Returns:
            dict: search results
        """
        print("\n" + "─"*70)
        print("STEP 4: SIMILARITY SEARCH")
        print("─"*70)
        
        # Initialize search if not done
        if self.search is None:
            self.search = SimilaritySearch(
                self.indexing.index,
                embeddings,
                product_ids
            )
        
        # Extract query embedding
        query_embedding = self.feature_extractor.extract_embedding(query_image)
        
        # Search
        results = self.search.search_similar_products(query_embedding, k=k)
        
        return results
    
    def step5_evaluate(self, retrieved_ids, relevant_ids, k_values=[1, 5, 10]):
        """
        Step 5: Evaluate results (if ground truth available)
        
        Args:
            retrieved_ids: list of retrieved product indices
            relevant_ids: list of relevant product indices
            k_values: k values for evaluation
            
        Returns:
            dict: evaluation metrics
        """
        print("\n" + "─"*70)
        print("STEP 5: EVALUATION")
        print("─"*70)
        
        self.evaluator = EvaluationMetrics()
        
        metrics = self.evaluator.compute_metrics_for_query(
            query_id=0,
            retrieved=retrieved_ids,
            relevant=relevant_ids,
            k_values=k_values
        )
        
        self.evaluator.print_results(metrics)
        
        return metrics
    
    def run_full_pipeline(self, sample_size=1000, model_name='resnet50', index_type='IVF'):
        """
        Run complete pipeline from start to finish
        
        Args:
            sample_size: number of products to use
            model_name: CNN model to use
            index_type: FAISS index type
        """
        # Step 1: Load data
        train_data = self.step1_load_data(sample_size=sample_size)

        print("\nDATA DEBUG")
        print("Train images:", len(train_data['images']))
        # print("Train images shape:", train_data['images'].shape)
        # print("Train labels shape:", train_data['labels'].shape)
        # print("Train product IDs shape:", train_data['product_ids'].shape)
        
        if train_data is None:
            print("❌ Failed to load data!")
            return
        
        images = train_data['images']
        labels = train_data['labels']
        product_ids = train_data['product_ids']
        
        # Step 2: Extract features
        embeddings = self.step2_extract_features(images, model_name=model_name)

        print("\nDEBUG INFO")
        print("Embeddings type:", type(embeddings))

        if embeddings is not None:
            print("Embeddings shape:", embeddings.shape)
            print("Embeddings size:", embeddings.size)
        else:
            print("Embeddings is None")
        
        # Step 3: Build index
        index = self.step3_build_index(embeddings, index_type=index_type)
        
        print("\n" + "="*70)
        print("✅ PIPELINE SETUP COMPLETE!")
        print("="*70)
        print(f"\nReady to search similar products!")
        print(f"Use step4_search_products() to find visually similar items")
        
        # Step 4: Demo search with first image
        print("\n" + "─"*70)
        print("DEMO: SEARCHING WITH FIRST IMAGE")
        print("─"*70)
        
        query_image = images[0]
        results = self.step4_search_products(query_image, embeddings, product_ids, k=10)
        
        # Display results
        print("\n📊 Top-10 Similar Products:")
        for i in range(min(10, len(results['product_ids']))):
            print(f"   #{i+1}: Product {results['product_ids'][i]:5d} "
                  f"| Similarity: {results['similarities'][i]:.4f}")
        
        return {
            'images': images,
            'embeddings': embeddings,
            'product_ids': product_ids,
            'labels': labels,
            'index': index,
            'pipeline': self
        }
    
    def save_pipeline_state(self, save_dir=RESULTS_DIR):
        """Save pipeline state for later use"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.feature_extractor.get_model_info() if self.feature_extractor else None,
            'index_info': self.indexing.get_index_info() if self.indexing else None
        }
        
        with open(save_dir / 'pipeline_state.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"✅ Pipeline state saved to {save_dir / 'pipeline_state.json'}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Visual Product Similarity Pipeline')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of images to use')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model: resnet50 or efficientnet_b0')
    parser.add_argument('--index-type', type=str, default='IVF',
                       help='Index type: FLAT, IVF, or HNSW')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run Step 5 evaluation after pipeline execution')
    parser.add_argument('--query-id', type=int, default=0,
                       help='Query image index to evaluate')
    parser.add_argument('--eval-k-values', type=str, default='1,5,10',
                       help='Comma-separated K values for evaluation metrics')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ImageSimilarityPipeline()
    
    # Run pipeline
    result = pipeline.run_full_pipeline(
        sample_size=args.sample_size,
        model_name=args.model,
        index_type=args.index_type
    )

    if args.evaluate and result is not None:
        # Parse evaluation K values
        try:
            k_values = [int(x.strip()) for x in args.eval_k_values.split(',') if x.strip()]
        except ValueError:
            k_values = [1, 5, 10]
            print("⚠️ Invalid --eval-k-values format. Falling back to 1,5,10.")

        query_id = max(0, min(args.query_id, len(result['labels']) - 1))
        query_label = result['labels'][query_id]

        # Build relevant set: all indices sharing the same label except query itself
        relevant_ids = [idx for idx, label in enumerate(result['labels'])
                        if label == query_label and idx != query_id]

        retrieved_ids = result['pipeline'].step4_search_products(
            result['images'][query_id],
            result['embeddings'],
            result['product_ids'],
            k=max(k_values)
        )['indices'].tolist()

        print(f"\nEvaluating query_id={query_id} with label={query_label}")
        pipeline.step5_evaluate(retrieved_ids, relevant_ids, k_values=k_values)

    # Save state only if pipeline completed successfully
    if result is not None and pipeline.feature_extractor is not None and pipeline.indexing is not None:
        pipeline.save_pipeline_state()
    else:
        print("⚠️ Pipeline did not complete successfully. Skipping state save.")
    
    print("\n" + "="*70)
    print("🎉 PIPELINE EXECUTION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
