"""
Data Loader Module
Handles Stanford Online Products Dataset loading and preprocessing
Intermediate Level: Dataset handling, augmentation, batch processing
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_PATH, PROCESSED_DATA_PATH, IMG_SIZE, NORMALIZE_MEAN, 
    NORMALIZE_STD, BATCH_SIZE, NUM_WORKERS
)


class DataLoader:
    """
    Load and preprocess Stanford Online Products Dataset
    
    Key Features:
    - Automatic dataset download via TensorFlow Datasets
    - Image normalization and preprocessing
    - Batch processing for efficiency
    - Train/Val/Test split
    """
    
    def __init__(self, dataset_name="stanford_online_products"):
        """Initialize DataLoader"""
        self.dataset_name = dataset_name
        self.dataset_path = DATASET_PATH
        self.processed_path = PROCESSED_DATA_PATH
        self.img_size = IMG_SIZE
        self.batch_size = BATCH_SIZE
        self.normalize_mean = np.array(NORMALIZE_MEAN)
        self.normalize_std = np.array(NORMALIZE_STD)
        
        # Create directories
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ DataLoader initialized for {dataset_name}")
    
    def download_dataset(self):
        """
        Download Stanford Online Products dataset from TensorFlow Datasets
        
        Returns:
            dict: Dataset info
        """
        print("📥 Downloading Stanford Online Products dataset...")
        print("   First run may take 10-15 minutes (~2.6 GB)")
        
        try:
            # Download dataset
            (train_ds, test_ds), info = tfds.load(
                'stanford_online_products',
                split=['train', 'test'],
                shuffle_files=True,
                with_info=True,
                download=True
            )
            
            print(f"✅ Dataset downloaded successfully!")
            print(f"   Training samples: {info.splits['train'].num_examples}")
            print(f"   Test samples: {info.splits['test'].num_examples}")
            
            return train_ds, test_ds, info
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("   Manual download: https://www.tensorflow.org/datasets/catalog/stanford_online_products")
            return None, None, None
    
    def preprocess_image(self, image):
        """
        Preprocess image: resize, normalize
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy array: Preprocessed image (224, 224, 3)
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize (ImageNet normalization)
        image_array = (image_array - self.normalize_mean) / self.normalize_std
        
        return image_array
    
    def load_and_process_dataset(self, split='train', sample_size=None):
        """
        Load and preprocess dataset
        
        Args:
            split: 'train' or 'test'
            sample_size: Limit number of samples (for testing)
            
        Returns:
            dict: Contains images, labels, product_ids
        """
        print(f"🔄 Loading {split} dataset...")
        
        train_ds, test_ds, info = self.download_dataset()
        
        if train_ds is None:
            return None
        
        dataset = train_ds if split == 'train' else test_ds
        
        images = []
        labels = []
        product_ids = []
        
        count = 0
        for sample in tqdm(dataset, desc=f"Processing {split} images"):
            if sample_size and count >= sample_size:
                break
            
            try:
                # Extract image and metadata
                image = sample['image'].numpy()

                # Product/category label
                label = sample['class_id'].numpy()

                # Higher-level category
                product_id = sample['super_class_id'].numpy()
                
                # Preprocess
                processed_img = self.preprocess_image(image)
                
                images.append(processed_img)
                labels.append(label)
                product_ids.append(product_id)
                
                count += 1
                
            except Exception as e:
                print(f"   ⚠️  Error processing sample: {e}")
                continue
        
        print(f"✅ Loaded {len(images)} images")
        
        return {
            'images': np.array(images),
            'labels': np.array(labels),
            'product_ids': np.array(product_ids)
        }
    
    def get_batch(self, images, labels=None, batch_size=None, shuffle=True):
        """
        Create batches for processing
        
        Args:
            images: numpy array of images
            labels: optional labels
            batch_size: batch size
            shuffle: whether to shuffle
            
        Yields:
            tuple: (batch_images, batch_labels) or batch_images
        """
        batch_size = batch_size or self.batch_size
        n_samples = len(images)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_images = images[batch_indices]
            
            if labels is not None:
                batch_labels = labels[batch_indices]
                yield batch_images, batch_labels
            else:
                yield batch_images
    
    def save_processed_dataset(self, data, split='train'):
        """
        Save processed dataset to disk
        
        Args:
            data: dict with 'images', 'labels', 'product_ids'
            split: 'train' or 'test'
        """
        save_path = self.processed_path / f"{split}_data.npz"
        
        print(f"💾 Saving {split} dataset to {save_path}...")
        np.savez(
            save_path,
            images=data['images'],
            labels=data['labels'],
            product_ids=data['product_ids']
        )
        print(f"✅ Saved! Size: {save_path.stat().st_size / (1024**2):.2f} MB")
    
    def load_processed_dataset(self, split='train'):
        """
        Load previously processed dataset from disk
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            dict: Contains images, labels, product_ids
        """
        load_path = self.processed_path / f"{split}_data.npz"
        
        if not load_path.exists():
            print(f"❌ Dataset not found at {load_path}")
            return None
        
        print(f"📂 Loading processed {split} dataset...")
        data = np.load(load_path)
        
        return {
            'images': data['images'],
            'labels': data['labels'],
            'product_ids': data['product_ids']
        }
    
    def get_dataset_info(self):
        """Return dataset statistics"""
        train_ds, test_ds, info = self.download_dataset()
        
        if info:
            return {
                'train_samples': info.splits['train'].num_examples,
                'test_samples': info.splits['test'].num_examples,
                'total_samples': info.splits['train'].num_examples + info.splits['test'].num_examples,
                'img_shape': info.features['image'].shape
            }
        return None


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    
    # Load dataset (first time will download)
    train_data = loader.load_and_process_dataset(split='train', sample_size=1000)
    
    if train_data:
        print(f"\n📊 Dataset Info:")
        print(f"   Images shape: {train_data['images'].shape}")
        print(f"   Labels shape: {train_data['labels'].shape}")
        print(f"   Unique labels: {len(np.unique(train_data['labels']))}")
        
        # Save processed data
        loader.save_processed_dataset(train_data, split='train')
