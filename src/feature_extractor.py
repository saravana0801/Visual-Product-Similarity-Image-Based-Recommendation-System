"""
Feature Extractor Module
Extract embeddings using pretrained CNN models
Intermediate Level: Transfer learning, model architecture, embedding extraction
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURE_EXTRACTOR_MODEL, PRETRAINED_WEIGHTS, EMBEDDING_DIM, 
    BATCH_SIZE, DEVICE, USE_GPU, MODEL_DIR
)


class FeatureExtractor:
    """
    Extract visual embeddings from product images using pretrained CNN
    
    Key Concepts:
    - Transfer Learning: Use pretrained ImageNet weights
    - Feature Extraction: Remove classification layer, keep feature layers
    - Embedding: Dense vector representation of image content
    
    Architecture: ResNet50
    - Input: 224x224 RGB image
    - Output: 2048-dimensional embedding vector
    """
    
    def __init__(self, model_name=FEATURE_EXTRACTOR_MODEL, device=DEVICE):
        """
        Initialize feature extractor
        
        Args:
            model_name: 'resnet50' or 'efficientnet_b0'
            device: 'cuda' or 'cpu'
        """
        self.model_name = model_name
        self.device = device if (USE_GPU and torch.cuda.is_available()) else 'cpu'
        self.model_dir = MODEL_DIR
        
        print(f"🚀 Initializing {model_name} on {self.device}")
        
        self.model = self._load_model()
        self.embedding_dim = self._get_embedding_dim()
        
        print(f"✅ Feature Extractor ready!")
        print(f"   Model: {model_name}")
        print(f"   Embedding Dimension: {self.embedding_dim}")
        print(f"   Device: {self.device}")
    
    def _load_model(self):
        """
        Load pretrained model and remove classification layer
        
        Transfer Learning Concept:
        - Load weights trained on ImageNet (1000 classes)
        - Remove last fully-connected layer (classifier)
        - Keep convolutional layers (learned features)
        - Output: features from penultimate layer
        """
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove final classification layer
            model.fc = nn.Identity()
            
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            # Remove final classification layer
            model.classifier[1] = nn.Identity()
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Set to evaluation mode (disable dropout, batch norm updates)
        model.eval()
        model.to(self.device)
        
        # Freeze parameters (we're not training)
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _get_embedding_dim(self):
        """Get embedding dimension based on model"""
        if self.model_name == 'resnet50':
            return 2048
        elif self.model_name == 'efficientnet_b0':
            return 1280
        return 2048
    
    def extract_embedding(self, image):
        """
        Extract embedding for single image
        
        Args:
            image: numpy array (224, 224, 3) with values in [-1, 1]
            
        Returns:
            numpy array: embedding vector (embedding_dim,)
        """
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Add batch dimension
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)  # Change from (1, H, W, C) to (1, C, H, W)
        image = image.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(image)
        
        # Convert to numpy and squeeze
        embedding = embedding.cpu().numpy().squeeze()
        
        return embedding
    
    def extract_batch_embeddings(self, image_batch):
        """
        Extract embeddings for batch of images
        
        Args:
            image_batch: numpy array (batch_size, 224, 224, 3)
            
        Returns:
            numpy array: embeddings (batch_size, embedding_dim)
        """
        # Convert to tensor
        if isinstance(image_batch, np.ndarray):
            image_batch = torch.from_numpy(image_batch).float()

        image_batch = image_batch.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
        
        image_batch = image_batch.to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model(image_batch)
        
        # Convert to numpy
        embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def extract_embeddings_from_dataset(self, images, batch_size=None):
        """
        Extract embeddings for entire dataset
        
        Args:
            images: numpy array (n_samples, 224, 224, 3)
            batch_size: batch size for processing
            
        Returns:
            numpy array: embeddings (n_samples, embedding_dim)
        """
        batch_size = batch_size or BATCH_SIZE
        n_samples = len(images)
        
        all_embeddings = np.zeros((n_samples, self.embedding_dim), dtype=np.float32)
        
        print(f"🔄 Extracting embeddings from {n_samples} images...")
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Embedding extraction"):
            end_idx = min(i + batch_size, n_samples)
            batch_images = images[i:end_idx]
            
            batch_embeddings = self.extract_batch_embeddings(batch_images)
            all_embeddings[i:end_idx] = batch_embeddings
        
        print(f"✅ Extracted {n_samples} embeddings")
        print(f"   Shape: {all_embeddings.shape}")
        print(f"   Dtype: {all_embeddings.dtype}")
        
        return all_embeddings
    
    def save_embeddings(self, embeddings, save_path=None):
        """
        Save embeddings to disk
        
        Args:
            embeddings: numpy array (n_samples, embedding_dim)
            save_path: path to save (default: data/embeddings.npy)
        """
        if save_path is None:
            save_path = Path(__file__).parent.parent / "data" / "embeddings.npy"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving embeddings to {save_path}...")
        np.save(save_path, embeddings)
        print(f"✅ Saved! Size: {save_path.stat().st_size / (1024**2):.2f} MB")
    
    def load_embeddings(self, load_path=None):
        """
        Load embeddings from disk
        
        Args:
            load_path: path to load from (default: data/embeddings.npy)
            
        Returns:
            numpy array: embeddings
        """
        if load_path is None:
            load_path = Path(__file__).parent.parent / "data" / "embeddings.npy"
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            print(f"❌ Embeddings not found at {load_path}")
            return None
        
        print(f"📂 Loading embeddings from {load_path}...")
        embeddings = np.load(load_path)
        print(f"✅ Loaded! Shape: {embeddings.shape}")
        
        return embeddings
    
    def get_model_info(self):
        """Return model information"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'input_size': 224,
            'pretrained': True,
            'frozen': True
        }


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = FeatureExtractor(model_name='resnet50')
    
    # Create dummy image batch for testing
    dummy_images = np.random.randn(4, 224, 224, 3).astype(np.float32)
    dummy_images = (dummy_images - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Extract embeddings
    embeddings = extractor.extract_batch_embeddings(dummy_images)
    
    print(f"\n📊 Embeddings Info:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean(axis=0)[:5]}")
    print(f"   Std: {embeddings.std(axis=0)[:5]}")
    
    # Model info
    info = extractor.get_model_info()
    print(f"\n🤖 Model Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
