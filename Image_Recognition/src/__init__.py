"""
Image Recognition Package
Visual Product Similarity & Recommendation System
"""

__version__ = "1.0.0"
__author__ = "Deep Learning Specialist"

from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .indexing import FAISSIndexing
from .retrieval import SimilaritySearch
from .evaluation import EvaluationMetrics

__all__ = [
    "DataLoader",
    "FeatureExtractor", 
    "FAISSIndexing",
    "SimilaritySearch",
    "EvaluationMetrics"
]
