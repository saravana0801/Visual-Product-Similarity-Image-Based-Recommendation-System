# Visual Product Similarity & Image-Based Recommendation System

A complete image-based recommendation pipeline that finds visually similar products using deep learning embeddings and FAISS similarity search.

## Project Overview

This repository implements an end-to-end system for matching product images by visual similarity, enabling image-based recommendations similar to Amazon's "Similar Items" feature.

Key capabilities:
- Extract visual embeddings from product images using pretrained CNNs (ResNet50 / EfficientNet-B0)
- Build a fast FAISS nearest neighbor index for similarity search
- Retrieve and rank Top-K visually similar products
- Filter duplicate product IDs and surface the next unique similar item
- Evaluate search quality using Precision@K, Recall@K, MAP, and MRR
- Run an interactive Streamlit dashboard for live demo and product search

## Features

- Feature extraction from product images using pretrained CNNs
- Vector indexing with FAISS (FLAT, IVF, HNSW)
- Cosine similarity search with normalized embeddings
- Duplicate result deduplication to avoid repeated product IDs
- Modular pipeline split into data loading, embedding extraction, indexing, retrieval, and evaluation
- Streamlit UI for image upload, search, and visualization
- CLI evaluation workflow for Step 5 metrics

## Repository Structure

```
ImageRecognition/
├── README.md
├── instruction.txt
├── requirements.txt
├── config.py
├── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── embeddings.npy
│   └── processed/
├── models/
│   └── faiss_index.bin
├── results/
│   └── pipeline_state.json
└── src/
    ├── data_loader.py
    ├── feature_extractor.py
    ├── indexing.py
    ├── retrieval.py
    └── evaluation.py
```

## Tech Stack

- Python 3.8+
- PyTorch / Torchvision
- FAISS
- Streamlit
- NumPy, Pandas, Pillow
- TensorFlow Datasets (Stanford Online Products)

## Installation

```bash
cd /Users/saravanakarthikeyan/Documents/AI_ML/Guvi_Project/ImageRecognition\ copy\ 2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Author
**Saravana Karthikeyan**  
Aspiring AI/ML Engineer  
