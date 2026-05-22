"""
Streamlit Web Application
Interactive UI for Visual Product Similarity System
"""

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import base64
from io import BytesIO
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_extractor import FeatureExtractor
from src.indexing import FAISSIndexing
from src.retrieval import SimilaritySearch
from config import (
    APP_TITLE, APP_DESCRIPTION, IMG_SIZE, NORMALIZE_MEAN, 
    NORMALIZE_STD, MODEL_DIR, DATA_DIR
)


# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
    }
    .product-card {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .similarity-score {
        font-size: 24px;
        font-weight: bold;
        color: #4ECDC4;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Configuration")

with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "🤖 Select Feature Extractor Model",
        ["ResNet50", "EfficientNet-B0"]
    )
    model_map = {"ResNet50": "resnet50", "EfficientNet-B0": "efficientnet_b0"}
    selected_model = model_map[model_choice]
    
    # Number of results
    k_results = st.slider(
        "📊 Number of Similar Products to Show",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )
    
    # Similarity threshold
    sim_threshold = st.slider(
        "🎯 Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )
    
    st.divider()
    
    # Information
    st.info(
        "💡 **How it works:**\n"
        "1. Upload or select a product image\n"
        "2. System extracts visual features\n"
        "3. Searches database for similar items\n"
        "4. Returns Top-K recommendations"
    )


# ==================== MAIN CONTENT ====================
st.markdown(f"<h1 class='main-header'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-size: 16px;'>{APP_DESCRIPTION}</p>", 
            unsafe_allow_html=True)

# ==================== INITIALIZATION ====================
@st.cache_resource
def load_models():
    """Load models and index"""
    try:
        # Load feature extractor
        feature_extractor = FeatureExtractor(model_name='resnet50')
        
        # Load index
        indexing = FAISSIndexing()
        index = indexing.load_index()
        
        # Load embeddings
        embeddings_path = DATA_DIR / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
        else:
            embeddings = None
        # Try to load processed dataset (images + product ids + optional names)
        processed_path = DATA_DIR / "processed" / "train_data.npz"
        product_images = None
        product_ids = None
        product_names = None

        if processed_path.exists():
            try:
                data = np.load(processed_path, allow_pickle=True)
                # images saved are preprocessed (normalized). Keep as-is and denormalize for display
                if 'images' in data:
                    product_images = data['images']
                if 'product_ids' in data:
                    product_ids = data['product_ids']
                if 'product_names' in data:
                    product_names = data['product_names']
            except Exception as e:
                st.warning(f"Could not load processed dataset: {e}")

        # Fallback product_ids when embeddings available
        if product_ids is None and embeddings is not None:
            product_ids = np.arange(len(embeddings))

        return feature_extractor, index, embeddings, product_ids, product_images, product_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None


def preprocess_image(image):
    """Preprocess PIL image"""
    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    # Convert to numpy
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Normalize
    image_array = (image_array - np.array(NORMALIZE_MEAN)) / np.array(NORMALIZE_STD)
    
    return image_array


def image_to_data_uri(img, width=120):
    """Convert PIL image to a base64 data URI for HTML embedding."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def display_results(query_image, results, k, product_images=None, product_ids_map=None, product_names=None):
    """Display search results with a thumbnail-enabled HTML table."""

    # Prepare rows with image URIs
    rows = []
    for i in range(min(k, len(results['product_ids']))):
        sim = float(results['similarities'][i])
        pid = int(results['product_ids'][i])
        product_name = None
        img_src = ""

        if product_images is not None and product_ids_map is not None:
            map_idx = product_ids_map.get(pid, None)
            if map_idx is None:
                try:
                    map_idx = int(pid)
                except Exception:
                    map_idx = None
            if map_idx is not None and 0 <= map_idx < len(product_images):
                img_arr = product_images[map_idx]
                img = denormalize_to_pil(img_arr)
                if img is not None:
                    img_src = image_to_data_uri(img, width=100)

        if product_names is not None:
            try:
                if product_ids_map and pid in product_ids_map:
                    product_name = str(product_names[product_ids_map[pid]])
                else:
                    product_name = str(product_names[int(pid)])
            except Exception:
                product_name = None

        rows.append({
            'Rank': i + 1,
            'Product ID': pid,
            'Product Name': product_name or "N/A",
            'Similarity': f"{sim:.4f}",
            'Image': img_src
        })

    # Build HTML table
    table_rows = []
    for row in rows:
        img_html = f'<img src="{row["Image"]}" width="100" style="border:1px solid #ddd;border-radius:8px;padding:2px;"/>' if row['Image'] else "No image"
        table_rows.append(
            f"<tr>"
            f"<td style='padding:8px;border:1px solid #ddd;text-align:center;width:5%;'>{row['Rank']}</td>"
            
            f"<td style='padding:8px;border:1px solid #ddd;text-align:center;width:15%;'>{row['Product ID']}</td>"
            # f"<td style='padding:8px;border:1px solid #ddd;text-align:left;max-width:220px;word-break:break-word;'>" 
            # f"{row['Product Name']}</td>"
            f"<td style='padding:8px;border:1px solid #ddd;text-align:center;width:15%;'>{img_html}</td>"
            f"<td style='padding:8px;border:1px solid #ddd;text-align:center;width:15%;'>{row['Similarity']}</td>"
            f"</tr>"
        )

    table_html = (
        "<div style='overflow-x:auto; width:100%;'>"
        "<table style='width:100%;border-collapse:collapse;font-family:Arial, sans-serif;table-layout:fixed;word-wrap:break-word;'>"
        "<thead>"
        "<tr style='background:#f4f4f4;'>"
        "<th style='padding:10px;border:1px solid #ddd;width:10%;'>Rank</th>"
        
        "<th style='padding:10px;border:1px solid #ddd;width:15%;'>Product ID</th>"
        # "<th style='padding:10px;border:1px solid #ddd;max-width:220px;white-space:normal;'>Name</th>"
        "<th style='padding:10px;border:1px solid #ddd;width:15%;'>Image</th>"
        "<th style='padding:10px;border:1px solid #ddd;width:15%;'>Similarity</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(table_rows) +
        "</tbody>"
        "</table>"
        "</div>"
    )

    st.subheader("🏆 Top Result")
    if rows:
        top1 = rows[0]
        if top1['Image']:
            st.image(top1['Image'], caption=f"#{top1['Rank']} - {top1['Product Name']} ({top1['Similarity']})", width=260)
        st.markdown(
            f"**Product ID:** {top1['Product ID']}<br/>"
            f"**Name:** {top1['Product Name']}<br/>"
            f"**Similarity:** {top1['Similarity']}",
            unsafe_allow_html=True
        )

    st.subheader(f"🔢 Top-{k} Matches")
    if rows:
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.warning("No results to display.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Results Returned", len(rows))
    with col_b:
        if rows:
            avg_sim = sum(float(r['Similarity']) for r in rows) / len(rows)
            st.metric("Avg Similarity", f"{avg_sim:.4f}")


def denormalize_to_pil(img_array):
    """Convert normalized numpy image back to PIL Image for display"""
    try:
        arr = img_array.copy()
        arr = (arr * np.array(NORMALIZE_STD)) + np.array(NORMALIZE_MEAN)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)
    except Exception:
        try:
            scaled = np.clip((img_array * 255), 0, 255).astype(np.uint8)
            return Image.fromarray(scaled)
        except Exception:
            return None


# ==================== MAIN APP ====================
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📚 How It Works", "ℹ️ About"])

with tab1:
    st.header("Search for Similar Products")
    
    # Load models
    feature_extractor, index, embeddings, product_ids, product_images, product_names = load_models()
    
    if index is None or embeddings is None:
        st.error("❌ Models not loaded! Please check configuration.")
        st.info("Run the main pipeline first: `python main.py`")
    else:
        # Initialize similarity search
        search = SimilaritySearch(index, embeddings, product_ids)
        # Build mapping from product_id -> index in product_images if available
        product_ids_map = None
        if product_images is not None and product_ids is not None:
            try:
                product_ids_map = {int(pid): idx for idx, pid in enumerate(product_ids)}
            except Exception:
                product_ids_map = None
        
        # Upload image
        uploaded_file = st.file_uploader(
            "📤 Upload a product image",
            type=["jpg", "jpeg", "png"],
            help="JPG, JPEG, or PNG format"
        )
              
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Uploaded Image", width=300)
            
            # Preprocess
            with st.spinner("🔄 Processing image..."):
                processed_image = preprocess_image(image)
                
                # Extract embedding
                query_embedding = feature_extractor.extract_embedding(processed_image)
                
                # Search
                results = search.search_similar_products(query_embedding, k=k_results)
                
                # Display results
                with col2:
                    display_results(image, results, k_results, product_images=product_images, product_ids_map=product_ids_map, product_names=product_names)
            
            # Download results
            if st.button("📥 Download Results as JSON"):
                import json
                results_json = search.format_results(results)
                st.download_button(
                    label="Download",
                    data=json.dumps(results_json, indent=2),
                    file_name="search_results.json",
                    mime="application/json"
                )
        else:
            st.info("👆 Upload an image to get started!")

with tab2:
    st.header("How Does It Work?")
    
    st.subheader("🏗️ Architecture")
    st.markdown("""
    ### 5-Step Pipeline:
    
    1. **📸 Image Upload**: User uploads product image
    2. **🧠 Feature Extraction**: CNN extracts visual features (embeddings)
    3. **⚡ Similarity Search**: FAISS finds similar embeddings
    4. **🏆 Ranking**: Results ranked by similarity score
    5. **📊 Display**: Top-K similar products shown to user
    """)
    
    st.subheader("🤖 Technical Details")
    st.markdown("""
    - **Model**: ResNet50 (pretrained on ImageNet)
    - **Embeddings**: 2048-dimensional feature vectors
    - **Database**: FAISS IVF index for fast retrieval
    - **Similarity**: Cosine similarity on normalized vectors
    - **Speed**: ~1-50ms per query
    """)
    
    st.subheader("📚 Key Concepts")
    
    concept1, concept2 = st.columns(2)
    
    with concept1:
        st.markdown("""
        #### Transfer Learning
        Using pretrained CNN weights to extract features
        - Learned on 1M+ images (ImageNet)
        - Captures visual patterns
        - Fast & efficient
        """)
    
    with concept2:
        st.markdown("""
        #### Vector Search
        Finding similar items in embedding space
        - FAISS index for speed
        - Approximate nearest neighbors
        - Production-ready
        """)

with tab3:
    st.header("About This Project")
    
    st.markdown(f"""
    ### 🛍️ Visual Product Similarity System
    
    **Objective**: Build an Amazon-style image-based recommendation system
    
    **Skills Demonstrated**:
    - ✅ Deep Learning (Transfer Learning)
    - ✅ Computer Vision (CNN Feature Extraction)
    - ✅ Vector Databases (FAISS)
    - ✅ Similarity Search Algorithms
    - ✅ Web Applications (Streamlit)
    
    **Dataset**: Stanford Online Products
    - ~120,000 product images
    - Multiple product categories
    - High-quality images with annotations
    
    **Use Cases**:
    - 🔍 Image-based product search
    - 📈 Recommendation engine
    - 🛒 E-commerce discovery
    - 👗 Fashion/furniture matching
    """)
    
    st.divider()
    
    st.markdown("""
    ### 📊 Performance Metrics
    
    | Metric | Value |
    |--------|-------|
    | Query Latency | 1-50ms |
    | Embedding Dimension | 2048 |
    | Index Type | FAISS IVF |
    | Model | ResNet50 |
    
    ### 🔗 Resources
    - [Stanford Online Products](https://www.tensorflow.org/datasets/catalog/stanford_online_products)
    - [FAISS Documentation](https://github.com/facebookresearch/faiss)
    - [ResNet Paper](https://arxiv.org/abs/1512.03385)
    """)


# ==================== FOOTER ====================
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Built with 💖 using Streamlit | Powered by Deep Learning
    </div>
    """, unsafe_allow_html=True)
