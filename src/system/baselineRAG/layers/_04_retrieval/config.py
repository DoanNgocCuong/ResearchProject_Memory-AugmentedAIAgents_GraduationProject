"""
Configuration settings for the retrieval system.
"""

from typing import List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables
load_dotenv()
print("Đang load .env từ:", find_dotenv())

# API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate required environment variables
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY is required in .env file")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL is required in .env file")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is required in .env file")

# Default configuration
DEFAULT_K = 4
DEFAULT_HYBRID_WEIGHTS = [0.7, 0.3]  # [vector_weight, keyword_weight]
DEFAULT_VECTOR_STORE_TYPE = "qdrant"
DEFAULT_COLLECTION_NAME = "documents"

# Embeddings configuration
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
try:
    INFERENCE_CLIENT = InferenceClient(provider="hf-inference", api_key=HUGGINGFACE_API_KEY)
except Exception as e:
    print(f"Lỗi khi khởi tạo InferenceClient: {e}")
    print("Vui lòng đảm bảo HUGGINGFACE_API_KEY đã được thiết lập chính xác trong file .env")
    raise

def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a text using Hugging Face Inference API.
    
    Args:
        text: Input text to get embedding for
        
    Returns:
        List of floats representing the embedding
    """
    if not text:
        return []
    try:
        embedding = INFERENCE_CLIENT.feature_extraction(model=EMBEDDINGS_MODEL_NAME, text=text)
        return embedding
    except Exception as e:
        print(f"Lỗi khi lấy embedding cho văn bản '{text[:50]}...': {e}")
        return None 