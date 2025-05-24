"""
Configuration settings for the retrieval system.
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_K = 4
DEFAULT_HYBRID_WEIGHTS = [0.7, 0.3]  # [vector_weight, keyword_weight]
DEFAULT_VECTOR_STORE_TYPE = "faiss"

# Embeddings configuration
DEFAULT_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_COLLECTION_NAME = "documents" 