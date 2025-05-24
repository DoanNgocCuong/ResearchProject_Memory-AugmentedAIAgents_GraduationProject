"""
Qdrant vector store implementation.
"""

from pathlib import Path
import sys
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from huggingface_hub import InferenceClient
from .base import BaseVectorStore
from config import (
    QDRANT_URL, 
    QDRANT_API_KEY, 
    DEFAULT_COLLECTION_NAME,
    HUGGINGFACE_API_KEY,
    EMBEDDINGS_MODEL_NAME
)

# Setup paths
current_file = Path(__file__)
retrieval_dir = current_file.parent.parent
sys.path.append(str(retrieval_dir))

class QdrantStore(BaseVectorStore):
    """Qdrant vector store implementation."""
    
    _clients: Dict[str, QdrantClient] = {}
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embeddings = None
    ):
        """Initialize Qdrant store with lazy loading."""
        self.url = url or QDRANT_URL
        self.api_key = api_key or QDRANT_API_KEY
        self.collection_name = collection_name
        self.embeddings = embeddings
        
        if not self.url or not self.api_key:
            raise ValueError("Qdrant URL and API key are required")
            
        # Initialize HuggingFace client
        self.hf_client = InferenceClient(
            provider="hf-inference",
            api_key=HUGGINGFACE_API_KEY
        )
    
    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client using singleton pattern."""
        client_key = f"{self.url}:{self.api_key}"
        if client_key not in self._clients:
            self._clients[client_key] = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )
        return self._clients[client_key]
    
    def create_store(self, documents: List[Document]) -> None:
        """Create a Qdrant vector store from documents."""
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Size for paraphrase-multilingual-mpnet-base-v2
                    distance=models.Distance.COSINE
                )
            )
            
        # Add documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Get embeddings using HuggingFace client
        embeddings = []
        for text in texts:
            try:
                embedding = self.hf_client.feature_extraction(
                    model=EMBEDDINGS_MODEL_NAME,
                    text=text
                )
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    print(f"Warning: Could not get embedding for text: {text[:50]}...")
            except Exception as e:
                print(f"Error getting embedding: {e}")
                continue
        
        # Only proceed if we have valid embeddings
        if embeddings:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=i,
                        vector=embedding,
                        payload={"text": texts[i], "metadata": metadatas[i]}
                    )
                    for i, embedding in enumerate(embeddings)
                ]
            )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        try:
            # Get query embedding
            query_vector = self.hf_client.feature_extraction(
                model=EMBEDDINGS_MODEL_NAME,
                text=query
            )
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
                score_threshold=0.5
            )
            
            # Convert results to Documents
            documents = []
            for result in search_results:
                payload = result.payload
                documents.append(
                    Document(
                        page_content=payload.get("text", ""),
                        metadata=payload.get("metadata", {})
                    )
                )
            
            return documents
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_retriever(self, k: int = 4):
        """Get a retriever from the Qdrant store."""
        return self.similarity_search 