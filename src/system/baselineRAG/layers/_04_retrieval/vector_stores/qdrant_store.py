"""
Qdrant vector store implementation.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from .base import BaseVectorStore
from ..config import QDRANT_URL, QDRANT_API_KEY, DEFAULT_COLLECTION_NAME

class QdrantStore(BaseVectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function = None
    ):
        self.url = url or QDRANT_URL
        self.api_key = api_key or QDRANT_API_KEY
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.client = None
        self.store = None
        
        if not self.url or not self.api_key:
            raise ValueError("Qdrant URL and API key are required")
            
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
    
    def create_store(self, documents: List[Document]) -> VectorStore:
        """Create a Qdrant vector store from documents."""
        self.store = Qdrant.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            client=self.client,
            collection_name=self.collection_name
        )
        return self.store
    
    def get_retriever(self, k: int) -> VectorStore:
        """Get a retriever from the Qdrant store."""
        if not self.store:
            self.store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function
            )
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        ) 