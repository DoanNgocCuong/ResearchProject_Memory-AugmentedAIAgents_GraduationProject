"""
Base class for vector stores.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def create_store(self, documents: List[Document]) -> VectorStore:
        """Create a vector store from documents."""
        pass
    
    @abstractmethod
    def get_retriever(self, k: int) -> VectorStore:
        """Get a retriever from the vector store."""
        pass 