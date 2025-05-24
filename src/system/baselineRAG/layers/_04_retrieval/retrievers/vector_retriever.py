"""
Vector-based retriever implementation.
"""

from pathlib import Path
import sys
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

# Setup paths
current_file = Path(__file__)
retrieval_dir = current_file.parent.parent
sys.path.append(str(retrieval_dir))

from retrievers.base import BaseRetriever
from vector_stores.qdrant_store import QdrantStore

class VectorRetriever(BaseRetriever):
    """Vector-based retriever using similarity search."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 4
    ):
        """
        Initialize vector retriever.
        
        Args:
            vector_store: Vector store for similarity search
            k: Number of documents to return
        """
        super().__init__()
        self.vector_store = vector_store
        self.k = k
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents using similarity search."""
        return self.vector_store.similarity_search(query, k=self.k) 