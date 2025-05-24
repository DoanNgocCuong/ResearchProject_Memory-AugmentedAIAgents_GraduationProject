"""
Vector-based retriever implementation.
"""

from typing import List, Optional
from langchain_core.documents import Document
from .base import BaseRetriever
from ..vector_stores.base import BaseVectorStore

class VectorRetriever(BaseRetriever):
    """Vector-based retriever implementation."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        k: int = 4
    ):
        self.vector_store = vector_store
        self.k = k
        self.retriever = vector_store.get_retriever(k)
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents using vector search."""
        if k is not None:
            self.retriever.search_kwargs["k"] = k
        return self.retriever.get_relevant_documents(query) 