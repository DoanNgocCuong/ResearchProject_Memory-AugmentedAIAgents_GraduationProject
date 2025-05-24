"""
Hybrid retriever implementation combining vector and BM25 search.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from .base import BaseRetriever
from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever
from ..vector_stores.base import BaseVectorStore

class HybridRetriever(BaseRetriever):
    """Hybrid retriever implementation."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        documents: List[Document],
        weights: List[float] = [0.7, 0.3],
        k: int = 4
    ):
        if not documents:
            raise ValueError("Documents are required for hybrid retriever")
            
        self.vector_store = vector_store
        self.documents = documents
        self.weights = weights
        self.k = k
        
        # Create individual retrievers
        vector_retriever = VectorRetriever(vector_store=vector_store, k=k)
        bm25_retriever = BM25Retriever(documents=documents, k=k)
        
        # Create ensemble retriever
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights
        )
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""
        if k is not None:
            for retriever in self.retriever.retrievers:
                if hasattr(retriever, "k"):
                    retriever.k = k
        return self.retriever.get_relevant_documents(query) 