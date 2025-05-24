"""
Hybrid retriever implementation combining vector and BM25 search.
"""

import os
import sys
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from base import BaseRetriever
from vector_retriever import VectorRetriever
from bm25_retriever import BM25Retriever

# Add the retrievers directory to Python path
sys.path.append(os.path.dirname(__file__))

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector and BM25 search."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        documents: List[Document],
        weights: Optional[List[float]] = None,
        k: int = 4
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for similarity search
            documents: Documents for BM25 search
            weights: Weights for each retriever [vector_weight, bm25_weight]
            k: Number of documents to return
        """
        super().__init__()
        
        # Create individual retrievers
        vector_retriever = VectorRetriever(vector_store, k=k)
        bm25_retriever = BM25Retriever(documents, k=k)
        
        # Create ensemble retriever with Runnable
        self.retriever = EnsembleRetriever(
            retrievers=[
                RunnablePassthrough() | vector_retriever.retrieve_documents,
                RunnablePassthrough() | bm25_retriever.retrieve_documents
            ],
            weights=weights or [0.7, 0.3]  # Default weights
        )
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid search."""
        return self.retriever.invoke(query) 