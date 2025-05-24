"""
BM25 retriever implementation.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from .base import BaseRetriever

class BM25Retriever(BaseRetriever):
    """BM25 retriever implementation."""
    
    def __init__(
        self,
        documents: List[Document],
        k: int = 4
    ):
        if not documents:
            raise ValueError("Documents are required for BM25 retriever")
            
        self.documents = documents
        self.k = k
        self.retriever = BM25Retriever.from_documents(documents)
        self.retriever.k = k
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents using BM25 search."""
        if k is not None:
            self.retriever.k = k
        return self.retriever.get_relevant_documents(query) 