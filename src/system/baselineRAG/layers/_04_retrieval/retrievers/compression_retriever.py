"""
Compression retriever implementation with LLM-based document compression.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from .base import BaseRetriever
from .vector_retriever import VectorRetriever
from ..vector_stores.base import BaseVectorStore
import os

class CompressionRetriever(BaseRetriever):
    """Compression retriever implementation."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        k: int = 4
    ):
        self.vector_store = vector_store
        self.k = k
        
        try:
            # Initialize LLM
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Create compressor
            compressor = LLMChainExtractor.from_llm(llm)
            
            # Create base retriever
            base_retriever = VectorRetriever(
                vector_store=vector_store,
                k=k
            )
            
            # Create compression retriever
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        except Exception as e:
            print(f"Warning: Could not initialize compression retriever: {e}")
            print("Falling back to vector retriever")
            self.retriever = VectorRetriever(
                vector_store=vector_store,
                k=k
            )
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents using compression search."""
        if k is not None and hasattr(self.retriever, "k"):
            self.retriever.k = k
        return self.retriever.get_relevant_documents(query) 