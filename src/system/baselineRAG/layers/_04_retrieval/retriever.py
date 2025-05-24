"""
This module helps find relevant documents based on user questions.
It uses different methods to search and rank documents.
"""

from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from .config import (
    DEFAULT_K,
    DEFAULT_HYBRID_WEIGHTS,
    DEFAULT_VECTOR_STORE_TYPE,
    DEFAULT_EMBEDDINGS
)
from .vector_stores.qdrant_store import QdrantStore
from .retrievers.vector_retriever import VectorRetriever
from .retrievers.bm25_retriever import BM25Retriever
from .retrievers.hybrid_retriever import HybridRetriever
from .retrievers.compression_retriever import CompressionRetriever

# Load environment variables
load_dotenv()

class DocumentRetriever:
    """
    Main retriever class that combines different retrieval strategies.
    """
    
    def __init__(
        self,
        retriever_type: str = "vector",
        vector_store_type: str = DEFAULT_VECTOR_STORE_TYPE,
        documents: Optional[List[Document]] = None,
        embeddings_model = None,
        hybrid_weights: List[float] = DEFAULT_HYBRID_WEIGHTS,
        k: int = DEFAULT_K,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "documents"
    ):
        """
        Initialize the document retriever.
        
        Args:
            retriever_type: Type of retriever to use
            vector_store_type: Type of vector store to use
            documents: List of documents for BM25 search
            embeddings_model: Embeddings model to use
            hybrid_weights: Weights for hybrid search
            k: Number of documents to return
            qdrant_url: URL for Qdrant server
            qdrant_api_key: API key for Qdrant
            collection_name: Name of the Qdrant collection
        """
        self.retriever_type = retriever_type
        self.vector_store_type = vector_store_type
        self.documents = documents
        self.embeddings_model = embeddings_model or DEFAULT_EMBEDDINGS
        self.hybrid_weights = hybrid_weights
        self.k = k
        
        # Initialize vector store
        if vector_store_type == "qdrant":
            self.vector_store = QdrantStore(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                embedding_function=self.embeddings_model.embed_query
            )
        
        # Initialize retriever
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the appropriate retriever based on type."""
        if self.retriever_type == "vector":
            self.retriever = VectorRetriever(
                vector_store=self.vector_store,
                k=self.k
            )
        elif self.retriever_type == "bm25":
            self.retriever = BM25Retriever(
                documents=self.documents,
                k=self.k
            )
        elif self.retriever_type == "hybrid":
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                documents=self.documents,
                weights=self.hybrid_weights,
                k=self.k
            )
        elif self.retriever_type == "compression":
            self.retriever = CompressionRetriever(
                vector_store=self.vector_store,
                k=self.k
            )
        else:
            raise ValueError(f"Unsupported retriever type: {self.retriever_type}")
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve documents relevant to the query."""
        return self.retriever.retrieve_documents(query, k)

if __name__ == "__main__":
    """
    This part runs when you run this file directly.
    It shows examples of how to use the DocumentRetriever class.
    """
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="This is a sample document about RAG architecture.",
            metadata={"source": "test1"}
        ),
        Document(
            page_content="Another document explaining vector databases.",
            metadata={"source": "test2"}
        )
    ]
    
    # Create vector store
    vector_store = FAISS.from_documents(sample_docs, DEFAULT_EMBEDDINGS)
    
    # Test vector retriever
    print("\nTesting vector retriever...")
    try:
        retriever = DocumentRetriever(
            vector_store=vector_store,
            retriever_type="vector"
        )
        docs = retriever.retrieve_documents("What is RAG?")
        print(f"Found {len(docs)} relevant documents using vector search")
    except Exception as e:
        print(f"Vector retriever test failed: {e}")
    
    # Test BM25 retriever
    print("\nTesting BM25 retriever...")
    try:
        retriever = DocumentRetriever(
            documents=sample_docs,
            retriever_type="bm25"
        )
        docs = retriever.retrieve_documents("What is RAG?")
        print(f"Found {len(docs)} relevant documents using BM25")
    except Exception as e:
        print(f"BM25 retriever test failed: {e}")
    
    # Test hybrid retriever
    print("\nTesting hybrid retriever...")
    try:
        retriever = DocumentRetriever(
            vector_store=vector_store,
            documents=sample_docs,
            retriever_type="hybrid"
        )
        docs = retriever.retrieve_documents("What is RAG?")
        print(f"Found {len(docs)} relevant documents using hybrid search")
    except Exception as e:
        print(f"Hybrid retriever test failed: {e}")
    

