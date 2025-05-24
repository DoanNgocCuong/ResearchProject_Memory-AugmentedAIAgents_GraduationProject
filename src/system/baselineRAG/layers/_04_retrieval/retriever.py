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
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

# Setup paths
current_file = Path(__file__)
retrieval_dir = current_file.parent
sys.path.append(str(retrieval_dir))

from config import (
    DEFAULT_K,
    DEFAULT_HYBRID_WEIGHTS,
    DEFAULT_VECTOR_STORE_TYPE,
    
)
from vector_stores.qdrant_store import QdrantStore
from retrievers.vector_retriever import VectorRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever

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
        collection_name: str = "test_collection"
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
        self.embeddings_model = embeddings_model
        self.hybrid_weights = hybrid_weights
        self.k = k
        
        # Initialize vector store
        if vector_store_type == "qdrant":
            self.vector_store = QdrantStore(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                embeddings=self.embeddings_model
            )
        
        # Initialize retriever
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the appropriate retriever based on type."""
        if self.retriever_type == "vector":
            if not self.vector_store:
                raise ValueError("Vector store is required for vector retriever")
            self.retriever = VectorRetriever(
                vector_store=self.vector_store,
                k=self.k
            )
            
        elif self.retriever_type == "bm25":
            if not self.documents:
                raise ValueError("Documents are required for BM25 retriever")
            self.retriever = BM25Retriever(
                documents=self.documents,
                k=self.k
            )
            
        elif self.retriever_type == "hybrid":
            if not self.vector_store or not self.documents:
                raise ValueError("Both vector store and documents are required for hybrid retriever")
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                documents=self.documents,
                weights=self.hybrid_weights,
                k=self.k
            )
            
        elif self.retriever_type == "compression":
            if not self.vector_store:
                raise ValueError("Vector store is required for compression retriever")
            self.retriever = CompressionRetriever(
                vector_store=self.vector_store,
                k=self.k
            )
            
        else:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}")
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents using the selected retriever."""
        return self.retriever.retrieve_documents(query)

if __name__ == "__main__":
    """
    This part runs when you run this file directly.
    It shows examples of how to use the DocumentRetriever class.
    """
    from langchain_core.documents import Document
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    from dotenv import load_dotenv
    import os
    
    # Load environment variables
    load_dotenv()
    
    # Initialize embedding model
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="The quick brown fox jumps over the lazy dog",
            metadata={"source": "test1.txt"}
        ),
        Document(
            page_content="A fast orange fox leaps across a sleepy canine",
            metadata={"source": "test2.txt"}
        ),
        Document(
            page_content="The weather is beautiful today",
            metadata={"source": "test3.txt"}
        )
    ]
    
    # Test vector retriever with Qdrant
    print("\nTesting vector retriever with Qdrant...")
    try:
        print("Initializing DocumentRetriever with vector store...")
        retriever = DocumentRetriever(
            retriever_type="vector",
            vector_store_type="qdrant",
            documents=sample_docs,
            embeddings_model=embeddings,
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY")
        )
        print(f"Qdrant URL: {os.getenv('QDRANT_URL')}")
        print(f"Collection name: {retriever.vector_store.collection_name}")
        print(f"Number of documents to process: {len(sample_docs)}")
        
        print("\nAttempting to retrieve documents...")
        docs = retriever.retrieve_documents("fox jumping")
        print(f"Found {len(docs)} relevant documents using vector search")
        print("\nVector Search Results:")
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Vector retriever test failed: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
    
    # Test BM25 retriever
    print("\nTesting BM25 retriever...")
    try:
        retriever = DocumentRetriever(
            retriever_type="bm25",
            documents=sample_docs
        )
        docs = retriever.retrieve_documents("fox jumping")
        print(f"Found {len(docs)} relevant documents using BM25")
        print("\nBM25 Search Results:")
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"BM25 retriever test failed: {e}")
    
    # Test hybrid retriever with Qdrant
    print("\nTesting hybrid retriever with Qdrant...")
    try:
        retriever = DocumentRetriever(
            retriever_type="hybrid",
            vector_store_type="qdrant",
            documents=sample_docs,
            embeddings_model=embeddings,
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY")
        )
        docs = retriever.retrieve_documents("fox jumping")
        print(f"Found {len(docs)} relevant documents using hybrid search")
        print("\nHybrid Search Results:")
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Hybrid retriever test failed: {e}")
    

