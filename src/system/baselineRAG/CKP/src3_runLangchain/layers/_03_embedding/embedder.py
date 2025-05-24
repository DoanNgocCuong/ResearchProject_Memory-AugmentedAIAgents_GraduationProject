"""
This module helps convert text into numbers (vectors) and store them.
This makes it easy to find similar texts quickly.
"""

from typing import List, Optional, Dict, Any, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Qdrant, Milvus, Chroma
from qdrant_client import QdrantClient
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import numpy as np
from dotenv import load_dotenv
import os
import sys
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import faiss
    except ImportError:
        print("FAISS is not installed. Please install it with:")
        print("pip install faiss-cpu")
        sys.exit(1)
        
    try:
        import qdrant_client
    except ImportError:
        print("Qdrant client is not installed. Please install it with:")
        print("pip install qdrant-client")
        sys.exit(1)
        
    try:
        import pymilvus
    except ImportError:
        print("PyMilvus is not installed. Please install it with:")
        print("pip install pymilvus")
        sys.exit(1)
        
    # try:
    #     import chromadb
    # except ImportError:
    #     print("ChromaDB is not installed. Please install it with:")
    #     print("pip install chromadb")
    #     sys.exit(1)

class DocumentEmbedder:
    """
    A class that helps convert text into numbers and store them.
    
    This class can:
    - Convert text into numbers (vectors)
    - Store vectors in different ways
    - Find similar texts quickly
    
    It supports different ways to store vectors:
    - FAISS (fast search)
    - Qdrant (powerful vector database)
    - Milvus (scalable vector database)
    - Chroma (simple and fast)
    """
    
    def __init__(
        self,
        embeddings_model: Optional[HuggingFaceEmbeddings] = None,
        vector_store_type: str = "faiss"
    ):
        """
        Start the DocumentEmbedder with optional AI model and storage type.
        
        Args:
            embeddings_model: Optional AI model for converting text to numbers
            vector_store_type: How to store vectors ('faiss', 'qdrant', 'milvus', 'chroma')
            
        Example:
            >>> from langchain_community.embeddings import HuggingFaceEmbeddings
            >>> embeddings = HuggingFaceEmbeddings()
            >>> embedder = DocumentEmbedder(embeddings_model=embeddings)
        """
        # Check dependencies
        check_dependencies()
        
        # Initialize HuggingFaceEmbeddings with a small, fast model
        if embeddings_model is None:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            self.embeddings_model = embeddings_model
            
        self.vector_store_type = vector_store_type
        self.vector_store = None

    def create_vector_store(
        self,
        documents: List[Document],
        persist_directory: Optional[str] = None
    ) -> VectorStore:
        """
        Convert documents into vectors and store them.
        
        Args:
            documents: List of documents to convert
            persist_directory: Where to save the vectors (optional)
            
        Returns:
            A store containing the document vectors
            
        Example:
            >>> store = embedder.create_vector_store(documents)
            >>> print(f"Created store with {len(documents)} documents")
        """
        try:
            if self.vector_store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings_model
                )
                if persist_directory:
                    self.vector_store.save_local(persist_directory)
                    
            elif self.vector_store_type == "qdrant":
                if not persist_directory:
                    persist_directory = "./qdrant_data"
                    
                # Create directory if it doesn't exist
                os.makedirs(persist_directory, exist_ok=True)
                
                client = QdrantClient(path=persist_directory)
                self.vector_store = Qdrant.from_documents(
                    documents=documents,
                    embedding=self.embeddings_model,
                    client=client,
                    collection_name="documents"
                )
                
            elif self.vector_store_type == "milvus":
                if not persist_directory:
                    persist_directory = "documents"
                
                try:
                    # Connect to Milvus
                    connections.connect(host="localhost", port="19530")
                except Exception as e:
                    raise RuntimeError(
                        "Could not connect to Milvus server. "
                        "Please make sure Milvus is running on localhost:19530. "
                        f"Error: {str(e)}"
                    )
                
                # Create collection if it doesn't exist
                if not Collection(persist_directory).exists():
                    # Define schema
                    dim = 384  # MiniLM-L6-v2 embedding dimension
                    fields = [
                        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                        FieldSchema(name="metadata", dtype=DataType.JSON)
                    ]
                    schema = CollectionSchema(fields=fields)
                    Collection(name=persist_directory, schema=schema)
                
                self.vector_store = Milvus.from_documents(
                    documents=documents,
                    embedding=self.embeddings_model,
                    collection_name=persist_directory
                )
                
            elif self.vector_store_type == "chroma":
                if not persist_directory:
                    persist_directory = "./chroma_data"
                    
                # Create directory if it doesn't exist
                os.makedirs(persist_directory, exist_ok=True)
                
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings_model,
                    persist_directory=persist_directory
                )
                
            else:
                raise ValueError(f"Unknown vector store type: {self.vector_store_type}")
                
            return self.vector_store
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to create vector store. Error: {str(e)}\n"
                "Please check that all required services are running and dependencies are installed."
            )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to the vector store.
        
        Args:
            documents: List of new documents to add
            
        Example:
            >>> embedder.add_documents(new_documents)
            >>> print("Added new documents to store")
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created yet")
            
        self.vector_store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Find documents similar to the query.
        
        Args:
            query: Text to find similar documents for
            k: How many similar documents to find
            
        Returns:
            List of similar documents
            
        Example:
            >>> similar_docs = embedder.similarity_search("What is RAG?")
            >>> print(f"Found {len(similar_docs)} similar documents")
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created yet")
            
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        Find similar documents with their similarity scores.
        
        Args:
            query: Text to find similar documents for
            k: How many similar documents to find
            
        Returns:
            List of (document, score) pairs
            
        Example:
            >>> results = embedder.similarity_search_with_score("What is RAG?")
            >>> for doc, score in results:
            >>>     print(f"Score: {score:.2f}")
            >>>     print(doc.page_content)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created yet")
            
        return self.vector_store.similarity_search_with_score(query, k=k)

    def save_vector_store(self, persist_directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            persist_directory: Where to save the store
            
        Example:
            >>> embedder.save_vector_store("./my_store")
            >>> print("Saved vector store to disk")
        """
        if self.vector_store is None:
            raise ValueError("Vector store not created yet")
            
        if self.vector_store_type == "faiss":
            self.vector_store.save_local(persist_directory)
        elif self.vector_store_type == "qdrant":
            self.vector_store.client.persist(persist_directory)
        elif self.vector_store_type == "milvus":
            # Milvus handles persistence automatically
            pass
        elif self.vector_store_type == "chroma":
            self.vector_store.persist()

    def load_vector_store(
        self,
        persist_directory: str,
        vector_store_type: Optional[str] = None
    ) -> VectorStore:
        """
        Load a vector store from disk.
        
        Args:
            persist_directory: Where the store is saved
            vector_store_type: How the store was saved (optional)
            
        Returns:
            The loaded vector store
            
        Example:
            >>> store = embedder.load_vector_store("./my_store")
            >>> print("Loaded vector store from disk")
        """
        store_type = vector_store_type or self.vector_store_type
        
        if store_type == "faiss":
            self.vector_store = FAISS.load_local(
                persist_directory,
                self.embeddings_model
            )
        elif store_type == "qdrant":
            client = QdrantClient(path=persist_directory)
            self.vector_store = Qdrant(
                client=client,
                collection_name="documents",
                embedding_function=self.embeddings_model.embed_query
            )
        elif store_type == "milvus":
            connections.connect(host="localhost", port="19530")
            self.vector_store = Milvus(
                collection_name=persist_directory,
                embedding_function=self.embeddings_model.embed_query
            )
        elif store_type == "chroma":
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings_model
            )
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
            
        return self.vector_store

def create_vectordb(documents: List[Document], persist_directory: str = "data/chroma") -> Chroma:
    """
    Create a vector database from documents using OpenAI embeddings
    
    Args:
        documents: List of documents to embed
        persist_directory: Directory to store the vector database
        
    Returns:
        Chroma vector database instance
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector database
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectordb.persist()
    
    return vectordb

if __name__ == "__main__":
    """
    This part runs when you run this file directly.
    It shows examples of how to use the DocumentEmbedder class.
    """
    from langchain_core.documents import Document
    
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
    
    # # Test Chroma vector store
    # print("\nTesting Chroma vector store...")
    # try:
    #     embedder = DocumentEmbedder(vector_store_type="chroma")
    #     store = embedder.create_vector_store(sample_docs)
    #     results = embedder.similarity_search("What is RAG?")
    #     print(f"Found {len(results)} similar documents using Chroma")
    # except Exception as e:
    #     print(f"Chroma test failed: {e}")
        
    # Test FAISS vector store
    print("Testing FAISS vector store...")
    try:
        embedder = DocumentEmbedder(vector_store_type="faiss")
        store = embedder.create_vector_store(sample_docs)
        results = embedder.similarity_search("What is RAG?")
        print(f"Found {len(results)} similar documents using FAISS")
    except Exception as e:
        print(f"FAISS test failed: {e}")
    
    # Test Qdrant vector store
    print("\nTesting Qdrant vector store...")
    try:
        embedder = DocumentEmbedder(vector_store_type="qdrant")
        store = embedder.create_vector_store(sample_docs)
        results = embedder.similarity_search("What is RAG?")
        print(f"Found {len(results)} similar documents using Qdrant")
    except Exception as e:
        print(f"Qdrant test failed: {e}")
    
    # Test Milvus vector store
    print("\nTesting Milvus vector store...")
    try:
        embedder = DocumentEmbedder(vector_store_type="milvus")
        store = embedder.create_vector_store(sample_docs)
        results = embedder.similarity_search("What is RAG?")
        print(f"Found {len(results)} similar documents using Milvus")
    except Exception as e:
        print(f"Milvus test failed: {e}")
        
    # Test vector database creation
    from layers._01_data_ingestion.loader import load_faq_data, preprocess_faq_data
    
    # Load and preprocess documents
    documents = load_faq_data("../../data/TinhNangApp.json")
    processed_docs = preprocess_faq_data(documents)
    
    # Create vector database
    print("Creating vector database...")
    vectordb = create_vectordb(processed_docs)
    print(f"Vector database created with {len(documents)} documents")
    
    # Test similarity search
    query = "What are the main features of the app?"
    results = vectordb.similarity_search(query, k=3)
    print("\nSimilarity search results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content[:100]}...")
        

