"""
This module helps break large documents into smaller pieces.
Breaking documents into smaller pieces helps the computer understand them better.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter
)
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class DocumentChunker:
    """
    A class that helps break documents into smaller pieces.
    
    This class can break documents in different ways:
    - By size (number of characters)
    - By sentences (using NLTK or spaCy)
    - By markdown headers
    - By special characters
    
    Breaking documents into smaller pieces helps the computer:
    - Understand the text better
    - Find information faster
    - Work with large documents
    """
    
    def __init__(self, embeddings_model: Optional[OpenAIEmbeddings] = None):
        """
        Start the DocumentChunker with optional AI model.
        
        Args:
            embeddings_model: Optional AI model for understanding text meaning
                            
        Example:
            >>> from langchain_openai import OpenAIEmbeddings
            >>> embeddings = OpenAIEmbeddings()
            >>> chunker = DocumentChunker(embeddings_model=embeddings)
        """
        self.embeddings_model = embeddings_model

    def chunk_by_size(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True
    ) -> List[Document]:
        """
        Break documents into pieces based on size.
        
        This method makes sure each piece is not too big.
        It can make pieces overlap to keep the meaning clear.
        
        Args:
            documents: List of documents to break
            chunk_size: Maximum size of each piece (default: 1000 characters)
            chunk_overlap: How much pieces should overlap (default: 200 characters)
            add_start_index: Keep track of where each piece starts (default: True)
            
        Returns:
            List of smaller document pieces
            
        Example:
            >>> chunker = DocumentChunker()
            >>> small_pieces = chunker.chunk_by_size(documents, chunk_size=500)
            >>> print(len(small_pieces))  # Shows how many pieces were made
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index
        )
        return splitter.split_documents(documents)

    def chunk_by_sentence(
        self,
        documents: List[Document],
        engine: str = "nltk",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Break documents into pieces based on sentences.
        
        This method uses NLTK or spaCy to find sentence breaks.
        It keeps sentences together to maintain meaning.
        
        Args:
            documents: List of documents to break
            engine: Which tool to use ('nltk' or 'spacy')
            chunk_size: Maximum size of each piece (default: 1000)
            chunk_overlap: How much pieces should overlap (default: 200)
            
        Returns:
            List of document pieces broken at sentences
            
        Example:
            >>> chunker = DocumentChunker()
            >>> sentence_pieces = chunker.chunk_by_sentence(documents, engine="nltk")
            >>> print(sentence_pieces[0].page_content)  # Shows first sentence piece
        """
        if engine == "nltk":
            splitter = NLTKTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif engine == "spacy":
            splitter = SpacyTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError("Engine must be 'nltk' or 'spacy'")
            
        return splitter.split_documents(documents)

    def chunk_markdown(
        self,
        documents: List[Document],
        headers_to_split_on: List[tuple] = None
    ) -> List[Document]:
        """
        Break markdown documents at headers.
        
        This method breaks documents at # headers in markdown.
        It keeps the structure of the document.
        
        Args:
            documents: List of markdown documents to break
            headers_to_split_on: Which headers to break at (default: #, ##, ###)
            
        Returns:
            List of document pieces broken at headers
            
        Example:
            >>> chunker = DocumentChunker()
            >>> header_pieces = chunker.chunk_markdown(markdown_docs)
            >>> print(header_pieces[0].metadata)  # Shows header information
        """
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        all_chunks = []
        for doc in documents:
            chunks = splitter.split_text(doc.page_content)
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_by_character(
        self,
        documents: List[Document],
        separator: str = "\n\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Break documents at special characters.
        
        This method breaks documents at characters like new lines.
        It's good for text with clear breaks.
        
        Args:
            documents: List of documents to break
            separator: Character to break at (default: new line)
            chunk_size: Maximum size of each piece (default: 1000)
            chunk_overlap: How much pieces should overlap (default: 200)
            
        Returns:
            List of document pieces broken at characters
            
        Example:
            >>> chunker = DocumentChunker()
            >>> line_pieces = chunker.chunk_by_character(documents, separator="\n")
            >>> print(line_pieces[0].page_content)  # Shows first line piece
        """
        splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

    def chunk_documents(
        self,
        documents: List[Document],
        strategy: str = "size",
        **kwargs
    ) -> List[Document]:
        """
        Break documents using any strategy.
        
        This is an easy way to break documents without knowing which specific method to use.
        
        Args:
            documents: List of documents to break
            strategy: How to break documents ('size', 'semantic', 'markdown', 'character')
            **kwargs: Extra settings for the specific strategy
            
        Returns:
            List of broken document pieces
            
        Example:
            >>> chunker = DocumentChunker()
            >>> # Break by size
            >>> pieces = chunker.chunk_documents(documents, strategy="size", chunk_size=500)
            >>> # Break by meaning
            >>> pieces = chunker.chunk_documents(documents, strategy="semantic")
        """
        if strategy == "size":
            return self.chunk_by_size(documents, **kwargs)
        elif strategy == "semantic":
            return self.chunk_by_semantic(documents, **kwargs)
        elif strategy == "markdown":
            return self.chunk_markdown(documents, **kwargs)
        elif strategy == "character":
            return self.chunk_by_character(documents, **kwargs)
        else:
            raise ValueError(f"Unknown breaking strategy: {strategy}")

if __name__ == "__main__":
    """
    This part runs when you run this file directly.
    It shows examples of how to use the DocumentChunker class.
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="This is a sample document. It contains multiple sentences. "
                        "We will test how it gets chunked. The chunking should preserve "
                        "the semantic meaning of the text.",
            metadata={"source": "test"}
        ),
        Document(
            page_content="Another sample document. This one is shorter but still needs "
                        "to be chunked properly. The chunking strategy should handle "
                        "documents of different lengths.",
            metadata={"source": "test"}
        )
    ]
    
    # Initialize chunker
    chunker = DocumentChunker()
    
    # Test size-based chunking
    print("Testing size-based chunking...")
    size_chunks = chunker.chunk_by_size(
        documents=sample_docs,
        chunk_size=50,
        chunk_overlap=10
    )
    print(f"Created {len(size_chunks)} chunks using size-based strategy")
    for i, chunk in enumerate(size_chunks[:2]):  # Print first 2 chunks
        print(f"\nChunk {i+1}:")
        print(chunk.page_content)
    
    # Test sentence-based chunking
    try:
        print("\nTesting sentence-based chunking...")
        sentence_chunks = chunker.chunk_by_sentence(
            documents=sample_docs,
            engine="nltk",
            chunk_size=50,
            chunk_overlap=10
        )
        print(f"Created {len(sentence_chunks)} chunks using sentence strategy")
        for i, chunk in enumerate(sentence_chunks[:2]):  # Print first 2 chunks
            print(f"\nChunk {i+1}:")
            print(chunk.page_content)
    except Exception as e:
        print(f"Sentence chunking test skipped: {e}")
    
    # Test markdown chunking
    print("\nTesting markdown chunking...")
    markdown_docs = [
        Document(
            page_content="# Header 1\nThis is under header 1\n## Header 2\nThis is under header 2",
            metadata={"source": "markdown_test"}
        )
    ]
    markdown_chunks = chunker.chunk_markdown(markdown_docs)
    print(f"Created {len(markdown_chunks)} chunks using markdown strategy")
    for i, chunk in enumerate(markdown_chunks):
        print(f"\nChunk {i+1}:")
        print(chunk.page_content)
    
    # Test generic chunking
    print("\nTesting generic chunking...")
    generic_chunks = chunker.chunk_documents(
        documents=sample_docs,
        strategy="size",
        chunk_size=50,
        chunk_overlap=10
    )
    print(f"Created {len(generic_chunks)} chunks using generic strategy")
