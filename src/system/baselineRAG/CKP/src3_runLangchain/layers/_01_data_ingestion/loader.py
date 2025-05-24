"""
This module helps load documents from different places like websites, PDF files, and CSV files.
It makes all documents look the same so they can be used easily in other parts of the program.
"""

from typing import List, Union, Dict, Any
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    JSONLoader
)
from langchain_core.documents import Document
import bs4
import json

class DataLoader:
    """
    A class that helps load documents from many different places.
    
    This class can get documents from:
    - Websites
    - PDF files
    - CSV files
    - Text files
    - Markdown files
    
    It makes all documents look the same so they can be used easily.
    """
    
    def __init__(self):
        """
        Start the DataLoader with tools to load different types of documents.
        
        This sets up all the tools needed to load documents from different places.
        Each tool knows how to read a specific type of document.
        """
        self.supported_loaders = {
            'web': WebBaseLoader,
            'pdf': PyPDFLoader,
            'csv': CSVLoader,
            'text': TextLoader,
            'markdown': UnstructuredMarkdownLoader
        }

    def load_web_documents(self, urls: List[str], bs_kwargs: dict = None) -> List[Document]:
        """
        Get documents from websites.
        
        Args:
            urls: List of website addresses to get documents from
            bs_kwargs: Special settings for reading websites (optional)
            
        Returns:
            A list of documents from the websites
            
        Example:
            >>> loader = DataLoader()
            >>> docs = loader.load_web_documents(["https://example.com"])
            >>> print(len(docs))  # Shows how many documents were found
        """
        if bs_kwargs is None:
            bs_kwargs = dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            )
            
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=bs_kwargs
        )
        return loader.load()

    def load_pdf_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Get documents from PDF files.
        
        Args:
            file_paths: List of PDF file locations on your computer
            
        Returns:
            A list of documents from the PDF files
            
        Example:
            >>> loader = DataLoader()
            >>> docs = loader.load_pdf_documents(["document.pdf"])
            >>> print(docs[0].page_content)  # Shows the first page of the PDF
        """
        all_docs = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
        return all_docs

    def load_csv_documents(self, file_path: str, **kwargs) -> List[Document]:
        """
        Get documents from a CSV file.
        
        Args:
            file_path: Location of the CSV file on your computer
            **kwargs: Extra settings for reading the CSV file
            
        Returns:
            A list of documents from the CSV file
            
        Example:
            >>> loader = DataLoader()
            >>> docs = loader.load_csv_documents("data.csv")
            >>> print(docs[0].metadata)  # Shows information about the first row
        """
        loader = CSVLoader(file_path, **kwargs)
        return loader.load()

    def load_text_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Get documents from text files.
        
        Args:
            file_paths: List of text file locations on your computer
            
        Returns:
            A list of documents from the text files
            
        Example:
            >>> loader = DataLoader()
            >>> docs = loader.load_text_documents(["notes.txt"])
            >>> print(docs[0].page_content)  # Shows the text content
        """
        all_docs = []
        for path in file_paths:
            loader = TextLoader(path)
            all_docs.extend(loader.load())
        return all_docs

    def load_markdown_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Get documents from markdown files.
        
        Args:
            file_paths: List of markdown file locations on your computer
            
        Returns:
            A list of documents from the markdown files
            
        Example:
            >>> loader = DataLoader()
            >>> docs = loader.load_markdown_documents(["README.md"])
            >>> print(docs[0].page_content)  # Shows the markdown content
        """
        all_docs = []
        for path in file_paths:
            loader = UnstructuredMarkdownLoader(path)
            all_docs.extend(loader.load())
        return all_docs

    def load_documents(self, source_type: str, source_paths: Union[str, List[str]], **kwargs) -> List[Document]:
        """
        Get documents from any supported source.
        
        This is an easy way to load documents without knowing which specific method to use.
        
        Args:
            source_type: Type of source ('web', 'pdf', 'csv', 'text', 'markdown')
            source_paths: Location(s) of the source(s)
            **kwargs: Extra settings for the specific loader
            
        Returns:
            A list of documents from the source
            
        Example:
            >>> loader = DataLoader()
            >>> # Load from a website
            >>> docs = loader.load_documents("web", ["https://example.com"])
            >>> # Load from a PDF
            >>> docs = loader.load_documents("pdf", ["document.pdf"])
        """
        if source_type not in self.supported_loaders:
            raise ValueError(f"Unsupported source type: {source_type}")
            
        if source_type == 'web':
            return self.load_web_documents(source_paths, **kwargs)
        elif source_type == 'pdf':
            return self.load_pdf_documents(source_paths)
        elif source_type == 'csv':
            return self.load_csv_documents(source_paths, **kwargs)
        elif source_type == 'text':
            return self.load_text_documents(source_paths)
        elif source_type == 'markdown':
            return self.load_markdown_documents(source_paths)

def load_faq_data(file_path: str) -> List[Document]:
    """
    Load FAQ data from JSON file and convert to Langchain Documents
    
    Args:
        file_path: Path to the JSON file containing FAQ data
        
    Returns:
        List of Langchain Documents with content and metadata
    """
    try:
        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert to Documents
        documents = []
        for item in data:
            doc = Document(
                page_content=item['content'],
                metadata={
                    'id': item['id'],
                    'feature_tag': item['meta_data']['feature_tag'],
                    'methods': item['meta_data']['methods'],
                    'application_values': item['meta_data']['application_values']
                }
            )
            documents.append(doc)
            
        return documents
        
    except Exception as e:
        print(f"Error loading FAQ data: {str(e)}")
        return []

def preprocess_faq_data(documents: List[Document]) -> List[Document]:
    """
    Preprocess FAQ documents by cleaning and normalizing content
    
    Args:
        documents: List of Langchain Documents
        
    Returns:
        List of preprocessed Documents
    """
    processed_docs = []
    for doc in documents:
        # Clean content
        content = doc.page_content.strip()
        
        # Create new document with cleaned content
        processed_doc = Document(
            page_content=content,
            metadata=doc.metadata
        )
        processed_docs.append(processed_doc)
        
    return processed_docs

if __name__ == "__main__":
    """
    This part runs when you run this file directly.
    It shows examples of how to use the DataLoader class.
    """
    # Test DataLoader
    loader = DataLoader()
    
    # Test web document loading
    print("Testing web document loading...")
    web_docs = loader.load_web_documents(
        urls=["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    )
    print(f"Loaded {len(web_docs)} web documents")
    
    # Test PDF document loading (if you have a PDF file)
    try:
        print("\nTesting PDF document loading...")
        pdf_docs = loader.load_pdf_documents(["test.pdf"])
        print(f"Loaded {len(pdf_docs)} PDF documents")
    except Exception as e:
        print(f"PDF loading test skipped: {e}")
    
    # Test CSV document loading (if you have a CSV file)
    try:
        print("\nTesting CSV document loading...")
        csv_docs = loader.load_csv_documents("test.csv")
        print(f"Loaded {len(csv_docs)} CSV documents")
    except Exception as e:
        print(f"CSV loading test skipped: {e}")
    
    # Test generic document loading
    print("\nTesting generic document loading...")
    try:
        docs = loader.load_documents(
            source_type="web",
            source_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"]
        )
        print(f"Loaded {len(docs)} documents using generic loader")
    except Exception as e:
        print(f"Generic loading test failed: {e}")

    # Test loading and preprocessing
    test_docs = load_faq_data("../../data/TinhNangApp.json")
    print(f"Loaded {len(test_docs)} documents")
    
    processed_docs = preprocess_faq_data(test_docs)
    print(f"Processed {len(processed_docs)} documents")
    
    # Print sample document
    if processed_docs:
        print("\nSample document:")
        print(f"Content: {processed_docs[0].page_content[:100]}...")
        print(f"Metadata: {processed_docs[0].metadata}")
