#!/usr/bin/env python3
"""
Vector Database Creation Script for Legal RAG System

This script creates a vector database from legal data stored in an Excel file.
It extracts answers and questions, creates embeddings, and stores them in Qdrant.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Load environment variables
load_dotenv()

# Get configuration from environment variables
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "VIMQA_dev")

# Initialize HuggingFace client for embeddings
client = InferenceClient(provider="hf-inference", api_key=HUGGINGFACE_API_KEY)

def get_embedding(text):
    """Get embedding for a single text using HuggingFace Inference API"""
    return client.feature_extraction(model=EMBEDDINGS_MODEL_NAME, text=text)

def upsert_to_qdrant(vectors, payloads, collection_name, ids=None):
    """
    Upsert vectors vào Qdrant với cấu trúc metadata giống LangChain
    """
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Kiểm tra và tạo collection nếu chưa có
    try:
        qdrant.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' đã tồn tại.")
    except Exception:
        print(f"Collection '{collection_name}' chưa tồn tại. Đang tạo mới...")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE
            )
        )
        print(f"Đã tạo collection '{collection_name}'.")

    if ids is None:
        ids = list(range(1, len(vectors)+1))
    
    # Log cấu trúc payload mẫu để kiểm tra
    if payloads:
        print("\nCấu trúc payload mẫu:")
        print(f"Metadata: {payloads[0].get('metadata', {})}")
        print(f"Page content: {payloads[0].get('page_content', '')[:100]}...")
    
    qdrant.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )
    )
    print(f"Đã upsert {len(vectors)} vectors vào collection '{collection_name}'")

def create_vector_database(excel_file_path, collection_name=None, start_idx=None, end_idx=None):
    """
    Create a vector database from an Excel file containing text data with metadata.
    
    Args:
        excel_file_path (str): Path to the Excel file
        collection_name (str, optional): Name for the Qdrant collection
        start_idx (int, optional): Start index of data to process (0-based)
        end_idx (int, optional): End index of data to process (exclusive)
    
    Returns:
        Qdrant: The created vector database
    """
    if collection_name is None:
        collection_name = COLLECTION_NAME
    
    print(f"Loading data from {excel_file_path}...")
    # Load Excel data
    data = pd.read_excel(excel_file_path)
    
    # Extract text and metadata columns
    texts = data["text"].dropna().tolist()
    doc_ids = data["doc_id"].dropna().tolist()
    titles = data["title"].dropna().tolist()
    
    # Tạo IDs dựa trên index của DataFrame (số thứ tự dòng trong Excel)
    # Thêm 1 vì Excel bắt đầu từ 1, trong khi pandas index bắt đầu từ 0
    ids = [idx + 1 for idx in data.index]
    
    # Ensure all columns have the same length
    min_length = min(len(texts), len(doc_ids), len(titles))
    if min_length != len(texts) or min_length != len(doc_ids) or min_length != len(titles):
        print(f"Warning: Mismatch between number of texts ({len(texts)}), doc_ids ({len(doc_ids)}), and titles ({len(titles)})")
        texts = texts[:min_length]
        doc_ids = doc_ids[:min_length]
        titles = titles[:min_length]
        ids = ids[:min_length]
    
    # Apply start and end indices if provided
    if start_idx is not None:
        start_idx = max(0, start_idx)
        texts = texts[start_idx:end_idx] if end_idx is not None else texts[start_idx:]
        doc_ids = doc_ids[start_idx:end_idx] if end_idx is not None else doc_ids[start_idx:]
        titles = titles[start_idx:end_idx] if end_idx is not None else titles[start_idx:]
        ids = ids[start_idx:end_idx] if end_idx is not None else ids[start_idx:]
    
    if end_idx is not None and start_idx is None:
        end_idx = min(len(texts), end_idx)
        texts = texts[:end_idx]
        doc_ids = doc_ids[:end_idx]
        titles = titles[:end_idx]
        ids = ids[:end_idx]
    
    print(f"Processing {len(texts)} documents (from index {start_idx or 0} to {end_idx or len(texts)})...")
    
    # Create payloads with metadata
    payloads = [
        {
            "metadata": {
                "source": excel_file_path,
                "doc_id": doc_id,
                "title": title,
                "excel_row": id  # Thêm số thứ tự dòng Excel vào metadata
            },
            "page_content": text
        }
        for text, doc_id, title, id in zip(texts, doc_ids, titles, ids)
    ]
    
    print(f"Getting embeddings for {len(texts)} documents...")
    # Get embeddings directly using the get_embedding function
    vectors = []
    for i, text in enumerate(texts):
        print(f"Embedding document {i+1}/{len(texts)} (Excel row {ids[i]})...")
        vector = get_embedding(text)
        vectors.append(vector)
        if (i + 1) % 10 == 0:  # Log progress every 10 documents
            print(f"Progress: {i+1}/{len(texts)} documents embedded ({(i+1)/len(texts)*100:.1f}%)")
    
    print(f"Upserting to Qdrant collection: {collection_name}")
    upsert_to_qdrant(vectors, payloads, collection_name, ids=ids)
    print(f"Successfully added {len(texts)} documents to Qdrant")

def create_vector_database_from_text_files(text_dir, collection_name=None):
    """
    Create a vector database from text files in a directory.
    
    Args:
        text_dir (str): Directory containing text files
        collection_name (str, optional): Name for the Qdrant collection
    
    Returns:
        Qdrant: The created vector database
    """
    if collection_name is None:
        collection_name = COLLECTION_NAME + "_text"
    
    print(f"Loading text files from {text_dir}...")
    documents = []
    
    # Walk through the directory and process each text file
    for root, _, files in os.walk(text_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split content into paragraphs
                    paragraphs = [p for p in content.split('\n\n') if p.strip()]
                    
                    # Create payloads with metadata
                    for paragraph in paragraphs:
                        if len(paragraph.strip()) > 50:
                            documents.append({
                                "metadata": {
                                    "source": file_path,
                                    "_id": os.path.basename(file_path)
                                },
                                "page_content": paragraph
                            })
                    
                    print(f"Processed {file_path}: {len(paragraphs)} paragraphs")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    if not documents:
        print("No documents found to process!")
        return
    
    print(f"Getting embeddings for {len(documents)} documents...")
    vectors = [get_embedding(doc["page_content"]) for doc in documents]
    
    print(f"Upserting to Qdrant collection: {collection_name}")
    upsert_to_qdrant(vectors, documents, collection_name)
    print(f"Successfully added {len(documents)} documents to Qdrant collection: {collection_name}")

def main():
    """Main function to run the script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a vector database from legal data')
    parser.add_argument('--excel', type=str, help='Path to Excel file with Q&A data')
    parser.add_argument('--text-dir', type=str, help='Directory containing text files to process')
    parser.add_argument('--collection', type=str, help='Name for the Qdrant collection')
    parser.add_argument('--start', type=int, help='Start index of data to process (0-based)')
    parser.add_argument('--end', type=int, help='End index of data to process (exclusive)')
    
    args = parser.parse_args()
    
    if args.excel:
        create_vector_database(args.excel, args.collection, args.start, args.end)
    elif args.text_dir:
        create_vector_database_from_text_files(args.text_dir, args.collection)
    else:
        print("Please provide either --excel or --text-dir argument")
        parser.print_help()

if __name__ == "__main__":
    main() 