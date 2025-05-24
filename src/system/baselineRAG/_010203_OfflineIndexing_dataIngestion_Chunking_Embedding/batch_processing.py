#!/usr/bin/env python3
"""
Batch Processing Script for Vector Database Creation

This script processes data in batches with retry logic and logging.
It sends 100 documents at a time, waits for 1 minute, then continues.
If an error occurs, it will retry up to 3 times with longer wait times.
"""

import os
import time
import logging
import argparse
import pandas as pd
from datetime import datetime
from create_vector_database import create_vector_database

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def process_batch(excel_file, collection_name, start_idx, end_idx=None, batch_size=100, wait_time=60):
    """
    Process data in batches with retry logic
    
    Args:
        excel_file (str): Path to Excel file
        collection_name (str): Name of Qdrant collection
        start_idx (int): Start index of data
        end_idx (int, optional): End index of data. If None, process until end of file
        batch_size (int): Number of documents to process in each batch
        wait_time (int): Time to wait between batches in seconds
    """
    # Read Excel file to get total number of rows
    data = pd.read_excel(excel_file)
    total_rows = len(data)
    
    # If end_idx is not provided, use total number of rows
    if end_idx is None:
        end_idx = total_rows
    
    total_docs = end_idx - start_idx
    num_batches = (total_docs + batch_size - 1) // batch_size
    
    logging.info(f"Starting batch processing:")
    logging.info(f"- Total documents to process: {total_docs}")
    logging.info(f"- Batch size: {batch_size}")
    logging.info(f"- Number of batches: {num_batches}")
    logging.info(f"- Wait time between batches: {wait_time} seconds")
    logging.info(f"- Processing documents from Excel rows {start_idx + 1} to {end_idx}")
    
    for batch_num in range(num_batches):
        batch_start = start_idx + (batch_num * batch_size)
        batch_end = min(batch_start + batch_size, end_idx)
        current_batch_size = batch_end - batch_start
        
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing batch {batch_num + 1}/{num_batches}")
        logging.info(f"Documents {batch_start} to {batch_end - 1} (Excel rows {batch_start + 1} to {batch_end})")
        logging.info(f"Current batch size: {current_batch_size} documents")
        logging.info(f"Overall progress: {batch_num * batch_size}/{total_docs} documents ({(batch_num * batch_size)/total_docs*100:.1f}%)")
        logging.info(f"{'='*50}\n")
        
        # Try up to 3 times with increasing wait times
        for attempt in range(3):
            try:
                create_vector_database(
                    excel_file_path=excel_file,
                    collection_name=collection_name,
                    start_idx=batch_start,
                    end_idx=batch_end
                )
                logging.info(f"Successfully processed batch {batch_num + 1}")
                break
            except Exception as e:
                retry_wait = (attempt + 1) * 300  # 5, 10, 15 minutes
                logging.error(f"Error processing batch {batch_num + 1} (attempt {attempt + 1}/3): {str(e)}")
                
                if attempt < 2:  # Don't wait after the last attempt
                    logging.info(f"Waiting {retry_wait} seconds before retry...")
                    time.sleep(retry_wait)
                else:
                    logging.error(f"Failed to process batch {batch_num + 1} after 3 attempts")
                    raise  # Re-raise the last exception
        
        # Wait between batches if not the last batch
        if batch_num < num_batches - 1:
            logging.info(f"Waiting {wait_time} seconds before next batch...")
            time.sleep(wait_time)
    
    logging.info("\nBatch processing completed!")

def main():
    parser = argparse.ArgumentParser(description='Process data in batches with retry logic')
    parser.add_argument('--excel', type=str, required=True, help='Path to Excel file')
    parser.add_argument('--collection', type=str, default='VIMQA_dev', help='Qdrant collection name')
    parser.add_argument('--start', type=int, default=0, help='Start index (0-based)')
    parser.add_argument('--end', type=int, help='End index (exclusive). If not provided, process until end of file')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of documents per batch')
    parser.add_argument('--wait-time', type=int, default=60, help='Wait time between batches in seconds')
    
    args = parser.parse_args()
    
    try:
        process_batch(
            excel_file=args.excel,
            collection_name=args.collection,
            start_idx=args.start,
            end_idx=args.end,
            batch_size=args.batch_size,
            wait_time=args.wait_time
        )
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 