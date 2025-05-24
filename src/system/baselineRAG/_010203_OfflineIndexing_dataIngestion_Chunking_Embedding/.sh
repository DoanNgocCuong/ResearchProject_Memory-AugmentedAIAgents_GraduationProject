python -m venv .venv
.venv/Scripts/activate
pip install -r src/system/baselineRAG/_010203_OfflineIndexing_dataIngestion_Chunking_Embedding/requirements_backend_vector_database.txt
cd src\system\baselineRAG\_010203_OfflineIndexing_dataIngestion_Chunking_Embedding

python src\system\baselineRAG\_010203_OfflineIndexing_dataIngestion_Chunking_Embedding\create_vector_database.py --excel src\datasets\dataset_full\vimqa_processed\corpus_vimqa_dev_300.xlsx

python src\system\baselineRAG\_010203_OfflineIndexing_dataIngestion_Chunking_Embedding\create_vector_database.py --excel src\datasets\dataset_full\vimqa_processed\corpus_vimqa_dev_300.xlsx --start 0 --end 10

python src\system\baselineRAG\_010203_OfflineIndexing_dataIngestion_Chunking_Embedding\batch_processing.py --excel src\datasets\dataset_full\vimqa_processed\corpus_vimqa_dev_300.xlsx --start 10 --end 200 --batch-size 10 --wait-time 60