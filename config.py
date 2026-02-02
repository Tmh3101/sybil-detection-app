import os

DATASET_ID = os.getenv('DATASET_ID', 'lens-protocol-mainnet')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
MAX_DAYS_RANGE = int(os.getenv('MAX_DAYS_RANGE', '7'))
