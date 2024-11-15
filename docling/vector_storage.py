import os
import time
import hashlib
import logging
from dotenv import load_dotenv
from pathlib import Path
import boto3
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Set logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')

# S3 configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_INPUT_PREFIX = os.getenv('S3_INPUT_PREFIX', "processed_files/")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize sentence transformer model for embedding generation
model = SentenceTransformer("all-mpnet-base-v2")

# Initialize S3 client
s3_client = boto3.client("s3")

def download_txt_from_s3(bucket: str, prefix: str, local_dir: Path):
    """Download all text files from an S3 prefix to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".txt"):
                local_path = local_dir / Path(obj["Key"]).name
                s3_client.download_file(bucket, obj["Key"], str(local_path))
                _log.info(f"Downloaded: {local_path}")

def generate_index_name(file_name):
    """Generate a unique, short, lowercase index name using a hash of the file name."""
    hash_object = hashlib.sha1(file_name.encode())  # Use SHA-1 for a 40-character hash
    short_hash = hash_object.hexdigest()[:10]  # Use the first 10 characters of the hash
    return f"pdf-{short_hash}".lower()  # Ensure lowercase and use hyphen

def upsert_with_retry(index, vector_id, embedding, retries=3, delay=2):
    """Upserts a vector with retries on failure."""
    for attempt in range(retries):
        try:
            index.upsert([(vector_id, embedding.tolist())])
            return
        except Exception as e:
            _log.warning(f"Upsert attempt {attempt + 1}/{retries} failed for vector_id: {vector_id}. Error: {e}")
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
    _log.error(f"Failed to upsert after {retries} attempts for vector_id: {vector_id}")

def generate_and_store_embeddings(txt_file_path):
    """
    Reads text from a txt file, generates embeddings for each line,
    and stores them in a unique Pinecone index for the PDF.
    """
    index_name = generate_index_name(txt_file_path.stem)
    EMBEDDING_DIM = 768  # Set to match the output of the model

    # Check if index exists; create if it doesn’t
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        _log.info(f"Created a new index: {index_name}")
    else:
        _log.info(f"Using existing index: {index_name}")

    # Verify that the index exists after creation
    available_indexes = pc.list_indexes().names()
    if index_name not in available_indexes:
        _log.error(f"Index {index_name} does not exist after creation. Exiting function.")
        return

    # Connect to the index
    try:
        index = pc.Index(index_name)
        _log.info(f"Connected to index: {index_name}")
    except Exception as e:
        _log.error(f"Failed to connect to index: {index_name}. Error: {e}")
        return

    # Read text content from the file
    with open(txt_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Generate embeddings for each line
    embeddings = model.encode(lines, show_progress_bar=True)

    # Store each embedding in the respective Pinecone index with unique IDs
    for i, (line, embedding) in enumerate(zip(lines, embeddings)):
        vector_id = f"{txt_file_path.stem.lower().replace('_', '-')}-{i}"
        upsert_with_retry(index, vector_id, embedding)
    
    _log.info(f"Embeddings for {txt_file_path.stem} stored in Pinecone index '{index_name}'.")

if __name__ == "__main__":
    local_txt_dir = Path("/tmp/txt_files")
    download_txt_from_s3(S3_BUCKET_NAME, S3_INPUT_PREFIX, local_txt_dir)

    for txt_file_path in local_txt_dir.glob("*.txt"):
        generate_and_store_embeddings(txt_file_path)

    _log.info("Embedding storage complete.")
