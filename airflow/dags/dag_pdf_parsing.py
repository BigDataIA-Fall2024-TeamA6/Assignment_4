from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
import datetime
import time
import gc
import pandas as pd 

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_INPUT_PREFIX = os.getenv("S3_INPUT_PREFIX", "pdf_files/")
S3_OUTPUT_PREFIX = os.getenv("S3_OUTPUT_PREFIX", "processed_files/")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

BATCH_SIZE = int(Variable.get("pdf_batch_size", default_var=5))
CHUNK_SIZE = int(Variable.get("pdf_chunk_size", default_var=1000))
IMAGE_RESOLUTION_SCALE = float(Variable.get("image_resolution_scale", default_var=1.5))
MAX_RETRIES = 3
RETRY_DELAY = 2

BASE_TMP_DIR = Path("/tmp")
PDF_INPUT_DIR = BASE_TMP_DIR / "pdf_files"
PROCESSED_OUTPUT_DIR = BASE_TMP_DIR / "processed_files"
PARQUET_OUTPUT_DIR = BASE_TMP_DIR / "output_parquet"

for dir_path in [PDF_INPUT_DIR, PROCESSED_OUTPUT_DIR, PARQUET_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def cleanup_old_files(directory: Path, max_age_hours: int = 1) -> None:
    """Remove files older than specified hours."""
    import time  # Lazy import
    current_time = time.time()
    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_hours * 3600:
                try:
                    file_path.unlink()
                    _log.info(f"Removed old file: {file_path}")
                except Exception as e:
                    _log.warning(f"Failed to remove {file_path}: {e}")

def retry_with_backoff(func, max_retries: int = MAX_RETRIES, base_delay: int = RETRY_DELAY):
    """Execute a function with exponential backoff retry."""
    import time  # Lazy import
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            _log.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

def check_memory_usage() -> bool:
    """Check if system has enough memory available."""
    import psutil  # Lazy import
    memory = psutil.virtual_memory()
    return memory.percent < 80

def extract_text_from_page(page) -> str:
    """
    Extract text content from a PDF page with OCR fallback.
    
    Args:
        page: A page object from the document converter
        
    Returns:
        str: Extracted text content from the page
    """
    import pytesseract  # Lazy import
    import gc  # Lazy import
    try:
        text_content = page.text.strip() if hasattr(page, 'text') else ""
        if not text_content and hasattr(page, 'image'):
            custom_config = r'--oem 3 --psm 6'
            text_content = pytesseract.image_to_string(
                page.image,
                config=custom_config,
                lang='eng'
            ).strip()
        text_content = ' '.join(text_content.split())
        return text_content if text_content else ""
    except Exception as e:
        _log.warning(f"Error extracting text from page {page.page_no}: {e}")
        return ""
    finally:
        if hasattr(page, 'image'):
            del page.image
        gc.collect()

def process_pdf_chunk(chunk_data: dict[str, any]) -> pd.DataFrame:
    """Process a single chunk of PDF data."""
    import gc
    try:
        rows = []
        for page in chunk_data['pages']:
            row = {
                "document": chunk_data['document_name'],
                "hash": chunk_data['document_hash'],
                "page_hash": create_hash(f"{chunk_data['document_hash']}:{page.page_no}"),
                "page_number": page.page_no,
                "content": extract_text_from_page(page),
            }
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        gc.collect()

def process_pdf(input_doc_path: Path) -> Path:
    """Process a PDF file with memory management."""
    from docling.datamodel.base_models import InputFormat  
    from docling.datamodel.pipeline_options import PdfPipelineOptions  
    from docling.document_converter import DocumentConverter, PdfFormatOption
    import time  # Lazy import
    import gc  # Lazy import

    pipeline_options = PdfPipelineOptions(images_scale=IMAGE_RESOLUTION_SCALE)
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    try:
        start_time = time.time()
        conv_res = doc_converter.convert(input_doc_path)
        chunks = []
        for chunk_start in range(0, len(conv_res.document.pages), CHUNK_SIZE):
            if not check_memory_usage():
                _log.warning("Low memory detected, triggering garbage collection")
                gc.collect()
            chunk_end = chunk_start + CHUNK_SIZE
            chunk_data = {
                'document_name': conv_res.input.file.name,
                'document_hash': conv_res.input.document_hash,
                'pages': conv_res.document.pages[chunk_start:chunk_end]
            }
            chunk_df = process_pdf_chunk(chunk_data)
            chunks.append(chunk_df)
            del chunk_data
            gc.collect()
        final_df = pd.concat(chunks, ignore_index=True)
        output_file = PARQUET_OUTPUT_DIR / f"{input_doc_path.stem}_{int(time.time())}.parquet"
        final_df.to_parquet(output_file, index=False)
        return output_file
    except Exception as e:
        _log.error(f"Error processing {input_doc_path}: {e}")
        raise
    finally:
        gc.collect()

def generate_and_store_embeddings(txt_file_path: Path) -> None:
    """Generate and store embeddings."""
    from sentence_transformers import SentenceTransformer  # Lazy import
    from pinecone import Pinecone, ServerlessSpec  # Lazy import
    import hashlib  # Lazy import
    import gc  # Lazy import

    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index_name = hashlib.md5(txt_file_path.stem.encode()).hexdigest()[:12]
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
    index = pc.Index(index_name)
    try:
        with open(txt_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        batch_size = 100
        for batch_start in range(0, len(lines), batch_size):
            if not check_memory_usage():
                _log.warning("Low memory detected, waiting for cleanup")
                gc.collect()
                time.sleep(5)
            batch_end = batch_start + batch_size
            batch = lines[batch_start:batch_end]
            embeddings = embedding_model.encode(batch, show_progress_bar=False)
            vectors = [
                (f"{txt_file_path.stem}-{i}", 
                 embedding.tolist(),
                 {"text": text.strip(), "line_number": i})
                for i, (embedding, text) in enumerate(zip(embeddings, batch))
            ]
            retry_with_backoff(lambda: index.upsert(vectors=vectors))
            del embeddings, vectors
            gc.collect()
            time.sleep(0.5)
    except Exception as e:
        _log.error(f"Error generating embeddings for {txt_file_path}: {e}")
        raise
    finally:
        gc.collect()

# DAG definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,  # Retry twice
    'retry_delay': datetime.timedelta(seconds=30),  
    'retry_exponential_backoff': True,  
    'max_retry_delay': datetime.timedelta(minutes=5), 
}

with DAG(
    dag_id="optimized_pdf_processing_pipeline",
    default_args=default_args,
    description="Optimized PDF processing and vector storage pipeline",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    concurrency=3,
    tags=['pdf', 'processing', 'vectors'],
) as dag:
    
    def download_from_s3():
        """Download PDF files from S3 with memory management."""
        s3 = boto3.client("s3")
        cleanup_old_files(PDF_INPUT_DIR)
        
        try:
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_INPUT_PREFIX):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.pdf'):
                        local_path = PDF_INPUT_DIR / Path(obj['Key']).name
                        retry_with_backoff(
                            lambda: s3.download_file(S3_BUCKET_NAME, obj['Key'], str(local_path))
                        )
        except Exception as e:
            _log.error(f"Error downloading files: {e}")
            raise
        finally:
            gc.collect()

    def process_pdfs():
        """Process PDFs in batches with memory management."""
        pdf_files = list(PDF_INPUT_DIR.glob("*.pdf"))
        _log.info(f"Found {len(pdf_files)} PDF files to process")
        
        for i in range(0, len(pdf_files), BATCH_SIZE):
            batch = pdf_files[i:i + BATCH_SIZE]
            for pdf_path in batch:
                if check_memory_usage():
                    try:
                        process_pdf(pdf_path)
                    except Exception as e:
                        _log.error(f"Error processing {pdf_path}: {e}")
                else:
                    _log.warning("Low memory, waiting before processing next file")
                    time.sleep(60)  # Wait for memory to free up
            
            gc.collect()

    def store_vectors():
        """Store vectors with memory management."""
        txt_files = list(PROCESSED_OUTPUT_DIR.glob("*.txt"))
        _log.info(f"Found {len(txt_files)} text files to process")
        
        for txt_file in txt_files:
            if check_memory_usage():
                try:
                    generate_and_store_embeddings(txt_file)
                except Exception as e:
                    _log.error(f"Error processing vectors for {txt_file}: {e}")
            else:
                _log.warning("Low memory, waiting before processing next file")
                time.sleep(60)
            
            gc.collect()

    # Task definitions
    task_download = PythonOperator(
        task_id="download_pdfs",
        python_callable=download_from_s3,
        execution_timeout=datetime.timedelta(minutes=30)
    )

    task_process = PythonOperator(
        task_id="process_pdfs",
        python_callable=process_pdfs,
        execution_timeout=datetime.timedelta(hours=2)
    )

    task_vectorize = PythonOperator(
        task_id="store_vectors",
        python_callable=store_vectors,
        execution_timeout=datetime.timedelta(hours=2)
    )

    # Define task dependencies
    task_download >> task_process >> task_vectorize
