from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import os
import logging
import boto3
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import pytesseract
from dotenv import load_dotenv
import time

# Set logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Load environment variables
load_dotenv('/opt/airflow/Assignment_4/.env')

# Pinecone and S3 Configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_INPUT_PREFIX = os.getenv('S3_INPUT_PREFIX', "pdf_files/")
S3_OUTPUT_PREFIX = os.getenv('S3_OUTPUT_PREFIX', "processed_files/")

# Initialize Pinecone and S3
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
s3_client = boto3.client("s3")

# Temporary Directories
PDF_INPUT_DIR = Path("/tmp/pdf_files")
TXT_OUTPUT_DIR = Path("/tmp/txt_files")
PDF_INPUT_DIR.mkdir(parents=True, exist_ok=True)
TXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer("all-mpnet-base-v2")

# Helper Functions
def download_from_s3(bucket: str, prefix: str, local_dir: Path):
    """Download files from S3 to a local directory."""
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    )
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pdf"):
                local_path = local_dir / Path(obj["Key"]).name
                try:
                    s3_client.download_file(bucket, obj["Key"], str(local_path))
                    _log.info(f"Downloaded: {local_path}")
                except Exception as e:
                    _log.error(f"Error downloading {obj['Key']} from S3: {e}")
                    raise

def process_pdf(input_pdf_path: Path) -> str:
    """Process a PDF file and extract content to text."""
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        conv_res = doc_converter.convert(input_pdf_path)
        text_content = []

        for page in conv_res.document.pages.values():
            text_content.append(getattr(page, "text", "") or "")
            if hasattr(page, "image"):
                text_content.append(pytesseract.image_to_string(page.image.pil_image))

        txt_output_path = TXT_OUTPUT_DIR / f"{input_pdf_path.stem}_output.txt"
        with open(txt_output_path, "w", encoding="utf-8") as file:
            file.write("\n".join(text_content))

        _log.info(f"Processed PDF to text: {txt_output_path}")
        return str(txt_output_path)

    except Exception as e:
        _log.error(f"Error processing PDF {input_pdf_path}: {e}")
        raise

def upload_to_s3(file_path: Path, bucket: str, prefix: str):
    """Upload a local file to an S3 bucket."""
    try:
        key = os.path.join(prefix, file_path.name)
        s3_client.upload_file(str(file_path), bucket, key)
        _log.info(f"Uploaded {file_path} to s3://{bucket}/{key}")
    except Exception as e:
        _log.error(f"Failed to upload {file_path} to S3: {e}")
        raise

def cleanup_temp_files(directory: Path):
    """Clean up temporary files in a directory."""
    for file in directory.glob("*"):
        try:
            file.unlink()
            _log.info(f"Deleted temporary file: {file}")
        except Exception as e:
            _log.warning(f"Failed to delete {file}: {e}")

def process_pdfs(**kwargs):
    """Process PDFs in batches, upload results to S3, and delete temporary files."""
    pdf_files = list(PDF_INPUT_DIR.glob("*.pdf"))
    _log.info(f"Found {len(pdf_files)} PDF files to process")

    processed_files = []

    for pdf_path in pdf_files:
        try:
            # Simulated PDF processing (extracting text and saving to a file)
            txt_output_path = process_pdf(pdf_path)
            
            # Upload processed text file to S3
            upload_to_s3(Path(txt_output_path), S3_BUCKET_NAME, S3_OUTPUT_PREFIX)
            
            # Push file path to the list
            processed_files.append(str(txt_output_path))
            _log.info(f"Processed and uploaded: {pdf_path}, Output: {txt_output_path}")

        except Exception as e:
            _log.error(f"Error processing {pdf_path}: {e}")
            raise

    # Push processed file paths to XCom
    if processed_files:
        kwargs['task_instance'].xcom_push(key='txt_file_paths', value=processed_files)
        _log.info(f"Pushed processed file paths to XCom: {processed_files}")
    else:
        _log.warning("No files were processed.")

    # Cleanup temporary files
    cleanup_temp_files(PDF_INPUT_DIR)
    _log.info("Cleaned up temporary files in PDF_INPUT_DIR.")
    _log.info("Completed processing PDFs.")

def generate_and_store_embeddings(**kwargs):
    """Generate embeddings for text files, store them in Pinecone, and clean up temporary files."""
    try:
        # Retrieve the text file paths from XCom
        txt_file_paths = kwargs['task_instance'].xcom_pull(task_ids='process_pdfs', key='txt_file_paths')
        if not txt_file_paths:
            raise ValueError("No txt_file_paths found in XCom.")

        _log.info(f"Retrieved txt_file_paths from XCom: {txt_file_paths}")

        def sanitize_index_name(name: str) -> str:
            """Sanitize index names to meet Pinecone requirements."""
            import re
            sanitized_name = re.sub(r'[^a-z0-9-]', '-', name.lower())
            sanitized_name = re.sub(r'-+', '-', sanitized_name).strip('-')
            return sanitized_name

        for txt_file_path in txt_file_paths:
            txt_file_path = Path(txt_file_path)
            _log.info(f"Generating embeddings for: {txt_file_path}")

            # Sanitize the index name
            index_name = sanitize_index_name(f"pdf-{txt_file_path.stem}")
            EMBEDDING_DIM = 768

            # Create Pinecone index if it doesn't exist
            if index_name not in pc.list_indexes():
                pc.create_index(
                    name=index_name,
                    dimension=EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
                )
                _log.info(f"Created index: {index_name}")

            # Generate embeddings
            index = pc.Index(index_name)
            with open(txt_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            embeddings = model.encode(lines, show_progress_bar=True)
            for i, (line, embedding) in enumerate(zip(lines, embeddings)):
                vector_id = f"{txt_file_path.stem.lower()}-{i}"
                metadata = {
                    "line_number": i,
                    "text": line.strip(),
                    "file_name": txt_file_path.name,
                    "file_path": str(txt_file_path),
                    "timestamp": datetime.now().isoformat(),
                }
                index.upsert([(vector_id, embedding.tolist(), metadata)])

            _log.info(f"Embeddings stored for: {txt_file_path}")

        # Cleanup temporary files
        cleanup_temp_files(TXT_OUTPUT_DIR)
        _log.info("Cleaned up temporary files in TXT_OUTPUT_DIR.")

    except Exception as e:
        _log.error(f"Error generating embeddings: {e}")
        raise


# DAG Definition
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
    'email_on_failure': False,
}

with DAG(
    dag_id="process_pdf_and_generate_embeddings",
    default_args=default_args,
    description="Process PDFs and store embeddings with delays and retries",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    download_pdfs_task = PythonOperator(
        task_id="download_pdfs",
        python_callable=lambda: download_from_s3(S3_BUCKET_NAME, S3_INPUT_PREFIX, PDF_INPUT_DIR),
        execution_timeout=timedelta(minutes=10),
    )

    process_pdfs_task = PythonOperator(
        task_id="process_pdfs",
        python_callable=process_pdfs,
        provide_context=True,
        execution_timeout=timedelta(hours=1),
    )

    generate_embeddings_task = PythonOperator(
        task_id="generate_embeddings",
        python_callable=generate_and_store_embeddings,
        provide_context=True,
        execution_timeout=timedelta(hours=1),
    )

    download_pdfs_task >> process_pdfs_task >> generate_embeddings_task
