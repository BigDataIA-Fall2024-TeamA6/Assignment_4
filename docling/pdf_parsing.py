import datetime
import logging
import time
from pathlib import Path
import pandas as pd
import boto3
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.utils import create_hash
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
import pytesseract  # Ensure Tesseract OCR is installed and pytesseract is available
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Configuration constants
IMAGE_RESOLUTION_SCALE = 2.0
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_INPUT_PREFIX = os.getenv("S3_INPUT_PREFIX", "pdf_files/")
S3_OUTPUT_PREFIX = os.getenv("S3_OUTPUT_PREFIX", "processed_files/")
IMAGE_OUTPUT_DIR = Path("/tmp/output_images")
PARQUET_OUTPUT_DIR = Path("/tmp/output_parquet")
PARQUET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

s3_client = boto3.client("s3")

def download_from_s3(bucket: str, prefix: str, local_dir: Path):
    """Download all PDF files from an S3 prefix to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pdf"):
                local_path = local_dir / Path(obj["Key"]).name
                s3_client.download_file(bucket, obj["Key"], str(local_path))
                _log.info(f"Downloaded: {local_path}")

def upload_to_s3(file_path: Path, bucket: str, prefix: str):
    """Upload a local file to an S3 bucket."""
    key = os.path.join(prefix, file_path.name)
    s3_client.upload_file(str(file_path), bucket, key)
    _log.info(f"Uploaded {file_path} to s3://{bucket}/{key}")

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image)

def generate_multimodal_pages_custom(conv_res):
    """Custom function to iterate through each page in the document and extract content."""
    for page_no, page in conv_res.document.pages.items():
        page_text = getattr(page, "text", None) or getattr(page, "content", None) or ""
        page_cells = getattr(page, "cells", [])
        page_segments = getattr(page, "segments", [])
        page_image = getattr(page, "image", None).pil_image if getattr(page, "image", None) else None
        yield page_text, page_cells, page_segments, page, page_image

def process_pdf(input_doc_path: Path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    start_time = time.time()
    conv_res = doc_converter.convert(input_doc_path)

    rows = []
    text_content = []

    for content_text, page_cells, page_segments, page, page_image in generate_multimodal_pages_custom(conv_res):
        dpi = 72  # Standard DPI for PDFs
        page_image_text = extract_text_from_image(page_image) if page_image else ""
        text_content.append(content_text)
        text_content.append(page_image_text)

        rows.append({
            "document": conv_res.input.file.name,
            "hash": conv_res.input.document_hash,
            "page_hash": create_hash(
                conv_res.input.document_hash + ":" + str(page.page_no - 1)
            ),
            "image": {
                "width": page_image.width if page_image else None,
                "height": page_image.height if page_image else None,
            } if page_image else None,
            "cells": page_cells,
            "contents": content_text,
            "segments": page_segments,
            "extracted_image_text": page_image_text,
            "extra": {
                "page_num": page.page_no + 1,
                "width_in_points": page.size.width,
                "height_in_points": page.size.height,
                "dpi": dpi,
            },
        })

    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = IMAGE_OUTPUT_DIR / f"{input_doc_path.stem}-table-{table_counter}.png"
            with element_image_filename.open("wb") as fp:
                element.image.pil_image.save(fp, "PNG")
            table_image_text = extract_text_from_image(element.image.pil_image)
            text_content.append(table_image_text)
            rows.append({
                "document": conv_res.input.file.name,
                "element_type": "table",
                "element_index": table_counter,
                "extracted_image_text": table_image_text
            })

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = IMAGE_OUTPUT_DIR / f"{input_doc_path.stem}-picture-{picture_counter}.png"
            with element_image_filename.open("wb") as fp:
                element.image.pil_image.save(fp, "PNG")
            picture_image_text = extract_text_from_image(element.image.pil_image)
            text_content.append(picture_image_text)
            rows.append({
                "document": conv_res.input.file.name,
                "element_type": "picture",
                "element_index": picture_counter,
                "extracted_image_text": picture_image_text
            })

    df = pd.json_normalize(rows)
    parquet_output_filename = PARQUET_OUTPUT_DIR / f"{input_doc_path.stem}.parquet"
    df.to_parquet(parquet_output_filename, index=False)

    txt_output_filename = f"{input_doc_path.stem}_output.txt"
    with open(txt_output_filename, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(text_content))

    upload_to_s3(parquet_output_filename, S3_BUCKET_NAME, S3_OUTPUT_PREFIX)
    upload_to_s3(Path(txt_output_filename), S3_BUCKET_NAME, S3_OUTPUT_PREFIX)

    end_time = time.time() - start_time
    _log.info(f"Document processed in {end_time:.2f} seconds.")
    return parquet_output_filename

if __name__ == "__main__":
    local_pdf_dir = Path("/tmp/pdf_files")
    download_from_s3(S3_BUCKET_NAME, S3_INPUT_PREFIX, local_pdf_dir)

    for pdf_path in local_pdf_dir.glob("*.pdf"):
        parquet_file = process_pdf(pdf_path)
        _log.info(f"Output file generated and uploaded: {parquet_file}")
