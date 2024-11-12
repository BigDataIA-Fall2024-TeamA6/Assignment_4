import datetime
import logging
import time
from pathlib import Path
import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.utils import create_hash
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
import pytesseract  # Ensure Tesseract OCR is installed and pytesseract is available

# Set logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Configuration constants
IMAGE_RESOLUTION_SCALE = 2.0
IMAGE_OUTPUT_DIR = Path("output_images")
PARQUET_OUTPUT_DIR = Path("output_parquet")
PARQUET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR."""
    return pytesseract.image_to_string(image)

def generate_multimodal_pages_custom(conv_res):
    """Custom function to iterate through each page in the document and extract content."""
    for page_no, page in conv_res.document.pages.items():
        # Attempt to retrieve page text if available
        page_text = getattr(page, "text", None) or getattr(page, "content", None) or ""
        page_cells = getattr(page, "cells", [])
        page_segments = getattr(page, "segments", [])

        # Extract the image for the page if it exists
        page_image = getattr(page, "image", None).pil_image if getattr(page, "image", None) else None

        yield page_text, page_cells, page_segments, page, page_image

def process_pdf(input_doc_path: Path):
    # Setup pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.generate_picture_images = True

    # Initialize document converter
    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    # Start conversion and timing
    start_time = time.time()
    conv_res = doc_converter.convert(input_doc_path)

    rows = []
    text_content = []

    # Use the custom page generation function to process pages
    for content_text, page_cells, page_segments, page, page_image in generate_multimodal_pages_custom(conv_res):
        
        # Set default DPI or retrieve it from page properties if available
        dpi = 72  # Standard DPI for PDFs; adjust if actual DPI is needed
        
        # Extract text from page image if needed
        page_image_text = extract_text_from_image(page_image) if page_image else ""

        # Append extracted text to text_content for this PDF
        text_content.append(content_text)
        text_content.append(page_image_text)  # Add OCR text if any

        # Save each page's data as a dictionary
        rows.append(
            {
                "document": conv_res.input.file.name,
                "hash": conv_res.input.document_hash,
                "page_hash": create_hash(
                    conv_res.input.document_hash + ":" + str(page.page_no - 1)
                ),
                "image": {
                    "width": page_image.width if page_image else None,
                    "height": page_image.height if page_image else None,
                    "bytes": page_image.tobytes() if page_image else None,
                } if page_image else None,
                "cells": page_cells,
                "contents": content_text,
                "segments": page_segments,
                "extracted_image_text": page_image_text,  # OCR text from page image
                "extra": {
                    "page_num": page.page_no + 1,
                    "width_in_points": page.size.width,
                    "height_in_points": page.size.height,
                    "dpi": dpi,
                },
            }
        )

    # Extract images of figures and tables and apply OCR if they contain text
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                IMAGE_OUTPUT_DIR / f"{input_doc_path.stem}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.image.pil_image.save(fp, "PNG")
            
            # OCR on table image
            table_image_text = extract_text_from_image(element.image.pil_image)
            text_content.append(table_image_text)  # Add OCR text from table

            rows.append({
                "document": conv_res.input.file.name,
                "element_type": "table",
                "element_index": table_counter,
                "extracted_image_text": table_image_text
            })

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                IMAGE_OUTPUT_DIR / f"{input_doc_path.stem}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.image.pil_image.save(fp, "PNG")
            
            # OCR on picture image
            picture_image_text = extract_text_from_image(element.image.pil_image)
            text_content.append(picture_image_text)  # Add OCR text from picture

            rows.append({
                "document": conv_res.input.file.name,
                "element_type": "picture",
                "element_index": picture_counter,
                "extracted_image_text": picture_image_text
            })

    # Convert rows to DataFrame and save as parquet for embeddings
    df = pd.json_normalize(rows)
    now = datetime.datetime.now()
    parquet_output_filename = PARQUET_OUTPUT_DIR / f"{input_doc_path.stem}_{now:%Y-%m-%d_%H%M%S}.parquet"
    df.to_parquet(parquet_output_filename, index=False)

    # Save all extracted content to a unique text file for this PDF
    txt_output_filename = f"{input_doc_path.stem}_output.txt"
    with open(txt_output_filename, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(text_content))

    end_time = time.time() - start_time
    _log.info(f"Document processed in {end_time:.2f} seconds.")
    _log.info(f"Text file generated: {txt_output_filename}")
    return parquet_output_filename

if __name__ == "__main__":
    # Specify paths to the PDF files you want to process
    pdf_paths = [
        Path("/Users/sahitinallamolu/NEU/BDIA/Assignment_4/pdf_files/10-years-after-global-financial-crisis.pdf"),
        Path("/Users/sahitinallamolu/NEU/BDIA/Assignment_4/pdf_files/ai-and-big-data-in-investments.pdf")
    ]

    for pdf_path in pdf_paths:
        parquet_file = process_pdf(pdf_path)
        _log.info(f"Output file generated: {parquet_file}")
