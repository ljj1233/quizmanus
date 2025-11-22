"""PDF ingestion helpers for user-provided documents.

This module wraps the MinerU pipeline to convert uploaded PDFs into Markdown
and then pushes the extracted content into Milvus using the existing vector
store utilities. All paths and model settings are parameterised to avoid
hard-coded environment dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from milvus_model.hybrid import BGEM3EmbeddingFunction

from src.RAG import vector_store_utils


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_pdf_with_mineru(pdf_path: Path, output_dir: Path, keep_images: bool = False) -> Path:
    """Parse a PDF into Markdown with MinerU.

    Args:
        pdf_path: Path to the uploaded PDF file.
        output_dir: Directory to store generated Markdown and images.
        keep_images: Whether to preserve images in the Markdown output.

    Returns:
        Path to the generated Markdown file.
    """

    try:  # Local import to avoid hard dependency during module import
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
    except ImportError as exc:  # pragma: no cover - requires optional runtime dep
        raise RuntimeError(
            "magic_pdf is required for PDF parsing. Please install it before uploading PDFs."
        ) from exc

    output_dir = ensure_dir(output_dir)
    image_dir = output_dir / "images"
    ensure_dir(image_dir)

    name_without_suffix = pdf_path.stem
    markdown_path = output_dir / f"{name_without_suffix}.md"

    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(str(pdf_path))

    dataset = PymuDocDataset(pdf_bytes)
    if dataset.classify() == SupportedPdfParseMethod.OCR:
        infer_result = dataset.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(FileBasedDataWriter(str(image_dir)))
    else:
        infer_result = dataset.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(FileBasedDataWriter(str(image_dir)))

    image_folder_name = image_dir.name if keep_images else ""
    pipe_result.dump_md(FileBasedDataWriter(str(output_dir)), f"{markdown_path.name}", image_folder_name)
    return markdown_path


def extract_markdown_chunks(markdown_path: Path) -> List[str]:
    """Split Markdown content into coarse-grained paragraphs for embedding."""

    content = markdown_path.read_text(encoding="utf-8")
    segments = [seg.strip() for seg in re.split(r"\n{2,}", content) if seg.strip()]
    return segments


def build_embedding_function(model_name: str, device: str = "cpu") -> BGEM3EmbeddingFunction:
    return BGEM3EmbeddingFunction(model_name=model_name, device=device)


def embed_documents(
    documents: List[str], embedding_function: BGEM3EmbeddingFunction
) -> Tuple[List[dict], dict]:
    """Embed documents and prepare records for insertion."""

    embeddings = embedding_function(documents)
    data_records = [
        {"subject": "user", "grade": "user", "title": f"chunk-{idx}", "content": doc}
        for idx, doc in enumerate(documents)
    ]
    return data_records, embeddings


def ingest_markdown_into_milvus(
    markdown_path: Path,
    db_uri: str,
    collection_name: str,
    embedding_model_name: str,
    embedding_device: str = "cpu",
) -> int:
    """Convert a Markdown file into Milvus vectors.

    Returns:
        Number of inserted chunks.
    """

    embedding_function = build_embedding_function(embedding_model_name, device=embedding_device)
    chunks = extract_markdown_chunks(markdown_path)
    data_records, embeddings = embed_documents(chunks, embedding_function)

    collection = vector_store_utils.create_collection(
        db_uri=db_uri, col_name=collection_name, dense_dim=embedding_function.dim["dense"]
    )
    vector_store_utils.add_data(collection, data_records, embeddings)
    return len(data_records)

