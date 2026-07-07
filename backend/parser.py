"""
parser.py
Responsible ONLY for extracting and cleaning raw text from an uploaded PDF.
No AI processing happens in this module.
"""

import io
import fitz  # PyMuPDF

from utils import clean_text
from config import get_logger

logger = get_logger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF file cannot be parsed."""


def parse_resume_pdf(file_bytes: bytes) -> str:
    """
    Extract clean, plain text from a PDF file's raw bytes.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        Cleaned plain text extracted from the PDF.

    Raises:
        PDFParseError: If the file cannot be opened or contains no extractable text.
    """
    if not file_bytes:
        raise PDFParseError("Empty file provided.")

    try:
        document = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to open PDF: %s", exc)
        raise PDFParseError(f"Could not open PDF file: {exc}") from exc

    try:
        text_chunks = []
        for page in document:
            text_chunks.append(page.get_text("text"))
        raw_text = "\n".join(text_chunks)
    finally:
        document.close()

    cleaned = clean_text(raw_text)

    if not cleaned or len(cleaned) < 20:
        raise PDFParseError(
            "No readable text could be extracted from this PDF. "
            "It may be a scanned image without a text layer."
        )

    return cleaned
