"""Leximi i dokumenteve PDF dhe ndarja e tyre në pjesë me mbivendosje për RAG."""

import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

from pypdf import PasswordType, PdfReader
from pypdf.errors import DependencyError, PyPdfError

from src.state import DocumentChunk

logger = logging.getLogger(__name__)

# Madhësitë maten në karaktere.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Pikat e preferuara të ndarjes, nga më e mira te më e dobëta.
_BREAK_POINTS = ("\n\n", "\n", ". ", " ")


class PDFLoadError(Exception):
    """PDF-ja nuk mund të lexohet ose nuk përmban tekst të nxjerrshëm."""


class PageText(NamedTuple):
    page: int  # Numri i faqes, duke nisur nga 1.
    text: str


def load_pdf(path: str | Path) -> list[PageText]:
    """Lexon tekstin e PDF-së dhe kthen vetëm faqet që përmbajnë tekst.

    Një faqe që nuk lexohet anashkalohet me paralajmërim, që një faqe e dëmtuar
    të mos e bllokojë gjithë dokumentin.

    Raises:
        PDFLoadError: skedari mungon, është i dëmtuar, i mbrojtur me fjalëkalim,
            ose nuk ka tekst (për shembull një PDF e skanuar).
    """
    path = Path(path)
    if not path.is_file():
        raise PDFLoadError(f"Skedari nuk u gjet: {path}")

    try:
        reader = PdfReader(path)
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise PDFLoadError(f"PDF-ja është e mbrojtur me fjalëkalim: {path}")
        raw_pages = list(reader.pages)
    except (PyPdfError, DependencyError, OSError) as exc:
        raise PDFLoadError(f"PDF-ja nuk mund të lexohet: {path} ({exc})") from exc

    pages = []
    for number, raw_page in enumerate(raw_pages, start=1):
        try:
            text = _clean_text(raw_page.extract_text() or "")
        except Exception as exc:  # pypdf ngre lloje të ndryshme gabimesh për faqe të dëmtuara.
            logger.warning("Faqja %d e %s u anashkalua: %s", number, path.name, exc)
            continue
        if text:
            pages.append(PageText(number, text))

    if not pages:
        raise PDFLoadError(f"PDF-ja nuk ka tekst të nxjerrshëm (mund të jetë e skanuar): {path}")
    return pages


def chunk_pages(
    pages: Sequence[PageText],
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Ndan faqet në pjesë të mbivendosura.

    Ndarja bëhet brenda secilës faqe, që `page` i një pjese të jetë gjithmonë i
    saktë për citimet. Si pasojë, një fjali që kalon nga një faqe te tjetra ndahet.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size duhet të jetë të paktën 1.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap duhet të jetë >= 0 dhe më i vogël se chunk_size.")

    stem = Path(source).stem
    chunks: list[DocumentChunk] = []
    for page in pages:
        for text in _split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                {
                    # ID-ja e qëndrueshme lejon përditësimin e dokumentit në bazën vektoriale.
                    "chunk_id": f"{stem}-p{page.page}-c{len(chunks):05d}",
                    "text": text,
                    "source": source,
                    "page": page.page,
                    "section": None,
                }
            )
    return chunks


def load_and_chunk_pdf(
    path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Lexon një PDF dhe e kthen të ndarë në pjesë, gati për bazën vektoriale."""
    path = Path(path)
    return chunk_pages(load_pdf(path), path.name, chunk_size, chunk_overlap)


def _clean_text(text: str) -> str:
    """Pastron tekstin pa prekur dhëmbëzimin, që është i rëndësishëm te shembujt e kodit."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = _find_chunk_end(text, start, chunk_size, chunk_overlap)
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= len(text):
            return
        start = _next_start(text, end, chunk_overlap)


def _find_chunk_end(text: str, start: int, chunk_size: int, chunk_overlap: int) -> int:
    hard_end = start + chunk_size
    if hard_end >= len(text):
        return len(text)
    # Pjesa duhet të jetë më e gjatë se mbivendosja, përndryshe ndarja nuk përparon.
    earliest = start + max(chunk_size // 2, chunk_overlap + 1)
    for separator in _BREAK_POINTS:
        position = text.rfind(separator, earliest, hard_end)
        if position != -1:
            return position + len(separator)
    return hard_end


def _next_start(text: str, end: int, chunk_overlap: int) -> int:
    start = end - chunk_overlap
    if chunk_overlap == 0 or text[start - 1].isspace():
        return start
    # Mbivendosja nis te fjala tjetër e plotë, jo në mes të një fjale.
    match = re.compile(r"\s").search(text, start, end)
    return match.end() if match else start
