"""Leximi i dokumenteve PDF dhe ndarja e tyre në pjesë me mbivendosje për RAG."""

import logging
import re
from collections import Counter
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

# Një rresht është titull kur shkronjat e tij janë të paktën kaq herë më të mëdha se teksti kryesor.
HEADING_SIZE_RATIO = 1.2
_MAX_HEADING_LENGTH = 120
# Një titull që fillon brenda kaq karaktereve nga fillimi i pjesës e emërton atë pjesë.
_HEADING_LOOKAHEAD = 100


class PDFLoadError(Exception):
    """PDF-ja nuk mund të lexohet ose nuk përmban tekst të nxjerrshëm."""


class PageText(NamedTuple):
    page: int  # Numri i faqes, duke nisur nga 1.
    text: str
    headings: tuple[str, ...] = ()  # Titujt e seksioneve në faqe, sipas radhës.


class _TextRun(NamedTuple):
    text: str
    size: float


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

    extracted: list[tuple[int, str, list[_TextRun]]] = []
    for number, raw_page in enumerate(raw_pages, start=1):
        runs: list[_TextRun] = []
        try:
            text = _clean_text(raw_page.extract_text(visitor_text=_run_collector(runs)) or "")
        except Exception as exc:  # pypdf ngre lloje të ndryshme gabimesh për faqe të dëmtuara.
            logger.warning("Faqja %d e %s u anashkalua: %s", number, path.name, exc)
            continue
        if text:
            extracted.append((number, text, runs))

    if not extracted:
        raise PDFLoadError(f"PDF-ja nuk ka tekst të nxjerrshëm (mund të jetë e skanuar): {path}")

    body_size = _body_font_size(run for _, _, runs in extracted for run in runs)
    return [
        PageText(number, text, tuple(name for _, name in _locate_headings(text, _find_headings(runs, body_size))))
        for number, text, runs in extracted
    ]


def chunk_pages(
    pages: Sequence[PageText],
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Ndan faqet në pjesë të mbivendosura.

    Ndarja bëhet brenda secilës faqe, që `page` i një pjese të jetë gjithmonë i
    saktë për citimet. Si pasojë, një fjali që kalon nga një faqe te tjetra ndahet.
    `section` është titulli i fundit para pjesës, edhe nga një faqe e mëparshme.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size duhet të jetë të paktën 1.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap duhet të jetë >= 0 dhe më i vogël se chunk_size.")

    stem = Path(source).stem
    chunks: list[DocumentChunk] = []
    section: str | None = None
    for page in pages:
        positions = _locate_headings(page.text, page.headings)
        for start, text in _split_text(page.text, chunk_size, chunk_overlap):
            chunk_section = section
            for position, heading in positions:
                if position <= start + _HEADING_LOOKAHEAD:
                    chunk_section = heading
            chunks.append(
                {
                    # ID-ja e qëndrueshme lejon përditësimin e dokumentit në bazën vektoriale.
                    "chunk_id": f"{stem}-p{page.page}-c{len(chunks):05d}",
                    "text": text,
                    "source": source,
                    "page": page.page,
                    "section": chunk_section,
                }
            )
        if positions:
            section = positions[-1][1]
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


def _run_collector(runs: list[_TextRun]):
    """Kthen funksionin që pypdf thërret për çdo copëz teksti, me madhësinë reale të shkronjave."""

    def collect(text, cm, tm, font_dict, font_size):
        scale = abs(tm[3] or tm[0] or 1) * abs(cm[3] or cm[0] or 1)
        runs.append(_TextRun(text, round(font_size * scale, 1)))

    return collect


def _body_font_size(runs: Iterator[_TextRun]) -> float:
    """Madhësia e tekstit kryesor: ajo që mbulon më shumë karaktere në dokument."""
    sizes: Counter[float] = Counter()
    for run in runs:
        sizes[run.size] += len(run.text.strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _find_headings(runs: Sequence[_TextRun], body_size: float) -> tuple[str, ...]:
    """Bashkon copëzat me shkronja më të mëdha se teksti kryesor në tituj.

    Hapësirat mes copëzave mund të mos jenë të sakta; emri përfundimtar merret
    nga teksti i faqes te `_locate_headings`.
    """
    threshold = body_size * HEADING_SIZE_RATIO
    groups: list[list[str]] = []
    current_size: float | None = None
    for run in runs:
        if not run.text.strip():
            if current_size is not None:
                groups[-1].append(" ")
            continue
        if body_size <= 0 or run.size < threshold:
            current_size = None
            continue
        if current_size is None or abs(run.size - current_size) > 0.5:
            groups.append([])
            current_size = run.size
        groups[-1].append(run.text)

    headings: list[str] = []
    for group in groups:
        heading = " ".join("".join(group).split())
        if 2 <= len(heading) <= _MAX_HEADING_LENGTH and any(char.isalpha() for char in heading):
            if not headings or headings[-1] != heading:
                headings.append(heading)
    return tuple(headings)


def _locate_headings(text: str, headings: Sequence[str]) -> list[tuple[int, str]]:
    """Gjen çdo titull në tekstin e faqes dhe kthen (pozicioni, emri siç shfaqet në tekst).

    Krahasimi nuk merr parasysh hapësirat dhe shkronjat e mëdha, sepse copëzat e PDF-së
    shpesh e ndajnë një fjalë në mes ("AT A GL" + "ANCE").
    """
    positions: list[tuple[int, str]] = []
    search_from = 0
    for heading in headings:
        pattern = r"\s*".join(re.escape(char) for char in heading if not char.isspace())
        match = re.compile(pattern, re.IGNORECASE).search(text, search_from)
        if match:
            positions.append((match.start(), " ".join(match.group().split())))
            search_from = match.end()
    return positions


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[tuple[int, str]]:
    """Kthen çifte (pozicioni i fillimit në faqe, teksti) për secilën pjesë."""
    start = 0
    while start < len(text):
        end = _find_chunk_end(text, start, chunk_size, chunk_overlap)
        raw = text[start:end]
        piece = raw.strip()
        if piece:
            yield start + len(raw) - len(raw.lstrip()), piece
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
