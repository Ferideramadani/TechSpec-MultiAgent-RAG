from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from src.utils.pdf_loader import (
    PageText,
    PDFLoadError,
    chunk_pages,
    load_and_chunk_pdf,
    load_pdf,
)


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Krijon një PDF të vërtetë me tekst, ku çdo faqe është një listë rreshtash."""
    count = len(pages)
    page_ids = [4 + 2 * i for i in range(count)]
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Count %d /Kids [%s] >>"
        % (count, b" ".join(b"%d 0 R" % page_id for page_id in page_ids)),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, lines in zip(page_ids, pages):
        body = b"BT /F1 10 Tf 40 780 Td 12 TL "
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            body += b"(%s) Tj T* " % escaped.encode("latin-1")
        body += b"ET"
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (page_id + 1)
        )
        objects[page_id + 1] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(body), body)

    output = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(output)
        output += b"%d 0 obj\n%s\nendobj\n" % (object_id, objects[object_id])
    xref = len(output)
    size = len(objects) + 1
    output += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for object_id in sorted(objects):
        output += b"%010d 00000 n \n" % offsets[object_id]
    output += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref)
    path.write_bytes(output)
    return path


def test_load_pdf_keeps_page_numbers_and_skips_blank_pages(tmp_path):
    path = write_pdf(tmp_path / "spec.pdf", [["First page"], [], ["Third page"]])

    pages = load_pdf(path)

    assert [page.page for page in pages] == [1, 3]
    assert "First page" in pages[0].text
    assert "Third page" in pages[1].text


def test_load_and_chunk_pdf_returns_document_chunks(tmp_path):
    lines = [f"POST /api/v1/orders creates order number {i}." for i in range(30)]
    path = write_pdf(tmp_path / "api spec.pdf", [lines, lines])

    chunks = load_and_chunk_pdf(path, chunk_size=300, chunk_overlap=60)

    assert len(chunks) > 2
    assert {chunk["source"] for chunk in chunks} == {"api spec.pdf"}
    assert {chunk["page"] for chunk in chunks} == {1, 2}
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert chunks[0]["chunk_id"].startswith("api spec-p1-c00000")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "nuk u gjet"),
        (b"", "nuk mund"),
        (b"this is not a pdf", "nuk mund"),
    ],
)
def test_load_pdf_rejects_missing_empty_and_invalid_files(tmp_path, content, message):
    path = tmp_path / "bad.pdf"
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(PDFLoadError, match=message):
        load_pdf(path)


def test_load_pdf_rejects_pdf_without_text(tmp_path):
    path = write_pdf(tmp_path / "scanned.pdf", [[], []])

    with pytest.raises(PDFLoadError, match="tekst"):
        load_pdf(path)


def test_load_pdf_rejects_password_protected_file(tmp_path):
    source = write_pdf(tmp_path / "plain.pdf", [["Secret"]])
    writer = PdfWriter(clone_from=PdfReader(source))
    writer.encrypt("password", algorithm="RC4-128")
    locked = tmp_path / "locked.pdf"
    writer.write(locked)

    with pytest.raises(PDFLoadError, match="fjalëkalim"):
        load_pdf(locked)


def test_load_pdf_skips_unreadable_page_and_warns(tmp_path, monkeypatch, caplog):
    path = write_pdf(tmp_path / "spec.pdf", [["Good page"], ["Broken page"]])
    original = PdfReader.pages.fget

    def pages_with_broken_second(reader):
        pages = list(original(reader))
        pages[1].extract_text = lambda: 1 / 0
        return pages

    monkeypatch.setattr(PdfReader, "pages", property(pages_with_broken_second))

    pages = load_pdf(path)

    assert [page.page for page in pages] == [1]
    assert "Faqja 2" in caplog.text


def test_chunks_overlap_and_lose_no_words():
    words = [f"word{i}" for i in range(500)]
    pages = [PageText(1, " ".join(words))]

    chunks = chunk_pages(pages, "doc.pdf", chunk_size=200, chunk_overlap=50)

    assert all(len(chunk["text"]) <= 200 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        first_word = current["text"].split()[0]
        assert first_word in previous["text"].split()
    found = {word for chunk in chunks for word in chunk["text"].split()}
    assert found == set(words)


def test_chunks_prefer_paragraph_boundaries():
    first = "A" * 70
    second = "B" * 70
    pages = [PageText(1, f"{first}\n\n{second}")]

    chunks = chunk_pages(pages, "doc.pdf", chunk_size=100, chunk_overlap=0)

    assert [chunk["text"] for chunk in chunks] == [first, second]


def test_no_chunk_mixes_text_from_different_pages():
    pages = [PageText(2, "a " * 40), PageText(5, "b " * 40)]

    chunks = chunk_pages(pages, "doc.pdf", chunk_size=50, chunk_overlap=10)

    assert {chunk["page"] for chunk in chunks} == {2, 5}
    for chunk in chunks:
        expected_word = "a" if chunk["page"] == 2 else "b"
        assert set(chunk["text"].split()) == {expected_word}
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_chunk_ids_are_stable_between_runs():
    pages = [PageText(1, "stable text " * 100)]

    assert chunk_pages(pages, "doc.pdf") == chunk_pages(pages, "doc.pdf")


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(100, 99), (100, 90), (10, 0), (1, 0), (3, 2)],
)
@pytest.mark.parametrize("text", ["x" * 500, "ab " * 200, "line\n" * 100])
def test_splitting_always_terminates_and_keeps_the_end(text, size, overlap):
    chunks = chunk_pages([PageText(1, text.strip())], "doc.pdf", size, overlap)

    assert chunks
    assert all(len(chunk["text"]) <= size for chunk in chunks)
    assert text.strip().endswith(chunks[-1]["text"])
    assert len(chunks) <= len(text)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (100, 100), (100, -1)])
def test_chunk_pages_rejects_invalid_sizes(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "text")], "doc.pdf", size, overlap)
