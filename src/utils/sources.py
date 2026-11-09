"""Formatimi i pjesëve të dokumentit dhe i burimeve të tyre (faqja dhe seksioni)."""

from collections.abc import Iterable, Sequence

from src.state import DocumentChunk


def format_chunk(chunk: DocumentChunk) -> str:
    """Pjesa e dokumentit si tag për udhëzimet e modelit; seksioni shtohet vetëm kur dihet."""
    section = chunk.get("section")
    section_attribute = f' section="{section.replace(chr(34), chr(39))}"' if section else ""
    return f'<chunk id="{chunk["chunk_id"]}" page="{chunk["page"]}"{section_attribute}>\n{chunk["text"]}\n</chunk>'


def describe_sources(chunk_ids: Iterable[str], chunks: Sequence[DocumentChunk]) -> str:
    """Faqet dhe seksionet e pjesëve të cituara, p.sh. "2 (Definitions), 3 (Benefits)"."""
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    cited = [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id and by_id[chunk_id]["page"] is not None]
    labels: list[str] = []
    for chunk in sorted(cited, key=lambda c: c["page"]):
        label = f"{chunk['page']} ({chunk['section']})" if chunk.get("section") else str(chunk["page"])
        if label not in labels:
            labels.append(label)
    return ", ".join(labels)
