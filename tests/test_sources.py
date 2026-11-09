from src.utils.sources import describe_sources, format_chunk

CHUNKS = [
    {"chunk_id": "c1", "text": "Tokens expire.", "source": "spec.pdf", "page": 3, "section": "Security"},
    {"chunk_id": "c2", "text": "More security.", "source": "spec.pdf", "page": 3, "section": "Security"},
    {"chunk_id": "c3", "text": "Intro.", "source": "spec.pdf", "page": 1, "section": None},
    {"chunk_id": "c4", "text": "Quoted.", "source": "spec.pdf", "page": 5, "section": 'The "Fast" Path'},
]


def test_format_chunk_includes_section_only_when_known():
    assert format_chunk(CHUNKS[0]) == '<chunk id="c1" page="3" section="Security">\nTokens expire.\n</chunk>'
    assert format_chunk(CHUNKS[2]) == '<chunk id="c3" page="1">\nIntro.\n</chunk>'


def test_format_chunk_keeps_the_attribute_valid_when_the_section_has_quotes():
    assert 'section="The \'Fast\' Path"' in format_chunk(CHUNKS[3])


def test_describe_sources_sorts_by_page_and_merges_duplicates():
    assert describe_sources(["c1", "c3", "c2"], CHUNKS) == "1, 3 (Security)"


def test_describe_sources_ignores_unknown_ids():
    assert describe_sources(["missing"], CHUNKS) == ""
