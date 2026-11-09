import json

import pytest

from src.state import create_initial_state
from src.utils.analysis_io import AnalysisFileError, analysis_from_json, analysis_to_json


def make_analysis():
    state = create_initial_state("data/spec.pdf", "How are orders created?")
    state["requirements"] = [{
        "requirement_id": "REQ-001",
        "kind": "functional",
        "description": "Users can create orders – with ë and ç.",
        "priority": "high",
        "source_chunk_ids": ["spec-p1-c00000"],
    }]
    state["quality_score"] = 80
    state["generated_documentation"] = "# Specifikimi teknik: spec.pdf\n"
    return {"state": state, "completed": ["parser", "retriever", "requirement"], "failed": ["critic"], "document": "spec.pdf"}


def saved(**changes):
    payload = json.loads(analysis_to_json(make_analysis()))
    payload.update(changes)
    return json.dumps(payload)


def test_round_trip_keeps_the_whole_analysis():
    original = make_analysis()

    loaded = analysis_from_json(analysis_to_json(original))

    assert loaded["state"] == original["state"]
    assert loaded["completed"] == original["completed"]
    assert loaded["failed"] == original["failed"]
    assert loaded["document"] == "spec.pdf"
    assert loaded["saved_at"]


def test_saved_file_is_readable_and_keeps_albanian_characters():
    text = analysis_to_json(make_analysis())

    assert "ë and ç" in text
    assert '"format": "techspec-analysis"' in text


def test_bytes_are_accepted():
    assert analysis_from_json(analysis_to_json(make_analysis()).encode("utf-8"))["document"] == "spec.pdf"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("not json", "nuk është JSON"),
        (b"\xff\xfe\x00", "nuk është JSON"),
        ("[1, 2]", "nuk është një analizë"),
        ('{"format": "something-else"}', "nuk është një analizë"),
        (saved(version=99), "Versioni"),
        (saved(state="text"), "mungon gjendja"),
        (saved(completed=["parser", "hacker"]), "'completed'"),
        (saved(failed="critic"), "'failed'"),
        (saved(document=None), "emri i dokumentit"),
    ],
)
def test_invalid_files_are_rejected_with_a_clear_message(data, message):
    with pytest.raises(AnalysisFileError, match=message):
        analysis_from_json(data)


def test_missing_state_fields_are_listed():
    payload = json.loads(analysis_to_json(make_analysis()))
    del payload["state"]["requirements"], payload["state"]["errors"]

    with pytest.raises(AnalysisFileError, match="errors, requirements"):
        analysis_from_json(json.dumps(payload))
