"""Ruajtja dhe hapja e një analize si JSON.

Një analizë e ruajtur mund të shfaqet sërish në ndërfaqe pa thirrur modelin
gjuhësor, për shembull gjatë një prezantimi kur API-ja është e mbingarkuar.
"""

import json
from datetime import datetime
from typing import Any, get_type_hints

from src.state import TechSpecState

FORMAT = "techspec-analysis"
VERSION = 1
STEPS = ("parser", "retriever", "requirement", "critic", "generator", "verifier")
_STATE_KEYS = frozenset(get_type_hints(TechSpecState))


class AnalysisFileError(ValueError):
    """Skedari nuk është një analizë e vlefshme; mesazhi është gati për përdoruesin."""


def analysis_to_json(analysis: dict[str, Any]) -> str:
    """Kthen analizën në JSON, me datën e ruajtjes."""
    return json.dumps(
        {
            "format": FORMAT,
            "version": VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "document": analysis["document"],
            "completed": analysis["completed"],
            "failed": analysis["failed"],
            "state": analysis["state"],
        },
        ensure_ascii=False,
        indent=2,
    )


def analysis_from_json(data: str | bytes) -> dict[str, Any]:
    """Lexon një analizë të ruajtur dhe kontrollon që ka formën e pritur.

    Raises:
        AnalysisFileError: skedari nuk është JSON, nuk është analizë e këtij
            projekti, ose i mungojnë fusha të gjendjes.
    """
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AnalysisFileError("Skedari nuk është JSON i vlefshëm.") from exc

    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise AnalysisFileError("Skedari nuk është një analizë e ruajtur nga TechSpec RAG.")
    if payload.get("version") != VERSION:
        raise AnalysisFileError(f"Versioni i skedarit ({payload.get('version')}) nuk mbështetet.")

    state = payload.get("state")
    if not isinstance(state, dict):
        raise AnalysisFileError("Skedarit i mungon gjendja e analizës.")
    missing = sorted(_STATE_KEYS - state.keys())
    if missing:
        raise AnalysisFileError(f"Skedarit i mungojnë fushat: {', '.join(missing)}.")

    steps = {"completed": payload.get("completed"), "failed": payload.get("failed")}
    for name, value in steps.items():
        if not isinstance(value, list) or not set(value) <= set(STEPS):
            raise AnalysisFileError(f"Lista '{name}' e hapave nuk është e vlefshme.")
    if not isinstance(payload.get("document"), str):
        raise AnalysisFileError("Skedarit i mungon emri i dokumentit.")

    return {
        "state": state,
        "completed": steps["completed"],
        "failed": steps["failed"],
        "document": payload["document"],
        "saved_at": payload.get("saved_at"),
    }
