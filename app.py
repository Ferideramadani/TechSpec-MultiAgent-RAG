"""Ndërfaqja Streamlit e TechSpec RAG: ekzekuton grafin me 6 agjentë dhe shfaq rezultatet.

Nisja: python -m streamlit run app.py
"""

import logging
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import PROJECT_ROOT
from src.graph import PIPELINE, build_graph
from src.state import RetrievedChunk, TechSpecState, create_initial_state
from src.utils.analysis_io import AnalysisFileError, analysis_from_json, analysis_to_json
from src.utils.sources import describe_sources

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
SOURCE_EXISTING = "Skedar ekzistues"
SOURCE_UPLOAD = "Ngarko PDF"

STEP_LABELS = {
    "parser": "Leximi dhe ndarja e dokumentit",
    "retriever": "Kërkimi i pjesëve relevante",
    "requirement": "Nxjerrja e kërkesave",
    "critic": "Vlerësimi i cilësisë",
    "generator": "Gjenerimi i testeve dhe dokumentacionit",
    "verifier": "Verifikimi i rezultateve",
}
# Agjentët i shënojnë gabimet me emrin e vet, që ndryshon nga emri i nyjes në graf.
AGENT_LABELS = {
    "document_parser": "Parser",
    "retriever": "Retriever",
    "requirement_extractor": "Requirement",
    "critic": "Critic",
    "generator": "Generator",
    "verifier": "Verifier",
}
KIND_LABELS = {"functional": "funksionale", "non_functional": "jofunksionale"}
TEST_TYPE_LABELS = {"positive": "pozitiv", "negative": "negativ", "boundary": "kufitar"}
STATUS_LABELS = {
    "supported": "i mbështetur",
    "partially_supported": "pjesërisht i mbështetur",
    "unsupported": "i pambështetur",
}


@st.cache_resource
def get_graph():
    return build_graph()


@contextmanager
def document_path(selected: Path | None, uploaded: Any) -> Iterator[Path]:
    """Jep rrugën e PDF-së. Skedari i ngarkuar ruhet përkohësisht vetëm gjatë analizës."""
    if selected:
        yield selected
        return
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(uploaded.name).name
        path.write_bytes(uploaded.getvalue())
        yield path


def run_analysis(path: Path, question: str) -> None:
    """Ekzekuton grafin, tregon progresin hap pas hapi dhe ruan rezultatin në sesion."""
    completed: list[str] = []  # Vetëm hapat që përfunduan me sukses.
    failed: list[str] = []
    final: TechSpecState | None = None
    with st.status("Duke analizuar dokumentin...", expanded=True) as status:
        try:
            for mode, chunk in get_graph().stream(
                create_initial_state(str(path), question), stream_mode=["updates", "values"]
            ):
                if mode == "updates":
                    for step, update in chunk.items():
                        if update and update.get("errors"):
                            failed.append(step)
                            status.write(f"Dështoi: {STEP_LABELS[step]}")
                        else:
                            completed.append(step)
                            status.write(f"U krye: {STEP_LABELS[step]}")
                else:
                    final = chunk
        except Exception as exc:
            # Gabimet e pritura i trajtojnë agjentët; këtu arrijnë vetëm problemet e sistemit.
            logger.exception("Grafi dështoi.")
            status.update(label="Analiza u ndërpre nga një gabim i sistemit.", state="error")
            st.session_state.analysis = None
            st.error(f"Gabim i sistemit: {exc}")
            return

        if final["errors"]:
            status.update(label="Analiza u ndal te një agjent.", state="error", expanded=False)
        else:
            status.update(label="Analiza përfundoi.", state="complete", expanded=False)

    st.session_state.analysis = {"state": final, "completed": completed, "failed": failed, "document": path.name}


def render_sidebar() -> None:
    with st.sidebar:
        st.title("TechSpec RAG")
        st.caption("Analiza e dokumentacionit teknik me 6 agjentë")

        st.subheader("Dokumenti")
        source = st.radio(
            "Burimi", [SOURCE_EXISTING, SOURCE_UPLOAD], horizontal=True, label_visibility="collapsed", key="source"
        )
        selected_path = None
        uploaded_file = None
        if source == SOURCE_EXISTING:
            files = sorted(path.name for path in DATA_DIR.glob("*.pdf"))
            selected_name = st.selectbox(
                "Skedari në dosjen data/", files, placeholder="Nuk ka PDF në data/", key="selected_file"
            )
            if selected_name:
                selected_path = DATA_DIR / selected_name
        else:
            uploaded_file = st.file_uploader("Zgjidh një PDF", type="pdf", key="uploaded_file")

        st.subheader("Pyetja")
        question = st.text_area(
            "Pyetja (në anglisht)",
            placeholder="What security capabilities must an API management solution provide?",
            key="question",
        )

        if st.button("Analizo", type="primary", width="stretch", key="analyze"):
            if not selected_path and not uploaded_file:
                st.error("Zgjidh ose ngarko një skedar PDF.")
            elif not question.strip():
                st.error("Shkruaj një pyetje për analizën.")
            else:
                with document_path(selected_path, uploaded_file) as path:
                    run_analysis(path, question.strip())
        st.caption("Analiza zgjat rreth 1 deri në 2 minuta, sepse 4 agjentë thërrasin modelin gjuhësor (Claude).")

        st.divider()
        st.subheader("Analizë e ruajtur")
        saved_file = st.file_uploader(
            "Hap një analizë të ruajtur (.json)",
            type="json",
            key="saved_analysis",
            help="Shfaq rezultatet e një analize të mëparshme pa thirrur modelin gjuhësor.",
        )
        # Hapet vetëm me klikim, që skedari i mbetur të mos mbishkruajë një analizë të re.
        if st.button("Hap", width="stretch", key="open_saved", disabled=saved_file is None):
            try:
                st.session_state.analysis = analysis_from_json(saved_file.getvalue())
            except AnalysisFileError as exc:
                st.error(str(exc))


def render_errors(state: TechSpecState, document: str) -> None:
    for error in state["errors"]:
        agent = AGENT_LABELS.get(error["agent"], error["agent"])
        # Te skedarët e ngarkuar rruga është dosje e përkohshme, ndaj shfaqet vetëm emri.
        message = error["message"].replace(state["document_path"], document)
        st.error(f"**{agent}:** {message}")


def render_metrics(state: TechSpecState, completed: list[str]) -> None:
    requirements, quality, tests, verification = st.columns(4)
    requirements.metric("Kërkesa", len(state["requirements"]) if "requirement" in completed else "-")
    has_score = "critic" in completed and state["quality_score"] is not None
    quality.metric("Cilësia", f"{state['quality_score']}%" if has_score else "-")
    tests.metric("Raste testimi", len(state["test_cases"]) if "generator" in completed else "-")
    if "verifier" in completed:
        verification.metric("Verifikimi", "Kaloi" if state["verification_passed"] else "Nuk kaloi")
    else:
        verification.metric("Verifikimi", "-")


def not_run(step: str) -> None:
    if step in st.session_state.analysis["failed"]:
        st.warning(f"Ky hap dështoi: {STEP_LABELS[step]}. Shih gabimin më lart.")
    else:
        st.info(f"Ky hap nuk u ekzekutua: {STEP_LABELS[step]}.")


def render_requirements(state: TechSpecState, completed: list[str]) -> None:
    if "requirement" not in completed:
        return not_run("requirement")
    if not state["requirements"]:
        st.info("Nuk u gjet asnjë kërkesë në pjesët e dokumentit që lidhen me pyetjen.")
    else:
        st.dataframe(
            [
                {
                    "ID": req["requirement_id"],
                    "Lloji": KIND_LABELS[req["kind"]],
                    "Prioriteti": req["priority"],
                    "Kërkesa": req["description"],
                    "Burimi (faqja, seksioni)": describe_sources(req["source_chunk_ids"], state["retrieved_chunks"]),
                }
                for req in state["requirements"]
            ],
            hide_index=True,
            width="stretch",
        )
    if state["api_endpoints"]:
        st.subheader("Pikat API")
        st.dataframe(
            [
                {
                    "Metoda": endpoint["method"],
                    "Rruga": endpoint["path"],
                    "Përshkrimi": endpoint["description"],
                    "Parametrat": ", ".join(endpoint["parameters"]),
                }
                for endpoint in state["api_endpoints"]
            ],
            hide_index=True,
            width="stretch",
        )


def render_quality(state: TechSpecState, completed: list[str]) -> None:
    if "critic" not in completed:
        return not_run("critic")
    if not state["requirement_reviews"]:
        st.info(state["quality_summary"] or "Nuk ka vlerësim.")
        return
    st.write(state["quality_summary"])
    descriptions = {req["requirement_id"]: req["description"] for req in state["requirements"]}
    for review in state["requirement_reviews"]:
        with st.expander(f"{review['requirement_id']} · {review['score']}%"):
            st.progress(review["score"] / 100)
            st.markdown("**Origjinale**")
            st.text(descriptions.get(review["requirement_id"], ""))
            if review["issues"]:
                st.markdown("**Problemet**")
                for issue in review["issues"]:
                    st.text(f"{issue['criterion']}: {issue['problem']}")
                st.markdown("**Versioni i përmirësuar**")
                st.text(review["improved_version"])
            else:
                st.success("Kërkesa plotëson të pesë kriteret.")


def render_tests(state: TechSpecState, completed: list[str]) -> None:
    if "generator" not in completed:
        return not_run("generator")
    if not state["test_cases"]:
        st.info("Nuk u gjenerua asnjë rast testimi, sepse nuk ka kërkesa.")
        return
    for case in state["test_cases"]:
        with st.expander(f"{case['test_id']} · {case['title']}"):
            st.caption(f"Lloji: {TEST_TYPE_LABELS[case['test_type']]} · Verifikon: {', '.join(case['requirement_ids'])}")
            if case["preconditions"]:
                st.markdown("**Parakushtet**")
                st.text("\n".join(f"- {condition}" for condition in case["preconditions"]))
            st.markdown("**Hapat**")
            st.text("\n".join(f"{number}. {step}" for number, step in enumerate(case["steps"], start=1)))
            st.markdown("**Rezultati i pritur**")
            st.text(case["expected_result"])


def render_verification(state: TechSpecState, completed: list[str]) -> None:
    if "verifier" not in completed:
        return not_run("verifier")
    results = state["verification_results"]
    if state["verification_passed"]:
        st.success("Verifikimi kaloi: asnjë element nuk është i pambështetur nga dokumenti.")
    else:
        st.error("Verifikimi nuk kaloi: të paktën një element është i pambështetur ose pa verdikt.")
    if not results:
        st.info("Nuk kishte asgjë për të verifikuar.")
        return
    counts = Counter(result["status"] for result in results)
    st.caption(" · ".join(f"{label}: {counts[status]}" for status, label in STATUS_LABELS.items()))
    st.dataframe(
        [
            {
                "Elementi": result["item_id"],
                "Statusi": STATUS_LABELS[result["status"]],
                "Shpjegimi": result["explanation"],
                "Supozime të pabazuara": "; ".join(result["unsupported_claims"]),
            }
            for result in results
        ],
        hide_index=True,
        width="stretch",
    )


def render_documentation(state: TechSpecState, document: str) -> None:
    documentation = state["generated_documentation"]
    if not documentation:
        st.info("Dokumentacioni krijohet pasi të përfundojë gjeneruesi.")
        return
    st.download_button(
        "Shkarko dokumentacionin (.md)",
        documentation,
        file_name=f"{Path(document).stem}_specifikimi.md",
        mime="text/markdown",
        key="download_doc",
    )
    with st.container(border=True):
        st.markdown(documentation)


def render_chunk(rank: int, chunk: RetrievedChunk) -> None:
    # Ngjashmëria kosinus mund të jetë edhe negative, por shiriti pranon vetëm 0 deri në 1.
    similarity = max(0.0, min(1.0, chunk["score"]))
    with st.container(border=True):
        title, score = st.columns([4, 1])
        section = f" · {chunk['section']}" if chunk.get("section") else ""
        title.markdown(f"**#{rank} · Faqja {chunk['page']}{section}**")
        score.markdown(f"**{similarity:.1%}**")
        st.progress(similarity)
        # st.code e ruan tekstin ashtu siç është, pa e interpretuar si Markdown.
        st.code(chunk["text"], language=None, wrap_lines=True)
        # Analizat e ruajtura para kërkimit hibrid nuk e kanë këtë fushë.
        keywords = " · gjetur edhe me fjalë kyçe" if chunk.get("keyword_score", 0) > 0 else ""
        st.caption(f"{chunk['chunk_id']}{keywords}")


def render_sources(state: TechSpecState, completed: list[str]) -> None:
    if "retriever" not in completed or not state["retrieved_chunks"]:
        return not_run("retriever")
    for rank, chunk in enumerate(state["retrieved_chunks"], start=1):
        render_chunk(rank, chunk)


def render_main() -> None:
    st.title("Analiza e dokumentacionit teknik")
    analysis = st.session_state.analysis
    if not analysis:
        st.info("Zgjidh një dokument, shkruaj pyetjen dhe kliko **Analizo** në panelin anësor.")
        st.caption(" → ".join(STEP_LABELS[name] for name, _ in PIPELINE))
        return

    state, completed, document = analysis["state"], analysis["completed"], analysis["document"]
    info, save = st.columns([4, 1], vertical_alignment="center")
    info.caption(f"Dokumenti: {document} · Pyetja: {state['query']}")
    save.download_button(
        "Ruaj analizën (.json)",
        analysis_to_json(analysis),
        file_name=f"{Path(document).stem}_analiza.json",
        mime="application/json",
        width="stretch",
        key="download_analysis",
    )
    if analysis.get("saved_at"):
        st.info(
            f"Kjo është një analizë e ruajtur më {analysis['saved_at'].replace('T', ' ')}. "
            "Rezultatet shfaqen nga skedari, pa thirrur modelin gjuhësor."
        )
    render_errors(state, document)
    render_metrics(state, completed)

    tabs = st.tabs(["Kërkesat", "Cilësia", "Testet", "Verifikimi", "Dokumentacioni", "Burimet"])
    with tabs[0]:
        render_requirements(state, completed)
    with tabs[1]:
        render_quality(state, completed)
    with tabs[2]:
        render_tests(state, completed)
    with tabs[3]:
        render_verification(state, completed)
    with tabs[4]:
        render_documentation(state, document)
    with tabs[5]:
        render_sources(state, completed)


def main() -> None:
    st.set_page_config(page_title="TechSpec RAG", page_icon=":material/description:", layout="wide")
    st.session_state.setdefault("analysis", None)
    render_sidebar()
    render_main()


main()
