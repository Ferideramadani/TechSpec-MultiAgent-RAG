"""Ndërfaqja Streamlit për testimin e agjentëve parser dhe retriever.

Nisja: python -m streamlit run app.py
"""

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from src.agents.parser import parse_document
from src.agents.retriever import retrieve_chunks
from src.config import PROJECT_ROOT
from src.state import RetrievedChunk, TechSpecState, create_initial_state

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
SOURCE_EXISTING = "Skedar ekzistues"
SOURCE_UPLOAD = "Ngarko PDF"


def run_agent(agent: Callable[[TechSpecState], dict[str, Any]], state: TechSpecState) -> dict[str, Any]:
    """Ekzekuton agjentin dhe i shndërron gabimet e sistemit në mesazhe për përdoruesin."""
    try:
        return agent(state)
    except Exception as exc:
        logger.exception("Agjenti %s dështoi.", agent.__name__)
        return {"errors": [{"agent": agent.__name__, "message": f"Gabim i sistemit: {exc}"}]}


def process_document(path: Path) -> None:
    """Ekzekuton parser-in. `document` ndryshon vetëm nëse përpunimi ka sukses."""
    result = run_agent(parse_document, create_initial_state(str(path), ""))
    for error in result.get("errors", []):
        # Rruga e plotë është dosje e përkohshme te ngarkimet, ndaj shfaqet vetëm emri.
        st.error(error["message"].replace(str(path), path.name))
    if result.get("errors"):
        return

    st.session_state.document = {
        "name": path.name,
        "page_count": result["page_count"],
        "chunk_count": result["chunk_count"],
    }
    st.session_state.search = None
    st.success(f"U përpunua: {path.name}")


def process_upload(uploaded_file: Any) -> None:
    """Përpunon skedarin e ngarkuar. Përmbajtja mbetet te baza vektoriale, jo në disk."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(uploaded_file.name).name
        path.write_bytes(uploaded_file.getvalue())
        process_document(path)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("TechSpec RAG")
        st.caption("Paneli i kontrollit")

        st.subheader("Dokumenti")
        source = st.radio(
            "Burimi",
            [SOURCE_EXISTING, SOURCE_UPLOAD],
            horizontal=True,
            label_visibility="collapsed",
            key="source",
        )

        selected_path = None
        uploaded_file = None
        if source == SOURCE_EXISTING:
            files = sorted(path.name for path in DATA_DIR.glob("*.pdf"))
            selected_name = st.selectbox(
                "Skedari në dosjen data/",
                files,
                placeholder="Nuk ka PDF në data/",
                key="selected_file",
            )
            if selected_name:
                selected_path = DATA_DIR / selected_name
        else:
            uploaded_file = st.file_uploader("Zgjidh një PDF", type="pdf", key="uploaded_file")

        if st.button("Përpuno dokumentin", type="primary", use_container_width=True, key="process"):
            with st.spinner("Duke lexuar dhe ruajtur dokumentin..."):
                if selected_path:
                    process_document(selected_path)
                elif uploaded_file:
                    process_upload(uploaded_file)
                else:
                    st.error("Zgjidh ose ngarko një skedar PDF.")

        document = st.session_state.document
        if document:
            st.divider()
            st.subheader("Dokumenti aktiv")
            st.write(document["name"])
            pages, chunks = st.columns(2)
            pages.metric("Faqe me tekst", document["page_count"])
            chunks.metric("Pjesë", document["chunk_count"])


def run_search(question: str) -> None:
    """Ekzekuton retriever-in mbi dokumentin aktiv dhe ruan rezultatin në sesion."""
    st.session_state.search = None
    document = st.session_state.document
    if not document:
        st.error("Përpuno së pari një dokument nga paneli anësor.")
        return

    result = run_agent(retrieve_chunks, create_initial_state(document["name"], question))
    for error in result.get("errors", []):
        st.error(error["message"])
    if not result.get("errors"):
        st.session_state.search = {"question": question, "chunks": result["retrieved_chunks"]}


def render_chunk(rank: int, chunk: RetrievedChunk) -> None:
    # Ngjashmëria kosinus mund të jetë edhe negative, por shiriti pranon vetëm 0 deri në 1.
    similarity = max(0.0, min(1.0, chunk["score"]))
    with st.container(border=True):
        title, score = st.columns([4, 1])
        title.markdown(f"**#{rank} · Faqja {chunk['page']}**")
        score.markdown(f"**{similarity:.1%}**")
        st.progress(similarity)
        # st.code e ruan tekstin ashtu siç është, pa e interpretuar si Markdown.
        st.code(chunk["text"], language=None, wrap_lines=True)
        st.caption(chunk["chunk_id"])


def render_main() -> None:
    st.title("Kërkimi në dokumentacionin teknik")
    st.caption("Testo agjentët Document Parser dhe Retrieval mbi dokumentet e tua.")

    with st.form("search_form"):
        question = st.text_input(
            "Pyetja (në anglisht)",
            placeholder="What are the benefits of API management?",
            key="question",
        )
        submitted = st.form_submit_button("Kërko", type="primary", key="search_button")

    if submitted:
        with st.spinner("Duke kërkuar..."):
            run_search(question)

    search = st.session_state.search
    if search:
        st.subheader(f"Pjesët e gjetura: {len(search['chunks'])}")
        st.caption(f"Pyetja: {search['question']}")
        for rank, chunk in enumerate(search["chunks"], start=1):
            render_chunk(rank, chunk)
    elif not st.session_state.document:
        st.info("Fillo duke përpunuar një dokument nga paneli anësor.")


def main() -> None:
    st.set_page_config(page_title="TechSpec RAG", page_icon=":material/description:", layout="wide")
    st.session_state.setdefault("document", None)
    st.session_state.setdefault("search", None)
    render_sidebar()
    render_main()


main()
