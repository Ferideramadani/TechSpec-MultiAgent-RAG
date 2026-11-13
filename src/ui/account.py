"""Pjesët e ndërfaqes për llogaritë: hyrja, regjistrimi, historiku dhe paneli i administratorit."""

from typing import Any

import streamlit as st

from src.utils import database
from src.utils.analysis_io import AnalysisFileError, analysis_from_json

PAGE_ANALYSIS = "Analiza"
PAGE_ADMIN = "Administrimi"
ROLE_LABELS = {"user": "përdorues", "admin": "administrator"}
_SESSION_KEYS = ("logged_in", "user_id", "username", "role", "analysis_limit", "analysis", "page")


def is_logged_in() -> bool:
    return st.session_state.get("logged_in") is True


def is_admin() -> bool:
    return is_logged_in() and st.session_state.get("role") == "admin"


def start_session(user: dict[str, Any]) -> None:
    """Ruan të dhënat e përdoruesit të kyçur dhe nis me një ekran të pastër."""
    st.session_state.logged_in = True
    st.session_state.user_id = user["id"]
    st.session_state.username = user["username"]
    st.session_state.role = user["role"]
    st.session_state.analysis_limit = user["analysis_limit"]
    st.session_state.analysis = None


def logout() -> None:
    for key in _SESSION_KEYS:
        st.session_state.pop(key, None)


def quota_label(used: int, limit: int) -> str:
    return f"{used} analiza (pa kufi)" if limit == database.UNLIMITED else f"{used} / {limit} analiza"


def render_auth_page() -> None:
    """Faqja për përdoruesit që nuk janë kyçur: hyrja ose regjistrimi."""
    st.title("TechSpec RAG")
    st.caption("Analiza e dokumentacionit teknik me 6 agjentë. Kyçu ose krijo një llogari për të vazhduar.")
    login_tab, register_tab = st.tabs(["Kyçu", "Regjistrohu"])

    with login_tab, st.form("login_form"):
        username = st.text_input("Emri i përdoruesit", key="login_username")
        password = st.text_input("Fjalëkalimi", type="password", key="login_password")
        if st.form_submit_button("Kyçu", type="primary", key="login_submit"):
            user = database.login_user(username, password)
            if user is None:
                st.error("Emri i përdoruesit ose fjalëkalimi është i gabuar.")
            else:
                start_session(user)
                st.rerun()

    with register_tab, st.form("register_form"):
        username = st.text_input("Emri i përdoruesit", key="register_username", help="3 deri në 32 shkronja, shifra ose _ . -")
        password = st.text_input(
            "Fjalëkalimi", type="password", key="register_password",
            help=f"Të paktën {database.MIN_PASSWORD_LENGTH} karaktere.",
        )
        confirm = st.text_input("Përsërit fjalëkalimin", type="password", key="register_confirm")
        if st.form_submit_button("Regjistrohu", type="primary", key="register_submit"):
            if password != confirm:
                st.error("Fjalëkalimet nuk përputhen.")
            else:
                try:
                    database.register_user(username, password)
                except database.UserError as exc:
                    st.error(str(exc))
                else:
                    start_session(database.login_user(username, password))
                    st.rerun()


def render_account_sidebar() -> None:
    """Përdoruesi i kyçur, kuota, butoni Dil, zgjedhja e faqes për adminin dhe historiku."""
    with st.sidebar:
        role = ROLE_LABELS.get(st.session_state.role, st.session_state.role)
        user_column, logout_column = st.columns([3, 2], vertical_alignment="center")
        user_column.markdown(f"**{st.session_state.username}**  \n{role}")
        if logout_column.button("Dil", width="stretch", key="logout"):
            logout()
            st.rerun()

        try:
            _, used, limit = database.can_user_analyze(st.session_state.user_id)
        except database.UserError:
            # Llogaria nuk ekziston më (p.sh. databaza u rikrijua): sesioni mbyllet.
            logout()
            st.rerun()
        st.caption(f"Kuota: {quota_label(used, limit)}")

        if is_admin():
            st.radio("Faqja", [PAGE_ANALYSIS, PAGE_ADMIN], horizontal=True, key="page")
        st.divider()


def render_history_sidebar() -> None:
    """Analizat e mëparshme të përdoruesit; një klikim e hap analizën pa thirrur modelin gjuhësor."""
    with st.sidebar:
        st.subheader("Historiku im")
        history = database.get_user_history(st.session_state.user_id)
        if not history:
            st.caption("Ende nuk ke asnjë analizë.")
        for item in history:
            st.button(
                f"{item['created_at'][:16].replace('T', ' ')} · {item['pdf_name']}",
                help=item["question"],
                width="stretch",
                key=f"history_{item['id']}",
                on_click=open_history_item,
                args=(item["result_json"],),
            )


def open_history_item(result_json: str) -> None:
    """Hap një analizë nga historiku dhe kalon te faqja e analizës."""
    try:
        st.session_state.analysis = analysis_from_json(result_json)
    except AnalysisFileError as exc:
        st.session_state.history_error = f"Kjo analizë nuk mund të hapet: {exc}"
        return
    st.session_state.history_error = None
    if is_admin():
        st.session_state.page = PAGE_ANALYSIS


def render_history_error() -> None:
    if st.session_state.get("history_error"):
        st.error(st.session_state.history_error)


def render_admin_page() -> None:
    """Përdoruesit, kuotat dhe historiku i të gjithëve. Shfaqet vetëm për administratorët."""
    st.title("Paneli i administratorit")

    users = database.get_all_users()
    st.subheader("Ndrysho limitin e një përdoruesi")
    with st.form("limit_form"):
        user = st.selectbox(
            "Përdoruesi", users, format_func=lambda u: f"{u['username']} ({quota_label(u['analyses_used'], u['analysis_limit'])})",
            key="limit_user",
        )
        new_limit = st.number_input(
            "Limiti i ri", min_value=database.UNLIMITED, step=1, value=database.DEFAULT_ANALYSIS_LIMIT,
            help="-1 do të thotë pa kufi.", key="limit_value",
        )
        if st.form_submit_button("Ruaj limitin", type="primary", key="limit_submit"):
            try:
                database.update_user_limit(user["id"], int(new_limit))
            except database.UserError as exc:
                st.error(str(exc))
            else:
                st.success(f"Limiti i {user['username']} tani është {quota_label(user['analyses_used'], int(new_limit))}.")
                users = database.get_all_users()

    st.subheader("Përdoruesit")
    st.dataframe(
        [
            {
                "ID": u["id"],
                "Përdoruesi": u["username"],
                "Roli": ROLE_LABELS.get(u["role"], u["role"]),
                "Analiza të kryera": u["analyses_used"],
                # Vetëm tekst: një kolonë që përzien tekst dhe numra nuk mund të shfaqet nga pyarrow.
                "Limiti": "pa kufi" if u["analysis_limit"] == database.UNLIMITED else str(u["analysis_limit"]),
            }
            for u in users
        ],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Historiku i të gjithë përdoruesve")
    history = database.get_all_history()
    if not history:
        st.info("Ende nuk është kryer asnjë analizë.")
        return
    st.dataframe(
        [
            {
                "ID": item["id"],
                "Përdoruesi": item["username"],
                "Dokumenti": item["pdf_name"],
                "Pyetja": item["question"],
                "Data": item["created_at"].replace("T", " "),
            }
            for item in history
        ],
        hide_index=True,
        width="stretch",
    )
    selected = st.selectbox(
        "Hap një analizë", history,
        format_func=lambda item: f"#{item['id']} · {item['username']} · {item['pdf_name']} · {item['question'][:60]}",
        key="admin_history_item",
    )
    st.button(
        "Hap analizën", key="admin_open_history", on_click=open_history_item, args=(selected["result_json"],)
    )
