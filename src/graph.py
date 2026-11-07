"""Grafi LangGraph që lidh të gjashtë agjentët me radhë.

Pas çdo agjenti, grafi vazhdon te tjetri vetëm nëse `errors` është bosh. Kështu
një gabim ndalon ekzekutimin menjëherë dhe agjentët pasardhës nuk punojnë me
të dhëna të paplota.
"""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.critic import critique_requirements
from src.agents.generator import generate_tests_and_docs
from src.agents.parser import parse_document
from src.agents.requirement import extract_requirements
from src.agents.retriever import retrieve_chunks
from src.agents.verifier import verify_outputs
from src.state import TechSpecState, create_initial_state

AgentNode = Callable[[TechSpecState], dict[str, Any]]

PIPELINE: tuple[tuple[str, AgentNode], ...] = (
    ("parser", parse_document),
    ("retriever", retrieve_chunks),
    ("requirement", extract_requirements),
    ("critic", critique_requirements),
    ("generator", generate_tests_and_docs),
    ("verifier", verify_outputs),
)


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(TechSpecState)
    for name, agent in PIPELINE:
        graph.add_node(name, agent)

    graph.add_edge(START, PIPELINE[0][0])
    for (name, _), (next_name, _) in zip(PIPELINE, PIPELINE[1:]):
        graph.add_conditional_edges(name, _continue_to(next_name), [next_name, END])
    graph.add_edge(PIPELINE[-1][0], END)
    return graph.compile()


def _continue_to(next_name: str) -> Callable[[TechSpecState], str]:
    def route(state: TechSpecState) -> str:
        return END if state["errors"] else next_name

    return route


def run_pipeline(document_path: str, query: str) -> TechSpecState:
    """Ekzekuton të gjithë analizën për një dokument dhe një pyetje."""
    return build_graph().invoke(create_initial_state(document_path, query))
