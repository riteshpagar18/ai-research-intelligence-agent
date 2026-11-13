"""LangGraph workflow for planning, researching, verifying, and writing."""

from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.planner import create_research_plan
from agents.verifier import verify_sources
from agents.writer import generate_report
from tools.web_search import search_web


class ResearchState(TypedDict, total=False):
    question: str
    conversation_history: list[dict[str, str]]
    progress_callback: Callable[[str], None] | None
    plan: list[str]
    sources: list[dict[str, Any]]
    verification: dict[str, Any]
    report: str


def _progress(state: ResearchState, message: str) -> None:
    callback = state.get("progress_callback")
    if callback:
        callback(message)


def plan_research(state: ResearchState) -> dict[str, Any]:
    _progress(state, "Planning research tasks with Gemini...")
    plan = create_research_plan(
        state["question"], state.get("conversation_history")
    )
    return {"plan": plan}


def search_sources(state: ResearchState) -> dict[str, Any]:
    _progress(state, "Searching the web with Tavily...")
    sources = []
    seen_urls = set()
    for task in state["plan"]:
        for source in search_web(task):
            url = source.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({**source, "source_id": f"S{len(sources) + 1}"})
    if not sources:
        raise ValueError("No sources found. Try a different research question.")
    return {"sources": sources}


def verify_evidence(state: ResearchState) -> dict[str, Any]:
    _progress(state, "Checking claims against source passages...")
    return {"verification": verify_sources(state["question"], state["sources"])}


def write_report(state: ResearchState) -> dict[str, Any]:
    _progress(state, "Writing the cited research report...")
    report = generate_report(
        state["question"], state["sources"], state["verification"]
    )
    return {"report": report}


def build_research_graph():
    """Build the explicit, inspectable research graph."""
    graph = StateGraph(ResearchState)
    graph.add_node("planner", plan_research)
    graph.add_node("search", search_sources)
    graph.add_node("verifier", verify_evidence)
    graph.add_node("writer", write_report)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "verifier")
    graph.add_edge("verifier", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


def run_research_agent(question, progress_callback=None, conversation_history=None):
    """Run the full workflow and return a report with traceable evidence."""
    initial_state: ResearchState = {
        "question": question,
        "progress_callback": progress_callback,
        "conversation_history": conversation_history or [],
    }
    result = build_research_graph().invoke(initial_state)
    return {key: result[key] for key in ("plan", "sources", "verification", "report")}
