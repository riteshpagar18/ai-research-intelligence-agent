import os

import streamlit as st
from dotenv import load_dotenv
from config import load_streamlit_secrets

load_dotenv()
load_streamlit_secrets(st)

st.set_page_config(page_title="AI Research Agent", page_icon="🔍", layout="wide")
st.title("🔍 AI Research & Intelligence Agent")
st.write(
    "Research with a transparent, multi-step workflow: Gemini plans and assesses "
    "evidence, Tavily finds sources, and Gemini writes a cited report."
)

if "research_history" not in st.session_state:
    st.session_state.research_history = []


def show_result(result):
    st.subheader("Research plan")
    for task in result["plan"]:
        st.markdown(f"- {task}")

    st.subheader("Research report")
    st.markdown(result["report"])

    verification = result.get("verification", {})
    with st.expander("Evidence assessment"):
        if verification.get("raw_assessment"):
            st.markdown(verification["raw_assessment"])
        claims = verification.get("claims", [])
        if claims:
            for claim in claims:
                status = claim.get("status", "unknown").replace("_", " ").title()
                st.markdown(f"**{status}:** {claim.get('claim', '')}")
                for item in claim.get("evidence", []):
                    st.markdown(
                        f"> “{item.get('quote', '')}” — [{item.get('source_id', '')}]"
                    )
        for limitation in verification.get("limitations", []):
            st.markdown(f"- Limitation: {limitation}")
        st.caption("AI-assisted evidence assessment; this is not independent proof.")

    st.subheader("Research sources")
    for source in result["sources"]:
        sid = source.get("source_id", "Source")
        with st.expander(f"[{sid}] {source['title']}"):
            st.write(source["content"])
            st.link_button("Open source", source["url"])

    from tools.pdf_export import create_report_pdf

    pdf_bytes = create_report_pdf(result["report"], result["sources"])
    left, right = st.columns(2)
    left.download_button(
        "Download report as PDF",
        data=pdf_bytes,
        file_name="research_report.pdf",
        mime="application/pdf",
        key=f"pdf-{result['id']}",
    )
    right.download_button(
        "Download report as Markdown",
        data=result["report"],
        file_name="research_report.md",
        mime="text/markdown",
        key=f"md-{result['id']}",
    )


with st.form("research_form"):
    question = st.text_area(
        "Research question",
        placeholder="Example: How is generative AI changing software development?",
        height=120,
    )
    submitted = st.form_submit_button("Start research", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please enter a research question.")
    elif len(question.strip()) < 10:
        st.warning("Please enter a more detailed research question.")
    elif not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is missing from your .env file.")
    elif not os.getenv("TAVILY_API_KEY"):
        st.error("TAVILY_API_KEY is missing from your .env file.")
    else:
        try:
            from workflow.pipeline import run_research_agent

            with st.status("Starting research...", expanded=True) as status:
                result = run_research_agent(
                    question.strip(),
                    progress_callback=lambda message: status.update(
                        label=message, state="running"
                    ),
                    conversation_history=st.session_state.research_history,
                )
                status.update(label="Research completed", state="complete")

            result["id"] = len(st.session_state.research_history) + 1
            result["question"] = question.strip()
            result["summary"] = result["report"][:700]
            st.session_state.research_history.append(result)
        except Exception as exc:
            st.error(f"Research failed: {exc}")
            st.info("Check your API keys, connection, and installed packages.")

if st.session_state.research_history:
    st.divider()
    history = st.session_state.research_history
    labels = [f"{item['id']}. {item['question']}" for item in history]
    selected = st.selectbox(
        "Conversation history (stored for this browser session)",
        range(len(history)),
        format_func=lambda index: labels[index],
    )
    result_to_show = history[selected]
    show_result(result_to_show)

st.divider()
st.caption(
    "Python · Streamlit · LangGraph · Google Gemini · Tavily | "
    "Evidence should be checked against the linked sources."
)
