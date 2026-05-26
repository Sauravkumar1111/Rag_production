"""
app.py — Streamlit RAG chat with source locations

Run:
    streamlit run app.py
"""

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import (
    ask,
    build_pipeline,
    doc_location,
    format_source_label,
    get_llm_info,
    get_pdf_page_text,
    validate_config,
)


def highlight_terms(text: str, answer: str) -> str:
    """Bold phrases from the answer that appear in the source chunk."""
    terms = []
    for part in re.split(r"[.!?\n]+", answer):
        part = part.strip()
        if len(part) > 12:
            terms.append(part)
    for term in sorted(terms, key=len, reverse=True)[:8]:
        if term.lower() in text.lower():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"**{m.group(0)}**", text, count=1)
    return text


@st.cache_resource(show_spinner="Loading RAG pipeline (PDFs + Notion)…")
def load_pipeline(llm_provider: str):
    validate_config(llm_provider=llm_provider)
    return build_pipeline(force_reingest=False, llm_provider=llm_provider)


def render_source(doc, rank: int, answer: str, key_prefix: str) -> None:
    loc = doc_location(doc)
    label = format_source_label(doc)

    with st.expander(f"**[{rank}]** {label}", expanded=(rank == 1)):
        c1, c2, c3 = st.columns(3)
        c1.metric("Document", loc["filename"])
        if loc["page_display"] is not None:
            c2.metric("PDF page", loc["page_display"])
        else:
            c2.metric("Location", "Notion" if loc["is_notion"] else "—")
        c3.metric("Chunk", loc["chunk_id"] or "—")

        st.caption(
            f"Chunk index: `{loc['chunk_index']}` · "
            f"Characters: `{loc['char_count']}` · "
            f"Path: `{loc['source']}`"
        )

        st.markdown("**Retrieved passage (used for answer)**")
        st.markdown(highlight_terms(doc.page_content, answer))

        if loc["filepath"] and loc["filepath"].endswith(".pdf") and loc["page"] is not None:
            try:
                page_idx = int(loc["page"])
                page_text = get_pdf_page_text(loc["filepath"], page_idx)
                if page_text.strip():
                    st.markdown(f"**Full PDF page {loc['page_display']} text**")
                    st.text_area(
                        f"Page {loc['page_display']} full text",
                        page_text,
                        height=220,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"{key_prefix}_page_{rank}_{loc['chunk_id']}_{loc['page']}",
                    )
            except Exception as e:
                st.warning(f"Could not load PDF page: {e}")

            pdf_path = Path(loc["filepath"])
            if pdf_path.exists():
                st.download_button(
                    f"Download {loc['filename']}",
                    data=pdf_path.read_bytes(),
                    file_name=loc["filename"],
                    mime="application/pdf",
                    key=f"{key_prefix}_dl_{rank}_{loc['chunk_id']}",
                )


def main() -> None:
    st.set_page_config(
        page_title="RAG Knowledge Base",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 RAG Knowledge Base")
    st.caption("Ask questions · Answers cite exact document, page, and chunk")

    with st.sidebar:
        st.header("Settings")

        llm_provider = st.selectbox(
            "Answer LLM",
            options=["openai", "ollama"],
            index=0 if os.getenv("LLM_PROVIDER", "openai").lower() != "ollama" else 1,
            help="OpenAI (cloud) or Ollama (local). Embeddings always use OpenAI.",
        )

        if llm_provider == "ollama":
            st.caption(f"Model: `{os.getenv('OLLAMA_MODEL', 'llama3.2')}`")
            st.caption(f"URL: `{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}`")
        else:
            st.caption(f"Model: `{os.getenv('OPENAI_LLM_MODEL', 'gpt-4o')}`")

        if st.button("🔄 Re-ingest all documents", use_container_width=True):
            load_pipeline.clear()
            with st.spinner("Clearing Pinecone and re-embedding…"):
                build_pipeline(force_reingest=True, llm_provider=llm_provider)
            load_pipeline.clear()
            st.success("Re-ingest complete.")
            st.rerun()

        st.divider()
        st.markdown(
            "**Sources**\n"
            "- `data/*.pdf` (ML papers)\n"
            "- Notion study page\n\n"
            "**Embeddings**\n"
            "- OpenAI `text-embedding-3-large`\n\n"
            "**Location shown**\n"
            "- File name · PDF page · Chunk ID"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        chain, retriever = load_pipeline(llm_provider)
        st.info(f"Answering with **{get_llm_info().get('label', llm_provider)}**")
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()

    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("docs"):
                st.markdown("---")
                st.markdown("**📎 Sources & exact locations**")
                for i, doc in enumerate(msg["docs"], 1):
                    render_source(doc, i, msg["content"], key_prefix=f"msg_{msg_idx}")

    question = st.chat_input("Ask a question about your documents…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                try:
                    answer, docs = ask(chain, retriever, question)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "docs": docs}
            )
            assistant_msg_idx = len(st.session_state.messages) - 1

            if docs:
                st.markdown("---")
                st.markdown("**📎 Sources & exact locations**")
                for i, doc in enumerate(docs, 1):
                    render_source(doc, i, answer, key_prefix=f"msg_{assistant_msg_idx}")
            else:
                st.warning("No source chunks retrieved.")


if __name__ == "__main__":
    main()
