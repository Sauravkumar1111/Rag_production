"""
Production RAG System — 2026
Stack: LangChain · OpenAI Embeddings · Pinecone · OpenAI or Ollama LLM
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# LangChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
NOTION_API_KEY      = os.getenv("NOTION_API_KEY")
NOTION_PAGE_ID      = os.getenv("NOTION_PAGE_ID")
NOTION_DATABASE_ID  = os.getenv("NOTION_DATABASE_ID")
INDEX_NAME          = os.getenv("PINECONE_INDEX", "rag-production-2026")
EMBED_MODEL         = "text-embedding-3-large"   # 3072-dim, best quality
EMBED_DIMS          = 3072
LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_LLM_MODEL    = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_active_llm_info: dict = {}


def validate_config(llm_provider: str | None = None) -> None:
    """Fail fast when required API keys or data sources are missing."""
    missing = [k for k in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and add your keys."
        )

    has_notion = bool(NOTION_API_KEY and (NOTION_PAGE_ID or NOTION_DATABASE_ID))
    data_path = Path(os.getenv("DATA_DIR", "data"))
    has_local = data_path.exists() and any(data_path.rglob("*.*"))

    if NOTION_API_KEY and not NOTION_PAGE_ID and not NOTION_DATABASE_ID:
        print("⚠  NOTION_API_KEY set but NOTION_PAGE_ID / NOTION_DATABASE_ID missing — Notion skipped")
    elif NOTION_API_KEY:
        parts = []
        if NOTION_PAGE_ID:
            parts.append("page")
        if NOTION_DATABASE_ID:
            parts.append("database")
        print(f"✅ Notion configured ({', '.join(parts)})")

    if not has_notion and not has_local:
        raise EnvironmentError(
            "No data sources configured. Set NOTION_PAGE_ID in .env "
            "and/or add .pdf / .txt / .md files under data/."
        )

    provider = (llm_provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if provider == "ollama":
        if not _ollama_is_reachable():
            raise EnvironmentError(
                f"Ollama is not reachable at {OLLAMA_BASE_URL}. "
                "Start Ollama and pull a model, e.g. `ollama pull llama3.2`."
            )
        print(f"✅ LLM: Ollama ({OLLAMA_MODEL}) · Embeddings: OpenAI ({EMBED_MODEL})")
    else:
        print(f"✅ LLM: OpenAI ({OPENAI_LLM_MODEL}) · Embeddings: OpenAI ({EMBED_MODEL})")


def _ollama_is_reachable() -> bool:
    import requests

    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        res = requests.get(url, timeout=5)
        return res.status_code == 200
    except Exception:
        return False


def get_llm_info() -> dict:
    """Return metadata for the LLM used in the last build_pipeline call."""
    return dict(_active_llm_info)


def get_llm(provider: str | None = None, model: str | None = None, temperature: float = 0):
    """
    Create the chat LLM for RAG answers.

    LLM_PROVIDER=openai  → ChatOpenAI (OPENAI_LLM_MODEL, default gpt-4o)
    LLM_PROVIDER=ollama  → ChatOllama (OLLAMA_MODEL, default llama3.2)
    """
    provider = (provider or LLM_PROVIDER).lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        chosen = model or OLLAMA_MODEL
        llm = ChatOllama(
            model=chosen,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )
        _active_llm_info.update(
            {
                "provider": "ollama",
                "model": chosen,
                "base_url": OLLAMA_BASE_URL,
                "label": f"Ollama · {chosen}",
            }
        )
        return llm

    if provider != "openai":
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'openai' or 'ollama'.")

    chosen = model or OPENAI_LLM_MODEL
    llm = ChatOpenAI(
        model=chosen,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
    )
    _active_llm_info.update(
        {
            "provider": "openai",
            "model": chosen,
            "label": f"OpenAI · {chosen}",
        }
    )
    return llm

# ─── 1. Document Ingestion ────────────────────────────────────────────────────

def _format_notion_id(raw_id: str) -> str:
    """Normalize a 32-char Notion ID to UUID form for the API."""
    clean = raw_id.strip().replace("-", "")
    if len(clean) != 32:
        raise ValueError(f"Invalid Notion ID (expected 32 hex chars): {raw_id!r}")
    return f"{clean[:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:]}"


def load_notion_page() -> list[Document]:
    """Load a single Notion document/page by NOTION_PAGE_ID."""
    if not NOTION_API_KEY or not NOTION_PAGE_ID:
        return []

    from langchain_community.document_loaders.notiondb import (
        NotionDBLoader,
        PAGE_URL,
    )

    page_id = _format_notion_id(NOTION_PAGE_ID)
    loader = NotionDBLoader(
        integration_token=NOTION_API_KEY,
        database_id=page_id,  # required by ctor; unused for page fetch
    )
    page_data = loader._request(PAGE_URL.format(page_id=page_id))
    doc = loader.load_page(page_data)
    title = doc.metadata.get("title") or page_id
    doc.metadata["source"] = f"notion:{title}"
    print(f"✅ Loaded Notion page: {title}")
    return [doc]


def load_notion_database() -> list[Document]:
    """Load all rows from a Notion database when NOTION_DATABASE_ID is set."""
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        return []

    from langchain_community.document_loaders import NotionDBLoader

    db_id = _format_notion_id(NOTION_DATABASE_ID)
    loader = NotionDBLoader(
        integration_token=NOTION_API_KEY,
        database_id=db_id,
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", "notion")
    print(f"✅ Loaded {len(docs)} pages from Notion database")
    return docs


def load_notion_documents() -> list[Document]:
    """Load Notion page and/or database content based on .env settings."""
    docs: list[Document] = []
    docs.extend(load_notion_page())
    docs.extend(load_notion_database())
    return docs


def _load_local_files(data_dir: str) -> list[Document]:
    """Load PDF, TXT, and MD files from data_dir (if any exist)."""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    docs: list[Document] = []
    loaders = [
        ("**/*.pdf", PyPDFLoader, {}),
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.md", TextLoader, {"encoding": "utf-8"}),
    ]
    for glob_pattern, loader_cls, loader_kwargs in loaders:
        pattern = glob_pattern.removeprefix("**/")
        if not list(path.rglob(pattern)):
            continue
        loader = DirectoryLoader(
            data_dir,
            glob=glob_pattern,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            show_progress=True,
        )
        docs.extend(loader.load())

    if docs:
        print(f"✅ Loaded {len(docs)} file(s) from {data_dir}/")
    return docs


def load_documents(data_dir: str | None = None) -> list[Document]:
    """
    Load knowledge base: optional local files (data/) + Notion page/database.
    """
    data_dir = data_dir or os.getenv("DATA_DIR", "data")
    docs = _load_local_files(data_dir)
    docs.extend(load_notion_documents())

    if not docs:
        raise ValueError(
            f"No documents found. Add files to {data_dir}/ or set NOTION_PAGE_ID in .env."
        )

    print(f"✅ Loaded {len(docs)} raw document(s) total")
    return docs


# ─── 2. Chunking ──────────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    Semantic-aware recursive chunking.

    chunk_size=512   → sweet spot for embedding models
    chunk_overlap=64 → preserves context across boundaries

    Splits on: paragraph → sentence → word → char (in that order)
    so chunks stay semantically coherent wherever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Enrich metadata on every chunk
    for i, chunk in enumerate(chunks):
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_id"]   = hashlib.md5(
            f"{src}-{i}-{chunk.page_content[:40]}".encode()
        ).hexdigest()[:12]
        chunk.metadata["chunk_index"] = i
        chunk.metadata["char_count"]  = len(chunk.page_content)

    print(f"✅ Created {len(chunks)} chunks "
          f"(avg {sum(c.metadata['char_count'] for c in chunks)//len(chunks)} chars each)")
    return chunks


# ─── 3. Pinecone Vector Store ─────────────────────────────────────────────────

def init_pinecone() -> Pinecone:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


def clear_pinecone_index(pc: Pinecone | None = None) -> None:
    """Delete all vectors from the Pinecone index (keeps the index)."""
    pc = pc or init_pinecone()
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"ℹ  Index '{INDEX_NAME}' does not exist yet — nothing to clear")
        return

    stats = pc.Index(INDEX_NAME).describe_index_stats()
    count = stats.get("total_vector_count", 0)
    if count == 0:
        print(f"ℹ  Index '{INDEX_NAME}' is already empty")
        return

    pc.Index(INDEX_NAME).delete(delete_all=True)
    print(f"🗑️  Cleared {count} vectors from '{INDEX_NAME}'")


def create_index_if_not_exists(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"🔧 Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print(f"✅ Index '{INDEX_NAME}' ready")
    else:
        print(f"✅ Index '{INDEX_NAME}' already exists")


def get_vector_store(pc: Pinecone) -> PineconeVectorStore:
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    index = pc.Index(INDEX_NAME)
    return PineconeVectorStore(index=index, embedding=embeddings)


def ingest(chunks: list[Document], vector_store: PineconeVectorStore) -> None:
    """Batch-upsert chunks into Pinecone (100 per batch to stay within limits)."""
    batch_size = 100
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        vector_store.add_documents(batch)
        print(f"  Upserted {min(start + batch_size, total)}/{total} chunks...")
    print("✅ Ingestion complete")


# ─── 4. Retriever ─────────────────────────────────────────────────────────────

def build_vector_retriever(vector_store: PineconeVectorStore, k: int = 6):
    """
    MMR (Maximal Marginal Relevance) retriever over Pinecone.

    fetch_k=20 candidates, then MMR selects the best k with diversity.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.6},
    )


def build_hybrid_retriever(
    vector_store: PineconeVectorStore,
    chunks: list[Document],
    k: int = 6,
):
    """
    Hybrid retrieval: dense vectors (Pinecone MMR) + sparse BM25 keyword search.

    EnsembleRetriever merges ranked results from both retrievers.
    """
    vector_retriever = build_vector_retriever(vector_store, k=k)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )


# ─── 5. RAG Chain ─────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a research assistant answering questions from ML/AI papers and enterprise RAG documentation.

INSTRUCTIONS:
- Answer ONLY from the provided context below.
- If the context does not contain the answer, say "I don't have enough information to answer this."
- Always cite the source document(s) you used (from the metadata).
- Be concise. Use bullet points for lists and statistics.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")


def format_docs(docs: list[Document]) -> str:
    """Format retrieved docs into a single context string with source labels."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        src    = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        label  = f"[{i}] {Path(src).name}" + (f" (p.{page})" if page != "" else "")
        formatted.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever, llm):
    """
    LCEL RAG chain:
      question → retriever → format docs → prompt → LLM → string
    RunnablePassthrough keeps the original question flowing through.
    """
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ─── 6. Main Pipeline ─────────────────────────────────────────────────────────

def build_pipeline(
    data_dir: str | None = None,
    force_reingest: bool = False,
    clear_index: bool = False,
    llm_provider: str | None = None,
):
    """
    Full pipeline: load → chunk → embed → store → retriever → chain.

    force_reingest=True  → clear old vectors and re-embed everything.
    clear_index=True     → delete Pinecone vectors before ingest (auto with force_reingest).
    """
    validate_config(llm_provider=llm_provider)
    print("\n🚀 Building Production RAG Pipeline\n" + "=" * 40)

    pc = init_pinecone()
    create_index_if_not_exists(pc)
    vector_store = get_vector_store(pc)

    if force_reingest or clear_index:
        clear_pinecone_index(pc)

    print("\n📄 Loading documents...")
    docs   = load_documents(data_dir)
    chunks = chunk_documents(docs)

    index_stats = pc.Index(INDEX_NAME).describe_index_stats()
    vector_count = index_stats.get("total_vector_count", 0)

    if force_reingest or vector_count == 0:
        print("\n📥 Upserting embeddings to Pinecone...")
        ingest(chunks, vector_store)
    else:
        print(f"⚡ Skipping Pinecone upsert — {vector_count} vectors already in index")

    # Build chain (hybrid retriever always needs in-memory chunks for BM25)
    llm       = get_llm(provider=llm_provider)
    retriever = build_hybrid_retriever(vector_store, chunks)
    chain     = build_rag_chain(retriever, llm)

    info = get_llm_info()
    print(f"\n✅ RAG pipeline ready — answers via {info.get('label', 'LLM')}\n")
    return chain, retriever


def format_source_label(doc: Document) -> str:
    """Human-readable label for a retrieved chunk."""
    loc = doc_location(doc)
    if loc["page_display"] is not None:
        return f"{loc['filename']} (page {loc['page_display']})"
    return loc["filename"]


def doc_location(doc: Document) -> dict:
    """Structured location info for UI / citations."""
    src = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    page_display = None
    if page is not None and page != "":
        try:
            page_display = int(page) + 1  # PyPDFLoader uses 0-based pages
        except (ValueError, TypeError):
            page_display = page

    path = Path(src) if src != "unknown" else None
    return {
        "source": src,
        "filename": path.name if path else src,
        "filepath": str(path.resolve()) if path and path.exists() else None,
        "page": page,
        "page_display": page_display,
        "chunk_id": doc.metadata.get("chunk_id"),
        "chunk_index": doc.metadata.get("chunk_index"),
        "char_count": doc.metadata.get("char_count"),
        "is_notion": "notion" in src.lower(),
    }


def get_pdf_page_text(filepath: str, page_index: int) -> str:
    """Extract full text of one PDF page (0-based index)."""
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    if page_index < 0 or page_index >= len(reader.pages):
        return ""
    return reader.pages[page_index].extract_text() or ""


def ask(chain, retriever, question: str) -> tuple[str, list[Document]]:
    """Run RAG query and return (answer, retrieved_chunks)."""
    docs = retriever.invoke(question)
    answer = chain.invoke(question)
    return answer, docs


def query(chain, question: str, retriever=None) -> str:
    """Run a single query through the RAG chain."""
    print(f"\n❓ Question: {question}")
    print("-" * 40)
    if retriever is not None:
        answer, docs = ask(chain, retriever, question)
        print(f"💬 Answer:\n{answer}\n")
        if docs:
            print("📎 Sources:")
            seen = set()
            for doc in docs:
                label = format_source_label(doc)
                if label in seen:
                    continue
                seen.add(label)
                print(f"   • {label}")
        print()
        return answer
    answer = chain.invoke(question)
    print(f"💬 Answer:\n{answer}\n")
    return answer


if __name__ == "__main__":
    chain, _ = build_pipeline(force_reingest=True)

    for q in [
        "What ROI do companies report from RAG deployments?",
        "What percentage of Fortune 500 companies use RAG in 2026?",
        "What are the main obstacles to enterprise RAG adoption?",
        "what is difference between lora and Qlora in 2026?",
    ]:
        query(chain, q)
