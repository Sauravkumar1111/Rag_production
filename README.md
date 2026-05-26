# RAG Knowledge Base

A Streamlit chat app for asking questions over a local or Notion-backed knowledge base. The pipeline uses LangChain, OpenAI embeddings, Pinecone vector search, BM25 keyword retrieval, and either OpenAI or Ollama for answer generation.

## Features

- Chat UI built with Streamlit.
- Local document ingestion from `data/`.
- Optional Notion page or database ingestion.
- Hybrid retrieval using Pinecone dense search and BM25 keyword search.
- Source details for answers, including document name, PDF page, chunk ID, and retrieved passage.
- Optional re-ingestion from the Streamlit sidebar.

## Project Structure

```text
.
+-- app.py              # Streamlit app
+-- rag_pipeline.py     # RAG ingestion, retrieval, and answer pipeline
+-- requirements.txt    # Python dependencies
+-- data/               # Add local PDF, TXT, or MD files here
`-- README.md
```
## pipeline Structure
User question
    |
    v
Streamlit app
    |
    v
Hybrid retriever
    |--------------------|
    v                    v
Dense vector search      BM25 keyword search
Pinecone + MMR           In-memory chunks
    |--------------------|
    v
Merged context
    |
    v
Prompt + LLM
    |
    v
Answer + citations + source chunks

## Requirements

- Python 3.11+
- OpenAI API key
- Pinecone API key
- Optional: Notion integration token and page/database ID
- Optional: Ollama for local answer generation

## Setup

Create a virtual environment:

```powershell
py -3.11 -m venv venv
```

Activate the virtual environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If you use Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=rag-production-2026

# Local files
DATA_DIR=data

# Answer model: openai or ollama
LLM_PROVIDER=openai
OPENAI_LLM_MODEL=gpt-4o

# Optional Notion source
NOTION_API_KEY=your_notion_api_key
NOTION_PAGE_ID=your_notion_page_id
NOTION_DATABASE_ID=your_notion_database_id

# Optional Ollama settings
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434
```

`OPENAI_API_KEY` and `PINECONE_API_KEY` are required. Add documents under `data/` and/or configure Notion with `NOTION_PAGE_ID` or `NOTION_DATABASE_ID`.

## Add Documents

Place supported files in the `data/` folder:

```text
data/
+-- paper.pdf
+-- notes.txt
`-- guide.md
```

The app loads `.pdf`, `.txt`, and `.md` files recursively from `data/`.

## Run the App

Start the Streamlit app:

```powershell
streamlit run app.py
```

On the first run, the pipeline creates the Pinecone index if needed, loads documents, chunks them, embeds them, and stores vectors in Pinecone. Use the sidebar button in the app to re-ingest all documents after changing your source files.

## Optional: Use Ollama

Install and start Ollama, then pull a model:

```powershell
ollama pull llama3.2
```

Set this in `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434
```

Embeddings still use OpenAI, even when the answer model is Ollama.

## Run the Pipeline Script

You can also run the pipeline directly:

```powershell
python rag_pipeline.py
```

This runs a forced re-ingestion and asks a few sample questions from the script.
