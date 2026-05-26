"""
demo.py — End-to-end demo runner

Loads your knowledge base (data/*.pdf + Notion study),
clears old Pinecone vectors, re-ingests, runs queries and eval.

Run:
    python demo.py
"""

import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_pipeline import build_pipeline, query, validate_config
from rag_eval import RAGEvaluator, run_eval_suite, save_report_json

load_dotenv()

# Questions aligned with data/: ML papers + Notion RAG study
TEST_QUESTIONS = [
    "What is the main idea behind the Transformer architecture?",
    "How does BERT differ from left-to-right language models?",
    "What is LoRA and what problem does it solve?",
    "How does QLoRA reduce memory use compared to full fine-tuning?",
    "What prompting techniques improve LLM reasoning?",
    "What ROI do companies report from RAG deployments?",
    "What percentage of Fortune 500 companies have deployed RAG?",
    "What are the main obstacles to enterprise RAG adoption?",
]

DEMO_QUESTIONS = [
    "Explain self-attention in one paragraph.",
    "What is LoRA and why is it useful for fine-tuning?",
    "What ROI do companies report from RAG according to the 2026 study?",
]


def main():
    print("\n" + "🏭 " * 20)
    print("PRODUCTION RAG SYSTEM — DEMO")
    print("🏭 " * 20 + "\n")

    validate_config()

    # Clear old FAQ vectors and re-ingest current knowledge base
    chain, retriever = build_pipeline(force_reingest=True)

    print("\n" + "─" * 40)
    print("DEMO QUERIES")
    print("─" * 40)
    for q in DEMO_QUESTIONS:
        query(chain, q)

    print("\n" + "─" * 40)
    print("RUNNING EVALUATION SUITE")
    print("─" * 40)
    evaluator = RAGEvaluator()
    report = run_eval_suite(chain, retriever, evaluator, TEST_QUESTIONS)

    save_report_json(report, "eval_report.json")

    print("\n✅ Demo complete. Check eval_report.json for full results.")


if __name__ == "__main__":
    main()
