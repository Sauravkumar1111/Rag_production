"""
RAG Evaluation — Offline metrics without a paid eval framework.

Metrics:
  • Faithfulness   — does the answer stay within the retrieved context?
  • Relevance      — does the retrieved context actually match the question?
  • Answer quality — judge the final answer with a second LLM call (LLM-as-judge)
  • Latency        — end-to-end response time tracking
"""

import os
import time
import json
import statistics
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    question:     str
    answer:       str
    contexts:     list[str]
    faithfulness: float          # 0.0 – 1.0
    relevance:    float          # 0.0 – 1.0
    quality:      float          # 0.0 – 1.0  (LLM-as-judge)
    latency_ms:   float
    passed:       bool = field(init=False)

    def __post_init__(self):
        self.passed = (
            self.faithfulness >= 0.7
            and self.relevance  >= 0.7
            and self.quality    >= 0.7
        )


# ─── LLM-as-judge prompts ─────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation judge. Rate FAITHFULNESS: does the answer contain ONLY
information that is present in the context? Penalise any claim not in the context.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object — no markdown, no explanation:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
""")

RELEVANCE_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation judge. Rate CONTEXT RELEVANCE: how well do the retrieved
passages answer the question? A score of 1.0 means every passage is directly useful.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Respond with ONLY a JSON object — no markdown, no explanation:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
""")

QUALITY_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation judge. Rate overall ANSWER QUALITY for this question.
Consider: accuracy, completeness, clarity, and appropriate use of citations.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object — no markdown, no explanation:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
""")


# ─── Evaluator ────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Runs three LLM-as-judge evaluations per query.
    Uses gpt-4o-mini for cost efficiency on eval calls.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        judge_model: str = "gpt-4o-mini",
    ):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for evaluation")
        self.llm = ChatOpenAI(
            model=judge_model,
            temperature=0,
            openai_api_key=api_key,
        )
        self._parser = StrOutputParser()

    def _judge(self, prompt: ChatPromptTemplate, **kwargs) -> tuple[float, str]:
        chain  = prompt | self.llm | self._parser
        raw    = chain.invoke(kwargs)
        # Strip accidental markdown fences
        clean  = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(clean)
        return float(parsed["score"]), parsed.get("reason", "")

    def evaluate_single(
        self,
        question: str,
        answer: str,
        docs: list[Document],
        latency_ms: float,
    ) -> EvalResult:
        contexts    = [d.page_content for d in docs]
        context_str = "\n\n---\n\n".join(contexts)

        faith_score, faith_reason = self._judge(
            FAITHFULNESS_PROMPT, context=context_str, answer=answer
        )
        rel_score, rel_reason = self._judge(
            RELEVANCE_PROMPT, question=question, context=context_str
        )
        qual_score, qual_reason = self._judge(
            QUALITY_PROMPT, question=question, answer=answer
        )

        result = EvalResult(
            question=question,
            answer=answer,
            contexts=contexts,
            faithfulness=faith_score,
            relevance=rel_score,
            quality=qual_score,
            latency_ms=latency_ms,
        )

        self._print_result(result, faith_reason, rel_reason, qual_reason)
        return result

    def _print_result(self, r: EvalResult, fr: str, rr: str, qr: str) -> None:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"\n{'='*50}")
        print(f"EVAL {status} — {r.question[:60]}")
        print(f"  Faithfulness : {r.faithfulness:.2f}  → {fr}")
        print(f"  Relevance    : {r.relevance:.2f}  → {rr}")
        print(f"  Quality      : {r.quality:.2f}  → {qr}")
        print(f"  Latency      : {r.latency_ms:.0f} ms")


# ─── Batch evaluation ─────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    results:          list[EvalResult]
    avg_faithfulness: float
    avg_relevance:    float
    avg_quality:      float
    avg_latency_ms:   float
    pass_rate:        float


def run_eval_suite(
    chain,
    retriever,
    evaluator: RAGEvaluator,
    test_questions: list[str],
) -> EvalReport:
    """
    Run the full evaluation suite over a list of test questions.
    Measures latency around the chain.invoke call.
    """
    results = []

    for question in test_questions:
        # Retrieve docs separately so we can pass them to the evaluator
        docs = retriever.invoke(question)

        start = time.perf_counter()
        answer = chain.invoke(question)
        latency_ms = (time.perf_counter() - start) * 1000

        result = evaluator.evaluate_single(question, answer, docs, latency_ms)
        results.append(result)

    report = EvalReport(
        results=results,
        avg_faithfulness=statistics.mean(r.faithfulness for r in results),
        avg_relevance=statistics.mean(r.relevance    for r in results),
        avg_quality=statistics.mean(r.quality       for r in results),
        avg_latency_ms=statistics.mean(r.latency_ms   for r in results),
        pass_rate=sum(r.passed for r in results) / len(results),
    )

    print_report(report)
    return report


def print_report(report: EvalReport) -> None:
    print("\n" + "=" * 50)
    print("📊 EVALUATION REPORT")
    print("=" * 50)
    print(f"  Questions evaluated : {len(report.results)}")
    print(f"  Pass rate           : {report.pass_rate*100:.1f}%")
    print(f"  Avg faithfulness    : {report.avg_faithfulness:.2f}")
    print(f"  Avg relevance       : {report.avg_relevance:.2f}")
    print(f"  Avg quality         : {report.avg_quality:.2f}")
    print(f"  Avg latency         : {report.avg_latency_ms:.0f} ms")
    print("=" * 50)

    failed = [r for r in report.results if not r.passed]
    if failed:
        print(f"\n⚠  {len(failed)} question(s) FAILED threshold (≥0.70 on all metrics):")
        for r in failed:
            print(f"   - {r.question[:70]}")


def save_report_json(report: EvalReport, path: str = "eval_report.json") -> None:
    data = {
        "summary": {
            "pass_rate":        round(report.pass_rate, 4),
            "avg_faithfulness": round(report.avg_faithfulness, 4),
            "avg_relevance":    round(report.avg_relevance, 4),
            "avg_quality":      round(report.avg_quality, 4),
            "avg_latency_ms":   round(report.avg_latency_ms, 1),
        },
        "results": [asdict(r) for r in report.results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 Report saved to {path}")
