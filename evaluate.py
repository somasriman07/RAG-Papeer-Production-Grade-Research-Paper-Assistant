import json
import os
import sys
from pathlib import Path
from uuid import uuid4

# Set a User Agent to resolve the Langchain/Tavily warning
os.environ["USER_AGENT"] = "ResearchAssistantEval/1.0"

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

load_dotenv()

PDF_PATH            = "documents/attention_all_you_need.pdf"
GOLDENS_FILE        = Path("goldens.json")
MAX_CONTEXTS        = 5
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7



# ---------------------------------------------------------------------------
# Custom DeepEval LLM wrapper supporting both Sync and True Async execution
# ---------------------------------------------------------------------------
from groq import Groq
import time

class GroqEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        super().__init__()

    def load_model(self):
        return self.client

    

    def _call(self, prompt: str, schema=None) -> str:
        messages = [
            {"role": "system", "content": "Respond with valid JSON only, no extra text."},
            {"role": "user", "content": prompt},
        ]
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 512,  # reduce from 1024 to save TPM
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = (attempt + 1) * 10  # 10s, 20s, 30s, 40s, 50s
                    print(f"[RATE LIMIT] Waiting {wait}s before retry {attempt + 1}/5...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded on Groq rate limit")
    def generate(self, prompt: str, schema=None):
        raw = self._call(prompt, schema)
        if schema is not None:
            return schema.model_validate_json(raw)
        return raw

    async def a_generate(self, prompt: str, schema=None):
        import asyncio
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


def generate_goldens() -> list[dict]:
    eval_model = GroqEvalModel()
    synthesizer = Synthesizer(model=eval_model)
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs


def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


def run_rag_query(graph, query: str, session_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": str(session_id)}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context


def main() -> None:
    pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens()

    docs = load_document(PDF_PATH)
    graph = build_graph(db_path="eval_checkpoints.db")

    eval_model = GroqEvalModel()
    
    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=eval_model),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=eval_model),
        ContextualRelevancyMetric(threshold=0.24, model=eval_model),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=eval_model),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=eval_model),
    ]

    test_cases = []
    for pair in pairs:
        session_id = f"evaluation_session_{uuid4()}"
        add_paper(docs, session_id)
        from backend.vector_store import search as vs_search
        debug_check = vs_search(query="transformer attention", session_id=session_id, k=3)
        print(f"[DEBUG] session={session_id} ingested_chunks_found={len(debug_check)}")
        if debug_check:
            print(f"[DEBUG] sample chunk: {debug_check[0].page_content[:150]}")

        query = pair["input"] + " as per the report in knowledge base"
        answer, retrieval_context = run_rag_query(graph, query, session_id)
        test_cases.append(
            LLMTestCase(
                input=pair["input"],
                actual_output=answer,
                expected_output=pair["expected_output"],
                retrieval_context=retrieval_context,
            )
        )

    # Parallel execution works seamlessly now that non-blocking coroutines are implemented
    results = evaluate(
    test_cases,
    metrics,
    async_config=AsyncConfig(max_concurrent=1, throttle_value=20),

)

    summary = []
    for test_result in results.test_results:
        summary.append({
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "success": test_result.success,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.success,
                    "reason": m.reason,
                }
                for m in test_result.metrics_data
            ],
        })

    results_path = Path("eval_results.json")
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {results_path}.")


if __name__ == "__main__":
    main()