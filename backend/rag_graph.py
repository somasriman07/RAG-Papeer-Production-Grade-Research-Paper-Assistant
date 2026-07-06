import hashlib
import logging
import os
import re
import sqlite3
import time
import warnings
from collections import defaultdict
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.models import ClaimVerificationResult, RelevancyDecision, RouterDecision
from backend.vector_store import search as vs_search

load_dotenv()

# ── Audit Logger ──────────────────────────────────────────────────────────────
# Writes structured logs to rag_audit.log for every node, routing decision,
# tool call, guardrail trigger, and rate-limit event. Completely separate from
# stdout so your terminal output stays clean.

logging.basicConfig(
    filename="rag_audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
audit = logging.getLogger("rag_audit")


def log_event(event: str, session_id: str, detail: str = "") -> None:
    """Central audit log call — all guardrail/routing/tool events go here."""
    safe_session = hashlib.sha256(session_id.encode()).hexdigest()[:12]
    audit.info(f"event={event} | session={safe_session} | {detail}")


# ── Rate & Abuse Prevention ───────────────────────────────────────────────────
# Two-layer defence:
#   1. Per-session request rate limiter (max N queries per time window).
#   2. Wallet-attack cap: hard limit on total LLM calls a session can trigger,
#      preventing prompt-injection loops that try to exhaust your API credits.

RATE_LIMIT_MAX_REQUESTS  = 10    # max queries per session per window
RATE_LIMIT_WINDOW_SEC    = 60    # rolling window in seconds
WALLET_MAX_LLM_CALLS     = 15    # hard cap on total LLM calls per session
                                  # (router + agent + relevancy + rewrite + generate = ~5-7 per query)

# In-memory stores — fine for single-process eval; swap for Redis in prod.
_session_request_log: dict[str, list[float]] = defaultdict(list)
_session_llm_call_count: dict[str, int]      = defaultdict(int)


def check_rate_limit(session_id: str) -> None:
    """Raises RuntimeError if session has exceeded request rate or wallet cap."""
    now = time.time()

    # 1. Wallet attack: total LLM call budget
    if _session_llm_call_count[session_id] >= WALLET_MAX_LLM_CALLS:
        log_event("WALLET_CAP_EXCEEDED", session_id,
                  f"llm_calls={_session_llm_call_count[session_id]}")
        raise RuntimeError(
            f"[SECURITY] Session exceeded maximum LLM call budget ({WALLET_MAX_LLM_CALLS}). "
            "This may indicate a prompt-injection loop. Session is halted."
        )

    # 2. Request rate: queries per rolling window
    window_start = now - RATE_LIMIT_WINDOW_SEC
    _session_request_log[session_id] = [
        t for t in _session_request_log[session_id] if t > window_start
    ]
    if len(_session_request_log[session_id]) >= RATE_LIMIT_MAX_REQUESTS:
        log_event("RATE_LIMIT_EXCEEDED", session_id,
                  f"requests_in_window={len(_session_request_log[session_id])}")
        raise RuntimeError(
            f"[SECURITY] Rate limit exceeded: more than {RATE_LIMIT_MAX_REQUESTS} "
            f"requests in {RATE_LIMIT_WINDOW_SEC}s. Please slow down."
        )

    _session_request_log[session_id].append(now)


def increment_llm_calls(session_id: str, node: str) -> None:
    """Track each LLM invocation against the wallet cap."""
    _session_llm_call_count[session_id] += 1
    log_event("LLM_CALL", session_id,
              f"node={node} | total_calls={_session_llm_call_count[session_id]}")


# ── Access Control ────────────────────────────────────────────────────────────
# Validates that session_ids are structurally legitimate (matching your known
# format) before any processing occurs. Rejects arbitrary/injected session IDs
# that could be used to cross-contaminate vector store collections.

_VALID_SESSION_RE = re.compile(
    r"^[a-zA-Z0-9_\-]{4,128}$"   # alphanumeric + underscore/dash, 4–128 chars
)


def validate_session_id(session_id: str) -> None:
    """Raises ValueError if session_id doesn't match the expected format."""
    if not _VALID_SESSION_RE.match(session_id):
        log_event("INVALID_SESSION_ID", session_id or "EMPTY",
                  f"rejected_value={repr(session_id[:60])}")
        raise ValueError(
            f"[SECURITY] Invalid session_id format: {repr(session_id[:60])}. "
            "Session IDs must be alphanumeric (4–128 chars)."
        )


# ── Output Sanitization ───────────────────────────────────────────────────────
# Strips prompt-injection patterns from LLM outputs before they reach the user
# or get stored in state. Also prevents the model from leaking system prompt
# instructions or internal tags through its answers.

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a?\s*(DAN|jailbreak|unrestricted)", re.IGNORECASE),
    re.compile(r"<\s*system\s*>.*?<\s*/\s*system\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[INST\].*?\[/INST\]", re.DOTALL),
    re.compile(r"<\|.*?\|>"),           # model-specific control tokens
    re.compile(r"#+\s*(system|prompt|instructions?)\s*:", re.IGNORECASE),
]

# Hard cap on output length — prevents the model generating multi-MB responses
# that could exhaust memory or cause downstream issues.
MAX_OUTPUT_CHARS = 4000


def sanitize_output(text: str, session_id: str) -> str:
    """Strip injection patterns and cap output length. Logs every triggered rule."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            log_event("OUTPUT_SANITIZED", session_id,
                      f"pattern={pattern.pattern[:60]}")
            text = pattern.sub("[REDACTED]", text)

    if len(text) > MAX_OUTPUT_CHARS:
        log_event("OUTPUT_TRUNCATED", session_id,
                  f"original_len={len(text)} | truncated_to={MAX_OUTPUT_CHARS}")
        text = text[:MAX_OUTPUT_CHARS] + "\n\n*[Response truncated for safety.]*"

    return text


# ── Context Length Management ─────────────────────────────────────────────────
# If retrieved context is large, trim it and inject a length-control instruction
# so the LLM produces a medium-length answer rather than a verbose dump that
# tanks your AnswerRelevancy score (confirmed issue from your eval results).

CONTEXT_CHAR_LIMIT   = 3000   # max chars fed to generate_answer_node as context
CONTEXT_CHUNK_LIMIT  = 3      # max chunks used for generation regardless of k


def build_context_and_prompt(docs: list[Document], query: str) -> str:
    trimmed = docs[:CONTEXT_CHUNK_LIMIT]
    context = "\n\n---\n\n".join(doc.page_content for doc in trimmed)
    if len(context) > CONTEXT_CHAR_LIMIT:
        context = context[:CONTEXT_CHAR_LIMIT]

    return (
        f"Answer the question in 2-3 sentences using ONLY the context below.\n"
        f"Do NOT add examples, metrics, or BLEU scores unless the question asks for them.\n"
        f"Do NOT make claims not explicitly stated in the context.\n"
        f"If the context is ambiguous, say so rather than guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )


# ── LLM Clients ──────────────────────────────────────────────────────────────

from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1024,
    timeout=60,
    max_retries=5,
)




# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Classify the user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for TWO types of questions:\n"
        "    (a) Questions about the content of uploaded research papers "
        "(e.g. methods, results, conclusions, authors).\n"
        "    (b) Questions that require live or current information that cannot be "
        "answered from general knowledge alone — such as current events, today's weather, "
        "live prices, recent news, or anything where the answer changes over time "
        "(e.g. 'Who is the current president?', 'What is the price of gold today?', "
        "'What is the weather in Delhi?').\n"
        "  verify_claim — The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer — A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Who invented the transformer?', "
        "'Explain backpropagation.').\n\n"
        "IMPORTANT: If the query contains phrases like 'as per the report', "
        "'in the knowledge base', 'according to the paper', or similar — "
        "ALWAYS classify as retrieve, never direct_answer."
        "Return ONLY a JSON object like: {{\"route\": \"retrieve\"}}"
    ),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | llm.with_structured_output(
    RouterDecision, method="json_mode"
)
import re

# PII patterns — add near your other constants at the top
_PII_PATTERNS = [
    (re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'), "PAN number"),
    (re.compile(r'\b[2-9]{1}[0-9]{11}\b'), "Aadhaar number"),
    (re.compile(r'\b[0-9]{10}\b'), "phone number"),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "email address"),
    (re.compile(r'\b(?:\d[ -]?){13,16}\b'), "credit/debit card number"),
    (re.compile(r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b'), "IFSC/bank account"),
    (re.compile(r'\b[A-Z]{1}[0-9]{7}\b'), "passport number"),
]

def _detect_pii(text: str) -> list[str]:
    """Return list of PII type names found in text."""
    found = []
    for pattern, label in _PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
    return found

def _redact_pii(text: str) -> str:
    """Redact detected PII from text before logging."""
    for pattern, label in _PII_PATTERNS:
        text = pattern.sub(f"[REDACTED {label.upper()}]", text)
    return text


def router_node(state: RAGState) -> dict:
    session_id = state["session_id"]
    validate_session_id(session_id)
    check_rate_limit(session_id)

    query = state["messages"][-1].content

    # ── PII Guardrail ──────────────────────────────────────────
    pii_found = _detect_pii(query)
    if pii_found:
        pii_types = ", ".join(pii_found)
        log_event("PII_DETECTED", session_id,
                  f"types={pii_types} | query={_redact_pii(query)[:80]}")
        safe_answer = (
            f"⚠️ I noticed your message may contain sensitive personal information "
            f"({pii_types}). For your safety, I can't process or store this information. "
            f"Please rephrase your question without sharing personal details."
        )
        return {
            "route": "direct_answer",
            "answer": safe_answer,
        }
    # ──────────────────────────────────────────────────────────

    increment_llm_calls(session_id, "router")
    log_event("ROUTER_NODE", session_id, f"query_len={len(query)}")
    decision: RouterDecision = router_chain.invoke({"query": query})
    log_event("ROUTE_DECISION", session_id, f"route={decision.route}")
    return {"route": decision.route}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(default=3, ge=1, le=5, description="Number of chunks to retrieve")


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=5, description="Number of web results to return")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    log_event("TOOL_CALL", session_id, f"tool=retrieve_from_vectorstore | k={k}")
    docs = vs_search(query=query, session_id=session_id, k=k)
    if not docs:
        log_event("RETRIEVAL_EMPTY", session_id, f"query={query[:80]}")
        return [ToolMessage(content="No relevant documents found in the vector store.",
                            tool_call_id=tool_call_id)]
    log_event("RETRIEVAL_SUCCESS", session_id, f"chunks_found={len(docs)}")
    summary = f"Retrieved {len(docs)} chunk(s) from the vector store."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    log_event("TOOL_CALL", "web_search_tool",
              f"tool=web_search | query={optimized_query[:80]}")
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(optimized_query, max_results=max_results)
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ]
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]


# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm   = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
base_tool_node  = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   - query: the semantic search query\n"
    "   - k: how many chunks to retrieve (1–5; use 2–3 for specific questions)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   - optimized_query: concise keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1–5)\n\n"
    "Choose the right source:\n"
    "- Questions about uploaded papers → retrieve_from_vectorstore\n"
    "- Current events or supplementary info → web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
)


# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true.\n\n"
    "You MUST respond with a JSON object containing exactly two fields:\n"
    "{\"is_relevant\": true/false, \"reason\": \"one sentence explanation\"}"

)
relevancy_llm = llm.with_structured_output(
    RelevancyDecision, method="json_mode"
)


QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: RAGState) -> dict:
    session_id = state["session_id"]
    current_attempts = state.get("retrieval_attempts", 0)
    increment_llm_calls(session_id, "agent_node")
    log_event("AGENT_NODE", session_id, f"retrieval_attempts={current_attempts}")

    lm = llm if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm
    messages = [{"role": "system", "content": RETRIEVE_SYSTEM}] + state["messages"]
    response = lm.invoke(messages)
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


def relevancy_check_node(state: RAGState) -> dict:
    session_id = state["session_id"]
    query = state["query"]
    docs  = state.get("retrieved_docs") or []
    increment_llm_calls(session_id, "relevancy_check")

    doc_snippets = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs[:3])
    if not doc_snippets:
        log_event("RELEVANCY_SKIP", session_id, "no_docs_retrieved")
        return {"is_relevant": False}

    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user",   "content": prompt},
    ])
    log_event("RELEVANCY_DECISION", session_id, f"is_relevant={decision.is_relevant}")
    return {"is_relevant": decision.is_relevant}


def query_rewrite_node(state: RAGState) -> dict:
    session_id    = state["session_id"]
    original_query = state["query"]
    rewrite_count  = state.get("rewrite_count", 0)
    increment_llm_calls(session_id, "query_rewrite")
    log_event("QUERY_REWRITE", session_id, f"rewrite_count={rewrite_count}")

    response  = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user",   "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = sanitize_output(response.content.strip(), session_id)
    log_event("QUERY_REWRITTEN", session_id, f"new_query={rewritten[:80]}")
    return {
        "messages":           [HumanMessage(content=rewritten)],
        "query":              rewritten,
        "retrieved_docs":     [],
        "retrieval_attempts": 0,
        "rewrite_count":      rewrite_count + 1,
        "is_relevant":        None,
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

verification_llm = llm.with_structured_output(
    ClaimVerificationResult, method="json_mode"
)


def verify_claim_node(state: RAGState) -> dict:
    session_id = state["session_id"]
    claim      = state["messages"][-1].content
    increment_llm_calls(session_id, "verify_claim")
    log_event("VERIFY_CLAIM", session_id, f"claim_len={len(claim)}")

    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    general_results = tavily_client.search(
        f"recent research superseding: {claim[:200]}", max_results=5,
    ).get("results", [])

    arxiv_results = tavily_client.search(
        f"site:arxiv.org {claim[:200]}", max_results=5,
    ).get("results", [])

    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )
    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    context = "\n".join(lines)
    prompt  = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict":      result.verdict_summary,
        "claim_source":       papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def generate_answer_node(state: RAGState) -> dict:
    session_id = state["session_id"]

    # If PII guardrail already set the answer, return it directly
    if state.get("answer") and state.get("route") == "direct_answer":
        existing = state["answer"]
        if "sensitive personal information" in existing:
            return {"answer": existing,
                    "messages": [AIMessage(content=existing)]}
    route      = state.get("route")
    query      = state["query"]
    increment_llm_calls(session_id, "generate_answer")
    log_event("GENERATE_ANSWER", session_id, f"route={route}")

    if route == "retrieve":
        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1:
            answer = (
                "I wasn't able to find relevant information in the uploaded papers "
                "to answer your question. You may want to rephrase your question "
                "or upload additional papers."
            )
        else:
            docs = state.get("retrieved_docs") or []
            if not docs:
                answer = "I don't know the answer."
            else:
                # Context length management: trim + inject medium-length instruction
                prompt = build_context_and_prompt(docs, query)
                answer = llm.invoke([{"role": "user", "content": prompt}]).content

    elif route == "verify_claim":
        verdict    = state.get("claim_verdict", "")
        papers     = state.get("superseding_papers") or []
        claim_text = state["query"]
        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:  # direct_answer
        prompt = (
            "Answer the question directly in 2–4 sentences. "
            "Be concise and include only what's directly relevant.\n\n"
            f"Question: {query}"
        )
        answer = llm.invoke([{"role": "user", "content": prompt}]).content

    # Output sanitization — applied to every answer before it leaves the graph
    answer = sanitize_output(answer, session_id)
    log_event("ANSWER_GENERATED", session_id, f"answer_len={len(answer)}")
    return {"answer": answer, "messages": [AIMessage(content=answer)]}


# ── Graph ─────────────────────────────────────────────────────────────────────

MAX_RETRIEVAL_ATTEMPTS = 3


def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    tc = tools_condition(state)
    if tc == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


def build_graph(db_path: str = "checkpoints.db"):
    conn        = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("router",          router_node)
    graph.add_node("agent_node",      agent_node)
    graph.add_node("retrieval",       base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite",   query_rewrite_node)
    graph.add_node("verify_claim",    verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router", route_query,
        {"retrieve": "agent_node", "verify_claim": "verify_claim",
         "direct_answer": "generate_answer"},
    )
    graph.add_conditional_edges(
        "agent_node", agent_routing,
        {"retrieval": "retrieval", "relevancy_check": "relevancy_check",
         "generate_answer": "generate_answer"},
    )
    graph.add_edge("retrieval", "agent_node")
    graph.add_conditional_edges(
        "relevancy_check", after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite",   "agent_node")
    graph.add_edge("verify_claim",    "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)