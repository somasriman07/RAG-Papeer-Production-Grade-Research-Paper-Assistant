import hashlib
import os
import uuid
from collections import defaultdict

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import CrossEncoder

load_dotenv()

# Detect best hardware acceleration device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_DIM    = 1024

# Parent chunks: larger, returned as final context for generation
PARENT_CHUNK_SIZE    = 600
PARENT_CHUNK_OVERLAP = 100

# Child chunks: small, indexed for precise retrieval
CHILD_CHUNK_SIZE     = 120
CHILD_CHUNK_OVERLAP  = 20

DENSE_CANDIDATES  = 15
BM25_CANDIDATES   = 15
RERANK_CANDIDATES = 15
FINAL_TOP_K       = 2
RRF_K             = 60

# ── Splitters ─────────────────────────────────────────────────────────────────
_parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_CHUNK_OVERLAP,
    add_start_index=True,
)
_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
    add_start_index=True,
)

# ── Embeddings ────────────────────────────────────────────────────────────────
base_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": DEVICE, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)
embedding_file_store = LocalFileStore("./cache/embeddings")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace="BAAI/bge-m3",
    query_embedding_cache=True,
    key_encoder="blake2b",
)

# ── In-memory query vector cache ──────────────────────────────────────────────
_query_vector_cache: dict[str, list[float]] = {}

def _get_query_vector(query: str) -> list[float]:
    key = hashlib.blake2b(query.encode(), digest_size=16).hexdigest()
    if key not in _query_vector_cache:
        _query_vector_cache[key] = embeddings.embed_query(query)
    return _query_vector_cache[key]

# ── Cross-encoder Reranker (lazy) ─────────────────────────────────────────────
_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-base",
            device=DEVICE,
            max_length=512,
        )
    return _reranker

# ── Qdrant Client ─────────────────────────────────────────────────────────────
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)

# ── Parent Document Store ─────────────────────────────────────────────────────
# In-memory store: {session_id: {parent_id: Document}}
# Stores full parent chunks so we can look them up after child retrieval.
_parent_store: dict[str, dict[str, Document]] = {}

# ── BM25 Store (on child chunks) ──────────────────────────────────────────────
# {session_id: (BM25Okapi, [child Documents])}
_bm25_store: dict[str, tuple[BM25Okapi, list[Document]]] = {}

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

def _build_bm25_index(docs: list[Document]) -> BM25Okapi:
    return BM25Okapi([_tokenize(doc.page_content) for doc in docs])

def _bm25_search(query: str, session_id: str, k: int) -> list[Document]:
    if session_id not in _bm25_store:
        return []
    index, docs = _bm25_store[session_id]
    scores      = index.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_indices if scores[i] > 0]

# ── RRF Fusion ────────────────────────────────────────────────────────────────
def _rrf_fuse(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = RRF_K,
) -> list[Document]:
    scores:  dict[str, float]    = defaultdict(float)
    doc_map: dict[str, Document] = {}
    for rank, doc in enumerate(dense_docs):
        key = hashlib.md5(doc.page_content.encode()).hexdigest()
        scores[key]  += 1.0 / (k + rank + 1)
        doc_map[key]  = doc
    for rank, doc in enumerate(sparse_docs):
        key = hashlib.md5(doc.page_content.encode()).hexdigest()
        scores[key]  += 1.0 / (k + rank + 1)
        doc_map[key]  = doc
    return [doc_map[key] for key in sorted(scores, key=lambda x: scores[x], reverse=True)]

# ── Cross-encoder Rerank ──────────────────────────────────────────────────────
def _rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    if not docs:
        return docs
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# ── Parent lookup ─────────────────────────────────────────────────────────────
def _get_parent_docs(
    child_docs: list[Document],
    session_id: str,
) -> list[Document]:
    """
    Given retrieved child chunks, return their unique parent documents.
    Deduplicates parents so the same parent isn't returned twice even if
    multiple child chunks map to it.
    """
    parents = _parent_store.get(session_id, {})
    seen_ids: set[str]      = set()
    result:   list[Document] = []

    for child in child_docs:
        parent_id = child.metadata.get("parent_id")
        if parent_id and parent_id not in seen_ids and parent_id in parents:
            seen_ids.add(parent_id)
            result.append(parents[parent_id])

    # Fallback: if no parent mapping found, return child docs as-is
    return result if result else child_docs

# ── Collection Helpers ────────────────────────────────────────────────────────
def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"

def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

# ── Public API ────────────────────────────────────────────────────────────────
def add_paper(docs: list[Document], session_id: str) -> None:
    """
    Parent Document Retrieval ingestion:

    1. Split docs into large parent chunks (800 chars) → stored in memory
    2. Split each parent into small child chunks (150 chars) → indexed in Qdrant + BM25
    3. Each child carries parent_id metadata pointing to its parent

    At search time: child chunks retrieved precisely → parent chunks returned
    for generation context. This gives precision (child) + completeness (parent).
    """
    if session_id not in _parent_store:
        _parent_store[session_id] = {}

    parent_chunks = _parent_splitter.split_documents(docs)
    child_chunks  = []

    for parent in parent_chunks:
        # Assign unique ID to each parent
        parent_id = str(uuid.uuid4())
        parent.metadata["parent_id"] = parent_id
        _parent_store[session_id][parent_id] = parent

        # Create child chunks from this parent, tagging each with parent_id
        children = _child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
        child_chunks.extend(children)

    # Index child chunks in Qdrant (dense) and BM25 (sparse)
    get_vectorstore(session_id).add_documents(child_chunks)

    existing_children = _bm25_store.get(session_id, (None, []))[1]
    all_children      = existing_children + child_chunks
    _bm25_store[session_id] = (_build_bm25_index(all_children), all_children)


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []
    seen:   set[str]  = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int = FINAL_TOP_K) -> list[Document]:
    """
    Full Parent Document Retrieval pipeline:

      Child Dense (Qdrant bge-m3) ─┐
                                    ├─ RRF → Rerank children → Fetch parents
      Child Sparse (BM25)         ─┘

    Why this fixes Contextual Recall:
    - Child chunks (150 chars) are sentence-level → precise retrieval hits
    - Parent chunks (800 chars) contain surrounding context → both En-De AND
      En-Fr benchmarks appear in the same parent even if only one child matched
    - No compression needed — child retrieval is already precise, parent provides
      the full coverage needed for Contextual Recall to score correctly
    """
    # Step 1: Retrieve child chunks from both dense and sparse
    dense_children  = get_vectorstore(session_id).similarity_search(
        query, k=DENSE_CANDIDATES
    )
    sparse_children = _bm25_search(query, session_id, k=BM25_CANDIDATES)

    # Step 2: RRF fusion of child results
    fused_children  = _rrf_fuse(dense_children, sparse_children)

    # Step 3: Rerank child chunks (small = faster, more precise scoring)
    reranked_children = _rerank(
        query, fused_children[:RERANK_CANDIDATES], top_k=k
    )

    # Step 4: Look up parent documents for the top reranked children
    parent_docs = _get_parent_docs(reranked_children, session_id)

    return parent_docs