import sys
import io
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "utf-8") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# core/retrieval.py
"""
SCRIPT 2 — Retrieval Layer

Three search methods:
  1. Similarity search  — pure cosine similarity (fast, standard)
  2. MMR search         — diverse results (avoids redundant chunks)
  3. Hybrid search      — semantic + keyword BM25 combined (best quality)

Why three methods?
  Different queries benefit from different retrieval strategies.
  Similarity is fast. MMR avoids repetition. Hybrid catches
  exact scheme names that semantic search sometimes misses.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
HF_CACHE_DIR    = BASE_DIR / "data" / "hf_cache"

# ── Constants ─────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "schemeseeker"
DEFAULT_K       = 5   # number of chunks to retrieve


# ── Load vectorstore ──────────────────────────────────────────────
def load_vectorstore() -> Chroma:
    """
    Load ChromaDB from disk.
    Called once at app startup — reused for all queries.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
        cache_folder  = str(HF_CACHE_DIR / "embeddings"),
    )

    vectorstore = Chroma(
        collection_name    = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory  = str(VECTORSTORE_DIR),
    )

    count = vectorstore._collection.count()
    print(f"✅ ChromaDB loaded — {count} chunks available")
    return vectorstore


# ── Method 1: Similarity Search ───────────────────────────────────
def similarity_search(
    vectorstore: Chroma,
    query      : str,
    k          : int = DEFAULT_K,
) -> list[Document]:
    """
    Pure cosine similarity search.

    How it works:
      1. Embed the query → 384-dim vector
      2. Calculate cosine similarity with every stored vector
      3. Return top-k most similar chunks

    Best for: General queries, quick lookups
    Weakness: Can return redundant chunks (same info repeated)

    Args:
        vectorstore: ChromaDB instance
        query      : User's question
        k          : Number of results to return

    Returns:
        List of Document objects sorted by similarity
    """
    results = vectorstore.similarity_search(
        query = query,
        k     = k,
    )
    return results


# ── Method 2: MMR Search ──────────────────────────────────────────
def mmr_search(
    vectorstore : Chroma,
    query       : str,
    k           : int = DEFAULT_K,
    fetch_k     : int = 20,
    lambda_mult : float = 0.7,
) -> list[Document]:
    """
    Maximal Marginal Relevance (MMR) search.

    How it works:
      1. Fetch fetch_k=20 most similar chunks (wider net)
      2. Iteratively pick chunks that are:
         → Relevant to the query (similarity)
         → Different from already-picked chunks (diversity)
      3. Return k diverse chunks

    lambda_mult controls the tradeoff:
      → 1.0 = pure similarity (same as similarity_search)
      → 0.0 = pure diversity (maximally different results)
      → 0.7 = 70% similarity, 30% diversity (our default)

    Best for: Government schemes where many chunks repeat
              the same eligibility criteria — MMR ensures
              we get info about DIFFERENT schemes
    Weakness: Slightly slower than pure similarity

    Args:
        vectorstore : ChromaDB instance
        query       : User's question
        k           : Number of results to return
        fetch_k     : How many to fetch before MMR reranking
        lambda_mult : Similarity vs diversity tradeoff
    """
    results = vectorstore.max_marginal_relevance_search(
        query       = query,
        k           = k,
        fetch_k     = fetch_k,
        lambda_mult = lambda_mult,
    )
    return results


# ── Method 3: Hybrid Search ───────────────────────────────────────
def hybrid_search(
    vectorstore: Chroma,
    query      : str,
    k          : int = DEFAULT_K,
    alpha      : float = 0.7,
) -> list[Document]:
    """
    Hybrid search — semantic + BM25 keyword combined.

    How it works:
      1. Semantic score  = cosine similarity (meaning match)
      2. Keyword score   = BM25 TF-IDF (exact word match)
      3. Final score     = alpha * semantic + (1-alpha) * keyword

    Why hybrid?
    → Pure semantic misses exact scheme names
      "PM Kisan" might not semantically match well
      but BM25 catches the exact keywords
    → Pure keyword misses synonyms
      "farmer support" doesn't keyword-match "agricultural scheme"
      but semantic search finds it
    → Hybrid gets the best of both

    alpha=0.7 means 70% semantic, 30% keyword weight.

    Args:
        vectorstore: ChromaDB instance
        query      : User's question
        k          : Number of results to return
        alpha      : Weight of semantic vs keyword (0-1)
    """
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.retrievers import EnsembleRetriever
        
        

        # Get all documents from ChromaDB for BM25
        # BM25 needs the full corpus to calculate term frequencies
        all_docs = vectorstore.get()
        texts    = all_docs["documents"]
        metas    = all_docs["metadatas"]

        if not texts:
            # Fallback to MMR if no documents
            return mmr_search(vectorstore, query, k)

        # Build Document objects for BM25
        bm25_docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(texts, metas)
        ]

        # BM25 retriever — keyword based
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = k

        # Semantic retriever — meaning based
        semantic_retriever = vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": k},
        )

        # Ensemble — combines both with weights
        ensemble = EnsembleRetriever(
            retrievers = [semantic_retriever, bm25_retriever],
            weights    = [alpha, 1 - alpha],
        )

        results = ensemble.invoke(query)
        return results[:k]

    except Exception as e:
        # BM25 can fail if rank-bm25 not installed
        # Graceful fallback to MMR
        print(f"  ⚠️  Hybrid search failed ({e}), falling back to MMR")
        return mmr_search(vectorstore, query, k)


# ── Format results for display ────────────────────────────────────
def format_results(results: list[Document]) -> list[dict]:
    """
    Convert Document objects into clean dicts for display.

    Extracts:
    → preview   : first 300 chars of content
    → question  : original question from metadata
    → source    : dataset source
    → full_text : complete chunk content
    """
    formatted = []
    for doc in results:
        formatted.append({
            "preview"   : doc.page_content[:300],
            "full_text" : doc.page_content,
            "question"  : doc.metadata.get("question", ""),
            "source"    : doc.metadata.get("source", ""),
            "scheme"    : doc.metadata.get("scheme_hint", ""),
        })
    return formatted


# ── Get retriever object for chain.py ─────────────────────────────
def get_retriever(vectorstore: Chroma, method: str = "mmr"):
    """
    Return a LangChain retriever object.

    Used by chain.py to plug into the RAG chain.
    The retriever is a Runnable — it accepts a query
    string and returns a list of Documents.

    Args:
        vectorstore: ChromaDB instance
        method     : "similarity", "mmr"

    Returns:
        LangChain BaseRetriever object
    """
    if method == "mmr":
        return vectorstore.as_retriever(
            search_type   = "mmr",
            search_kwargs = {
                "k"          : 5,
                "fetch_k"    : 20,
                "lambda_mult": 0.7,
            }
        )
    else:
        return vectorstore.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": 5},
        )


# ── Test all three methods ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SCHEMESEEKER — Retrieval Test")
    print("=" * 60 + "\n")

    # Load vectorstore
    vs = load_vectorstore()
    print()

    # Test queries — English and Hindi
    test_queries = [
        "schemes for farmers with small land holdings",
        "health insurance scheme for poor families",
        "PM Kisan eligibility criteria",
        "किसान योजना",          # Hindi: farmer scheme
        "महिलाओं के लिए योजना", # Hindi: schemes for women
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Query: '{query}'\n")

        # Method 1: Similarity
        print("1️⃣  Similarity Search:")
        sim_results = similarity_search(vs, query, k=2)
        for r in sim_results:
            print(f"   → {r.page_content[:120]}...")

        # Method 2: MMR
        print("\n2️⃣  MMR Search (diverse):")
        mmr_results = mmr_search(vs, query, k=2)
        for r in mmr_results:
            print(f"   → {r.page_content[:120]}...")

        # Method 3: Hybrid
        print("\n3️⃣  Hybrid Search (semantic + keyword):")
        hyb_results = hybrid_search(vs, query, k=2)
        for r in hyb_results:
            print(f"   → {r.page_content[:120]}...")

    print("\n" + "=" * 60)
    print("  ✅ Retrieval working — all 3 methods tested")
    print("  Next → python core/chain.py")
    print("=" * 60 + "\n")