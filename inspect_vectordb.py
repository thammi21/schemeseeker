# inspect_vectordb.py
"""
Inspect what's actually stored inside ChromaDB.
Run this to understand the internal structure.
"""
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

BASE_DIR        = Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
HF_CACHE_DIR    = BASE_DIR / "data" / "hf_cache"

print("Loading ChromaDB...\n")

embeddings = HuggingFaceEmbeddings(
    model_name    = "paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True},
    cache_folder  = str(HF_CACHE_DIR / "embeddings"),
)

vs = Chroma(
    collection_name    = "schemeseeker",
    embedding_function = embeddings,
    persist_directory  = str(VECTORSTORE_DIR),
)

# ── 1. How many chunks are stored? ───────────────────────────────
count = vs._collection.count()
print(f"Total chunks stored: {count}")

# ── 2. Peek at raw stored data ────────────────────────────────────
print("\n" + "=" * 60)
print("RAW STORED DATA — first 3 chunks:")
print("=" * 60)

raw = vs._collection.get(
    limit  = 3,
    include= ["documents", "metadatas", "embeddings"]
)

for i in range(3):
    print(f"\n--- Chunk {i+1} ---")
    print(f"TEXT    : {raw['documents'][i][:200]}...")
    print(f"METADATA: {json.dumps(raw['metadatas'][i], indent=2)}")
    print(f"VECTOR  : {raw['embeddings'][i][:8]}... ({len(raw['embeddings'][i])} dimensions)")

# ── 3. What does a query vector look like? ────────────────────────
print("\n" + "=" * 60)
print("QUERY EMBEDDING — what happens when you search:")
print("=" * 60)

query = "scheme for farmers"
query_vector = embeddings.embed_query(query)
print(f"\nQuery  : '{query}'")
print(f"Vector : {query_vector[:8]}... ({len(query_vector)} dimensions)")
print(f"\nThis vector is compared against all {count} stored vectors.")
print("ChromaDB finds the closest ones using cosine similarity.")

# ── 4. Show similarity scores ─────────────────────────────────────
print("\n" + "=" * 60)
print("SIMILARITY SCORES — actual cosine similarity values:")
print("=" * 60)

results_with_scores = vs.similarity_search_with_score(
    "scheme for farmers in India",
    k=5
)

print(f"\nQuery: 'scheme for farmers in India'\n")
for i, (doc, score) in enumerate(results_with_scores, 1):
    # ChromaDB returns L2 distance — lower = more similar
    # Convert to similarity: similarity ≈ 1 - (score/2)
    similarity = 1 - (score / 2)
    bar = "█" * int(similarity * 20)
    print(f"Rank {i}: [{bar:<20}] similarity={similarity:.3f}")
    print(f"        {doc.page_content[:100]}...")
    print()

# ── 5. Is Hindi stored? ───────────────────────────────────────────
print("=" * 60)
print("BILINGUAL CHECK — is Hindi data in the store?")
print("=" * 60)

# Search for Hindi content directly
hindi_results = vs.similarity_search("किसान योजना महिला", k=3)
print(f"\nHindi query results: {len(hindi_results)} found")
for doc in hindi_results:
    print(f"  → {doc.page_content[:150]}...")

# Check if any stored text contains Hindi characters
print("\nScanning stored text for Hindi characters...")
sample = vs._collection.get(limit=200, include=["documents"])
hindi_count = sum(
    1 for doc in sample["documents"]
    if any('\u0900' <= c <= '\u097F' for c in doc)
    # Unicode range for Devanagari script (Hindi)
)
print(f"Hindi chunks found in first 200: {hindi_count}")

if hindi_count == 0:
    print("\nℹ️  No Hindi text in stored chunks — but Hindi QUERIES work!")
    print("   Here's why:")
    print("   The dataset is English text about Indian schemes.")
    print("   The multilingual model maps Hindi queries to the")
    print("   same vector space as English text.")
    print("   So 'किसान योजना' → similar vector to 'farmer scheme'")
    print("   → retrieves the same English documents. ✅")

# ── 6. Prove multilingual mapping works ──────────────────────────
print("\n" + "=" * 60)
print("MULTILINGUAL PROOF — same meaning = similar vector:")
print("=" * 60)

import numpy as np

english  = "scheme for farmers"
hindi    = "किसान के लिए योजना"
unrelated = "cricket match score"

vecs = embeddings.embed_documents([english, hindi, unrelated])
e_vec, h_vec, u_vec = vecs

# Cosine similarity (vectors are normalized so dot product = cosine sim)
e_h_sim = float(np.dot(e_vec, h_vec))
e_u_sim = float(np.dot(e_vec, u_vec))

print(f"\n'{english}'")
print(f"  vs '{hindi}'")
print(f"  Similarity: {e_h_sim:.4f} ← high (same meaning)")
print(f"\n'{english}'")
print(f"  vs '{unrelated}'")
print(f"  Similarity: {e_u_sim:.4f} ← low (different meaning)")
print(f"\nThe multilingual model maps Hindi and English")
print(f"descriptions of the same concept to similar vectors.")
print(f"This is why Hindi queries find English documents.")

print("\n✅ Inspection complete!")