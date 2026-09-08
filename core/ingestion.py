import sys
import io
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "utf-8") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# core/ingestion.py
"""
SCRIPT 1 — Data Ingestion Pipeline

Dataset: satyajitdas/bharatschemes-v1
  - 2029 Q&A pairs about Indian government schemes
  - Covers Central + State schemes, NGOs, CSR programs
  - Persona-based queries already included
  - ChatML format: system + user + assistant messages

What this script does:
  1. Downloads dataset from Hugging Face
  2. Extracts Q&A pairs as text documents
  3. Chunks into smaller pieces
  4. Embeds using multilingual model
  5. Stores in ChromaDB
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
HF_CACHE_DIR    = DATA_DIR / "hf_cache"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

DATA_DIR.mkdir(exist_ok=True)
HF_CACHE_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE       = 800   # increased from 500 so Q&A pairs stay whole more often
CHUNK_OVERLAP    = 100   # increased from 50
COLLECTION_NAME  = "schemeseeker"
HF_DATASET_NAME  = "satyajitdas/bharatschemes-v1"


# ── Step 1: Load dataset ──────────────────────────────────────────
def load_scheme_dataset() -> list:
    """
    Load BharatSchemes Q&A dataset from Hugging Face.

    Dataset format — each record has:
    {
      "messages": [
        {"role": "system",    "content": "You are BharatSchemes Assistant..."},
        {"role": "user",      "content": "What is PM-KISAN?"},
        {"role": "assistant", "content": "PM-KISAN is..."}
      ]
    }

    We extract user question + assistant answer pairs.
    """
    print("📥 Loading BharatSchemes dataset from Hugging Face...")
    print(f"   Dataset: {HF_DATASET_NAME}")
    print("   First run downloads ~15MB — cached after\n")

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            HF_DATASET_NAME,
            cache_dir=str(HF_CACHE_DIR)
        )

        # Combine train + validation splits
        all_records = []
        for split in dataset.keys():
            all_records.extend(list(dataset[split]))

        print(f"✅ Loaded {len(all_records)} Q&A pairs\n")
        return all_records

    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        raise


# ── Text cleaning ──────────────────────────────────────────────────
def clean_mojibake(text: str) -> str:
    """
    Fix common mojibake patterns from UTF-8 text misread as
    Windows-1252 during dataset creation.

    Each key below is the exact character sequence produced by
    taking a UTF-8-encoded character and decoding those bytes as
    cp1252 — derived by encoding each target character to UTF-8
    and decoding the resulting bytes as cp1252, not by eyeballing
    copy-pasted text (which silently mangles these multi-byte
    sequences and is why a hand-typed version of this table is
    unreliable).
    """
    if not text:
        return ""

    # UTF-8 BOM misread as cp1252 (EF BB BF -> ï»¿), and a literal BOM char
    text = text.replace("﻿", "")
    text = text.replace("ï»¿", "")

    replacements = {
        "â€™": "'",      # â€™ -> ' (U+2019 right single quote)
        "â€œ": '"',      # â€œ -> " (U+201C left double quote)
        "â€": '"',      # â€\x9d -> " (U+201D right double quote)
        "â€”": "—", # â€" -> — (em dash)
        "â€“": "–", # â€" -> – (en dash)
        "â€¦": "...",    # â€¦ -> ellipsis
        "â‚¹": "₹", # â‚¹ -> ₹ (rupee sign)
        "Â "      : " ",      # Â  (nbsp) -> space
        "Â·"      : "·", # Â· -> ·
        "Ã©"      : "é", # Ã© -> é
        "â"            : "",       # stray leftover â byte
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove remaining non-printable characters, but keep
    # Hindi/Devanagari (U+0900-U+097F) and Arabic (U+0600-U+06FF)
    cleaned = []
    for char in text:
        cp = ord(char)
        if (
            cp >= 32
            or char in ("\n", "\t")
            or (0x0900 <= cp <= 0x097F)
            or (0x0600 <= cp <= 0x06FF)
        ):
            cleaned.append(char)

    return "".join(cleaned).strip()


# ── Step 2: Convert to Documents ─────────────────────────────────
def convert_to_documents(records: list) -> list[Document]:
    """
    Convert Q&A records into LangChain Documents.

    Each document combines the user question + assistant answer.
    This gives the embedding model full context — both
    the question pattern AND the answer content.

    Why combine Q+A?
    → User asking "farmer schemes in UP" should match
      documents that contain both the question pattern
      AND the detailed scheme answer
    → Better semantic matching than answer-only
    """
    print("🔄 Converting Q&A pairs to documents...")

    documents = []
    skipped   = 0

    for i, record in enumerate(records):
        try:
            messages = record.get("messages", [])

            if not messages or len(messages) < 2:
                skipped += 1
                continue

            # Extract user question and assistant answer
            user_msg      = ""
            assistant_msg = ""

            for msg in messages:
                role    = msg.get("role", "")
                content = msg.get("content", "")

                if role == "user":
                    user_msg = content
                elif role == "assistant":
                    assistant_msg = content

            # Skip if either is missing
            if not user_msg or not assistant_msg:
                skipped += 1
                continue

            # Clean mojibake before any further processing so the
            # length check, content, and metadata all use clean text
            user_msg      = clean_mojibake(user_msg)
            assistant_msg = clean_mojibake(assistant_msg)

            # Skip very short records
            if len(assistant_msg) < 50:
                skipped += 1
                continue

            # Build document content
            # Format: Question header + answer
            # This makes retrieval match question-style queries
            content = f"QUESTION: {user_msg}\n\nANSWER:\n{assistant_msg}"

            # Extract scheme name from answer for metadata
            # Simple heuristic — first line often names the scheme
            first_line  = assistant_msg.split("\n")[0][:100]
            scheme_hint = first_line if len(first_line) > 10 else "General"

            doc = Document(
                page_content=content,
                metadata={
                    "source"      : HF_DATASET_NAME,
                    "record_index": i,
                    "question"    : user_msg[:200],  # truncate for metadata
                    "scheme_hint" : scheme_hint,
                    "split"       : "train" if i < 1927 else "validation",
                }
            )
            documents.append(doc)

        except Exception as e:
            skipped += 1
            continue

    print(f"✅ Created {len(documents)} documents")
    if skipped:
        print(f"   Skipped {skipped} incomplete records\n")

    return documents


# ── Step 3: Chunk documents ───────────────────────────────────────
def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Smart chunking that keeps Q&A pairs together.

    Strategy:
    1. Try to keep each Q&A pair as one chunk
    2. Only split if the Q&A pair exceeds max_size
    3. When splitting, always include the QUESTION
       in every chunk so context is never lost
    """
    print(f"✂️  Splitting into chunks...")

    all_chunks = []
    MAX_SIZE   = CHUNK_SIZE
    OVERLAP    = CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = MAX_SIZE,
        chunk_overlap   = OVERLAP,
        separators      = ["\n\nQUESTION:", "\n\nANSWER:", "\n\n", "\n", ". ", " "],
        length_function = len,
    )

    for doc in documents:
        content = doc.page_content

        # If document fits in one chunk — keep it whole
        if len(content) <= MAX_SIZE:
            all_chunks.append(doc)
            continue

        # Extract the QUESTION part to prepend to every split
        question_prefix = ""
        if content.startswith("QUESTION:"):
            lines = content.split("\n\n", 2)
            if len(lines) >= 1:
                question_prefix = lines[0][:200] + "\n\n"

        # Split the document
        sub_chunks = splitter.split_documents([doc])

        # Prepend question to every chunk that doesn't have it
        for chunk in sub_chunks:
            if (question_prefix and
                not chunk.page_content.startswith("QUESTION:")):
                chunk.page_content = (
                    question_prefix + chunk.page_content
                )
            all_chunks.append(chunk)

    avg_size = (
        sum(len(c.page_content) for c in all_chunks)
        // max(len(all_chunks), 1)
    )
    print(f"✅ Created {len(all_chunks)} chunks (avg {avg_size} chars)\n")
    return all_chunks


# ── Step 4: Load embedding model ──────────────────────────────────
def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load multilingual embedding model.

    paraphrase-multilingual-MiniLM-L12-v2:
    → Supports Hindi + English natively
    → 384 dimensions
    → ~117MB download, cached after first run
    → Runs on CPU
    """
    print(f"🧠 Loading embedding model...")
    print(f"   {EMBEDDING_MODEL}\n")

    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
        cache_folder  = str(HF_CACHE_DIR / "embeddings"),
    )

    print(f"✅ Embedding model loaded\n")
    return embeddings


# ── Step 5: Create ChromaDB ───────────────────────────────────────
def create_vectorstore(
    chunks    : list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Embed chunks and persist to ChromaDB.

    ChromaDB stores:
    → Text content of each chunk
    → 384-dim embedding vector
    → Metadata (source, question, scheme_hint)

    Persisted to vectorstore/ — loads instantly next time.
    """
    print(f"💾 Building ChromaDB vector store...")
    print(f"   Embedding {len(chunks)} chunks")
    print(f"   This takes 3-8 minutes — grab a chai ☕\n")

    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        collection_name   = COLLECTION_NAME,
        persist_directory = str(VECTORSTORE_DIR),
    )

    count = vectorstore._collection.count()
    print(f"✅ ChromaDB created — {count} chunks stored\n")
    return vectorstore


# ── Load existing ChromaDB ────────────────────────────────────────
def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Load existing ChromaDB from disk — fast."""
    print(f"📂 Loading existing ChromaDB...")

    vectorstore = Chroma(
        collection_name    = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory  = str(VECTORSTORE_DIR),
    )

    count = vectorstore._collection.count()
    print(f"✅ Loaded — {count} chunks ready\n")
    return vectorstore


# ── Check if vectorstore exists ───────────────────────────────────
def vectorstore_exists() -> bool:
    """Check if ChromaDB has already been built."""
    return (VECTORSTORE_DIR / "chroma.sqlite3").exists()


# ── Save summary ──────────────────────────────────────────────────
def save_summary(num_records: int, num_docs: int, num_chunks: int):
    """Save ingestion summary for reference."""
    summary = {
        "dataset"        : HF_DATASET_NAME,
        "records_loaded" : num_records,
        "documents"      : num_docs,
        "chunks"         : num_chunks,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size"     : CHUNK_SIZE,
        "chunk_overlap"  : CHUNK_OVERLAP,
    }
    path = DATA_DIR / "ingestion_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📋 Summary saved → {path}\n")


# ── Main pipeline ─────────────────────────────────────────────────
def run_ingestion() -> Chroma:
    """
    Run complete ingestion pipeline.

    First run  → downloads data, builds ChromaDB (~5-8 min)
    Later runs → loads existing ChromaDB (~2 seconds)
    """
    print("\n" + "=" * 60)
    print("  SCHEMESEEKER — Ingestion Pipeline")
    print("=" * 60 + "\n")

    embeddings = load_embedding_model()

    if vectorstore_exists():
        print("✅ ChromaDB already exists — loading it\n")
        return load_vectorstore(embeddings)

    print("🚀 First time setup — building ChromaDB...\n")

    records   = load_scheme_dataset()
    documents = convert_to_documents(records)
    chunks    = chunk_documents(documents)
    vs        = create_vectorstore(chunks, embeddings)

    save_summary(len(records), len(documents), len(chunks))

    print("=" * 60)
    print(f"  ✅ DONE — {len(chunks)} chunks indexed")
    print("=" * 60 + "\n")

    return vs


# ── Run + test ────────────────────────────────────────────────────
if __name__ == "__main__":
    vs = run_ingestion()

    print("🔍 Testing search — 'farmer schemes India'\n")
    results = vs.similarity_search("schemes for farmers in India", k=3)

    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Preview : {doc.page_content[:200]}...")
        print()

    print("✅ Ingestion working!")
    print("   Next → python core/retrieval.py")