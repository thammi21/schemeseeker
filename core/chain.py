# core/chain.py
"""
SCRIPT 3 — RAG Chain

This is the brain of SchemeSeeker.
Combines retrieval + generation into one pipeline.

Flow:
  User query
      ↓
  Detect language (Hindi or English)
      ↓
  Retrieve top-5 relevant chunks from ChromaDB (MMR)
      ↓
  Build prompt: system + retrieved context + user query
      ↓
  Groq LLM generates answer from context only
      ↓
  Return structured answer + source citations

Week 4 RAG concepts used:
  - RetrievalQA pattern
  - Source citation
  - Context-grounded answering
  - Bilingual support
  - LCEL pipe chain
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
HF_CACHE_DIR    = BASE_DIR / "data" / "hf_cache"

# ── Constants ─────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "schemeseeker"
GROQ_MODEL      = "openai/gpt-oss-120b"


# ── Language detection ────────────────────────────────────────────
def detect_language(text: str) -> str:
    """
    Detect if query is Hindi or English.

    Simple approach: check if any Devanagari characters present.
    Devanagari Unicode range: U+0900 to U+097F

    Returns "hindi" or "english".
    """
    for char in text:
        if '\u0900' <= char <= '\u097F':
            return "hindi"
    return "english"


# ── Format retrieved docs for prompt ─────────────────────────────
def format_context(docs: list) -> str:
    """
    Format retrieved documents into a single context string.

    Each chunk is numbered and separated clearly.
    The LLM uses these numbers for source citation.

    Example output:
      [Source 1]
      QUESTION: What is PM Kisan?
      ANSWER: PM Kisan gives ₹6000 per year...

      [Source 2]
      QUESTION: Who is eligible for PM Kisan?
      ANSWER: All landholding farmer families...
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        formatted_parts.append(
            f"[Source {i}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted_parts)


# ── Load LLM ─────────────────────────────────────────────────────
def load_llm() -> ChatGroq:
    """
    Load Groq LLM for answer generation.

    temperature=0.3:
    → Low enough for factual, consistent answers
    → Not 0 because we want natural language variety
    → Not 0.7 because government info must be accurate
    """
    return ChatGroq(
        model       = GROQ_MODEL,
        api_key     = os.environ.get("GROQ_API_KEY"),
        temperature = 0.3,
        max_tokens  = 1500,
    )


# ── Load vectorstore ──────────────────────────────────────────────
def load_vectorstore() -> Chroma:
    """Load ChromaDB from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
        cache_folder  = str(HF_CACHE_DIR / "embeddings"),
    )

    return Chroma(
        collection_name    = COLLECTION_NAME,
        embedding_function = embeddings,
        persist_directory  = str(VECTORSTORE_DIR),
    )


# ── RAG Prompts ───────────────────────────────────────────────────
# Two prompts — one per language
# The LLM responds in the same language as the query

ENGLISH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are SchemeSeeker, an expert AI assistant helping Indian citizens
discover government welfare schemes they are eligible for.

YOUR JOB:
Synthesise the retrieved context chunks into a helpful answer.
The context is fragmented — pieces come from different schemes.
Your job is to identify ALL relevant schemes mentioned across
ALL chunks and present them together as one structured answer.

RULES:
1. Read ALL context chunks carefully before deciding
2. If ANY chunk contains scheme information relevant to the
   user — answer from it. Partial information is enough.
3. Combine information across multiple chunks freely
4. Structure your answer as:
   - Scheme name (bold)
   - Who is eligible
   - What benefit they get
   - How to apply / official site
5. Only use the fallback if NONE of the chunks contain
   ANY scheme information relevant to the query. When using the
   fallback, respond with EXACTLY this line and nothing else:
   "I don't have specific information about this in my database. Please check myscheme.gov.in"
6. NEVER mix a real answer with the fallback line
7. Never hallucinate — only state what is in the context

CONTEXT FROM SCHEME DATABASE:
{context}"""
    ),
    (
        "user",
        "{question}"
    )
])

HINDI_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """आप SchemeSeeker हैं — भारतीय नागरिकों को सरकारी योजनाएं
खोजने में मदद करने वाले AI सहायक।

आपका काम:
सभी context chunks को पढ़कर एक उपयोगी उत्तर बनाएं।
Context टुकड़ों में आता है — अलग-अलग योजनाओं से।
आपका काम है सभी प्रासंगिक योजनाओं को मिलाकर एक उत्तर देना।

नियम:
1. उत्तर देने से पहले सभी chunks ध्यान से पढ़ें
2. यदि किसी भी chunk में प्रासंगिक जानकारी है → उत्तर दें
3. कई chunks से जानकारी मिलाएं
4. उत्तर की संरचना:
   - योजना का नाम
   - पात्रता
   - लाभ
   - आवेदन कैसे करें
5. Fallback केवल तभी दें जब किसी भी chunk में
   कोई प्रासंगिक जानकारी न हो। Fallback देते समय ठीक यही
   पंक्ति लिखें, और कुछ नहीं:
   "मुझे इस बारे में जानकारी नहीं है। कृपया myscheme.gov.in देखें।"
6. उत्तर और fallback एक साथ कभी नहीं
7. केवल context में जो हो वही बताएं

संदर्भ:
{context}"""
    ),
    (
        "user",
        "{question}"
    )
])


# ── Build RAG chain ───────────────────────────────────────────────
def build_rag_chain(vectorstore: Chroma, language: str = "english"):
    """
    Build the full RAG chain using LCEL pipes.

    Chain flow:
      {"question": query, "context": retrieved_docs}
           ↓
      prompt (formats question + context into messages)
           ↓
      llm (Groq generates answer)
           ↓
      StrOutputParser (extracts text from AIMessage)

    The retriever is a Runnable — it accepts a string
    and returns a list of Documents.

    RunnablePassthrough passes the question through
    while the retriever fetches context in parallel.

    Args:
        vectorstore: ChromaDB instance
        language   : "english" or "hindi"

    Returns:
        LCEL chain ready for .invoke()
    """
    llm = load_llm()

    # MMR retriever — diverse, non-redundant results
    retriever = vectorstore.as_retriever(
        search_type   = "mmr",
        search_kwargs = {
            "k"          : 5,
            "fetch_k"    : 20,
            "lambda_mult": 0.7,
        }
    )

    # Pick prompt based on language
    prompt = HINDI_PROMPT if language == "hindi" else ENGLISH_PROMPT

    # Build LCEL chain
    # This is the RAG pattern:
    #   1. RunnablePassthrough passes question through unchanged
    #   2. retriever fetches relevant chunks for the question
    #   3. format_context converts docs to string
    #   4. Both feed into prompt
    #   5. LLM generates answer
    #   6. Parser extracts text

    chain = (
        {
            # Pass question through unchanged to prompt
            "question": RunnablePassthrough(),
            # Retrieve relevant chunks and format as string
            "context" : retriever | RunnableLambda(format_context),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── Main function: ask a question ─────────────────────────────────
def ask(
    question    : str,
    vectorstore : Chroma,
    verbose     : bool = False,
    language    : str = None,
) -> dict:
    """
    Main entry point — ask SchemeSeeker a question.

    Steps:
      1. Detect language
      2. Retrieve relevant chunks
      3. Generate answer
      4. Return answer + sources

    Args:
        question   : User's question (Hindi or English)
        vectorstore: ChromaDB instance
        verbose    : If True, show retrieved chunks
        language   : Language already detected from the user's
                     original query. If omitted, falls back to
                     detecting from `question` — but `question`
                     may be a profile-enhanced query with the
                     original-language text stripped out (see
                     eligibility.build_enhanced_query).

    Returns:
        dict with keys:
          answer   : LLM-generated answer string
          sources  : list of retrieved chunk previews
          language : detected language
          question : original question
    """
    # Step 1: Detect language
    if language is None:
        language = detect_language(question)

    if verbose:
        print(f"\n🔍 Query     : {question}")
        print(f"🌐 Language  : {language}")

    # Step 2: Build chain (with correct language prompt)
    chain, retriever = build_rag_chain(vectorstore, language)

    # Step 3: Retrieve chunks separately for source display
    retrieved_docs = retriever.invoke(question)

    if verbose:
        print(f"📚 Retrieved : {len(retrieved_docs)} chunks")
        print("\n--- Retrieved Context ---")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"[{i}] {doc.page_content[:150]}...")
        print("--- End Context ---\n")

    # Step 4: Generate answer using full chain
    answer = chain.invoke(question)

    # Step 5: Format sources for display
    sources = []
    seen    = set()  # deduplicate sources

    for doc in retrieved_docs:
        preview = doc.page_content[:200]

        # Skip duplicate chunks
        if preview in seen:
            continue
        seen.add(preview)

        sources.append({
            "preview" : preview,
            "question": doc.metadata.get("question", ""),
            "source"  : doc.metadata.get("source", ""),
            "scheme"  : doc.metadata.get("scheme_hint", ""),
        })

    return {
        "answer"  : answer,
        "sources" : sources,
        "language": language,
        "question": question,
    }


# ── Stream response (for Streamlit) ──────────────────────────────
def ask_streaming(
    question    : str,
    vectorstore : Chroma,
    language    : str = None,
):
    """
    Streaming version that also returns sources in final chunk.
    Fixes double-invocation — retrieves once, streams once.
    Yields str tokens, then final dict with sources.

    language: pass the language already detected from the user's
    original query. If omitted, falls back to detecting from
    `question` — but `question` may be a profile-enhanced query
    with the original-language text stripped out (see
    eligibility.build_enhanced_query), which would misdetect
    language and answer in the wrong one.
    """
    if language is None:
        language = detect_language(question)
    llm = load_llm()

    retriever = vectorstore.as_retriever(
        search_type   = "mmr",
        search_kwargs = {"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # Retrieve ONCE
    retrieved_docs = retriever.invoke(question)
    context        = format_context(retrieved_docs)

    # Pick prompt based on language
    prompt = HINDI_PROMPT if language == "hindi" else ENGLISH_PROMPT

    # Build chain without retriever — context already retrieved
    chain = prompt | llm | StrOutputParser()

    # Stream tokens
    for token in chain.stream({
        "question": question,
        "context" : context
    }):
        yield token

    # Collect sources
    sources = []
    seen    = set()
    for doc in retrieved_docs:
        preview = doc.page_content[:200]
        if preview not in seen:
            seen.add(preview)
            sources.append({
                "preview" : preview,
                "question": doc.metadata.get("question", ""),
                "source"  : doc.metadata.get("source", ""),
                "scheme"  : doc.metadata.get("scheme_hint", ""),
            })

    # Final yield — sources signal
    yield {"__sources__": sources, "__language__": language}


# ── Run directly ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SCHEMESEEKER — RAG Chain Test")
    print("=" * 60 + "\n")

    # Load vectorstore
    print("Loading vectorstore...")
    vs = load_vectorstore()
    print("✅ Loaded\n")

    # Test questions
    test_questions = [
        # English
        "What is PM Kisan and who is eligible?",
        "I am a woman from rural area, what housing schemes can I apply for?",
        "What health insurance schemes are available for poor families?",
        # Hindi
        "किसान के लिए कौन सी योजनाएं हैं?",
        "महिलाओं के लिए स्वास्थ्य योजनाएं कौन सी हैं?",
    ]

    for question in test_questions:
        print("\n" + "─" * 60)
        result = ask(question, vs, verbose=False)

        lang_emoji = "🇮🇳" if result["language"] == "hindi" else "🇬🇧"

        print(f"{lang_emoji} Q: {question}")
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📚 Sources used: {len(result['sources'])}")
        for i, src in enumerate(result['sources'][:2], 1):
            print(f"   [{i}] {src['preview'][:100]}...")

    print("\n" + "=" * 60)
    print("  ✅ RAG Chain working!")
    print("  Next → python core/eligibility.py")
    print("=" * 60 + "\n")