# 🇮🇳 SchemeSeeker — Indian Government Schemes AI Assistant

An AI-powered RAG system that helps Indian citizens discover government 
welfare schemes they are eligible for — in Hindi and English.
 
🔴 **[Live Demo →](https://schemeseeker.streamlit.app/)**
> ⚠️ First load takes 8–10 minutes to build the scheme database. Subsequent loads are instant.

---

## The Problem

India has 500+ central and state government schemes with a combined 
budget exceeding ₹15 lakh crore annually. Most eligible citizens never 
claim their benefits because:

- Scheme information is scattered across dozens of government websites
- Official documents are dense, jargon-heavy, and in English only
- No single place to ask "what am I eligible for?"
- A farmer in UP with 2 acres could qualify for 5+ schemes simultaneously
  but has no easy way to discover this

---

## What SchemeSeeker Does

```
User types (Hindi or English):
"I am a 35 year old farmer in Maharashtra with 2 acres of land"

SchemeSeeker:
  Step 1 → Extracts profile
           {occupation: farmer, age: 35, state: Maharashtra, land: 2 acres}

  Step 2 → Builds enhanced search query from profile

  Step 3 → Searches 12,000+ chunks from 2,000+ Q&A pairs
           using MMR retrieval (semantic + diversity)

  Step 4 → LLM synthesises answer from retrieved context only
           — never hallucinates

  Step 5 → Returns structured answer with scheme names,
           eligibility, benefits, how to apply, official links

Works in Hindi too:
"किसान के लिए कौन सी योजनाएं हैं?"
→ Full Hindi answer about agricultural schemes
```

---

## RAG Architecture

```
INDEXING PHASE (once, on first run):
─────────────────────────────────────
BharatSchemes Dataset (2,029 Q&A pairs)
    ↓ core/ingestion.py
Convert to LangChain Documents
    ↓ Smart chunking (800 chars, 100 overlap, QUESTION prefix preserved)
12,410 chunks
    ↓ paraphrase-multilingual-MiniLM-L12-v2 embedding model
384-dimensional vectors (supports Hindi + English natively)
    ↓
ChromaDB vector store (persisted to disk)

QUERYING PHASE (every user question):
──────────────────────────────────────
User question (Hindi or English)
    ↓ core/eligibility.py — PydanticOutputParser
Structured UserProfile {occupation, age, state, income...}
    ↓ build_enhanced_query()
"for farmer government scheme in Maharashtra aged 35 with 2.0 acres land"
    ↓ core/retrieval.py — MMR search (k=5, fetch_k=20)
Top 5 diverse relevant chunks
    ↓ core/chain.py — LCEL: prompt | llm | StrOutputParser
Groq Llama / GPT-OSS generates grounded answer
    ↓ app.py — Streamlit streaming UI
Answer streamed token by token + source citations
```

---

## Evaluation Results

Evaluated against 20 test cases covering 6 categories 
(agriculture, health, housing, education, finance, off-topic):

| Metric | Score |
|--------|-------|
| **Overall Score** | **72.6%** |
| English Queries | 73.2% |
| Hindi Queries | 71.0% |
| Fallback Accuracy | **100%** |
| Tests Passing (≥0.6) | 16 / 20 |
| Unnecessary Fallbacks | 0 |

**100% fallback accuracy** means the system never fabricated government 
scheme information — the critical safety requirement for a civic tool.

Remaining 4 failures are traceable to embedding model paraphrase 
sensitivity with near-homonym scheme names — a retrieval quality issue, 
not a hallucination issue.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Groq — openai/gpt-oss-120b | Fastest inference, free tier |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | Hindi + English native support |
| Vector DB | ChromaDB | Local, persistent, no API needed |
| Orchestration | LangChain LCEL | Pipe syntax, MMR retriever |
| Structured output | Pydantic v2 | Typed profile extraction |
| UI | Streamlit | Fast iteration, streaming support |
| Dataset | BharatSchemes (HuggingFace) | 2,029 scheme Q&A pairs |
| Language | Python 3.10+ | |

---

## Key Technical Features

**1. Multilingual RAG**
Hindi queries find English documents via cross-lingual embeddings.
`paraphrase-multilingual-MiniLM-L12-v2` maps "किसान" and "farmer"
to similar vector space — no translation layer needed.

**2. Eligibility-aware retrieval**
Extracts structured `UserProfile` from natural language before 
searching. "I am a 35 year old woman farmer" → 
`{occupation: farmer, gender: female, age: 35}` → enhanced 
search query → more precise retrieval.

**3. Smart chunking with QUESTION prefix**
Every split chunk retains the original question context —
prevents the LLM from receiving nameless content fragments
with no scheme identification.

**4. MMR retrieval**
Maximal Marginal Relevance returns diverse chunks rather than
5 near-identical fragments — covers more schemes per query.

**5. Grounded-only answers**
System prompt enforces: answer from context OR give fallback.
Never both. Never hallucinate. Verified at 100% fallback accuracy.

---

## Project Structure

```
schemeseeker/
├── app.py                    # Streamlit UI — entry point
├── core/
│   ├── ingestion.py          # PDF loading, chunking, ChromaDB
│   ├── retrieval.py          # Similarity, MMR, hybrid search
│   ├── chain.py              # RAG chain, bilingual prompts, streaming
│   └── eligibility.py        # Profile extraction, category mapping
├── utils/
│   └── helpers.py            # Formatting, display, session helpers
├── evaluation/
│   ├── evaluator.py          # 20-case evaluation pipeline
│   ├── eval_report.md        # Latest evaluation report
│   └── eval_results.json     # Raw evaluation data
├── data/
│   └── manual_schemes.py     # Curated entries for dataset gaps
└── vectorstore/              # ChromaDB (gitignored, built on first run)
```

---

## Run Locally

**1. Clone**
```bash
git clone https://github.com/thammi21/schemeseeker
cd schemeseeker
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add API key**

Get a free Groq API key at [console.groq.com](https://console.groq.com)

Create `.env`:
```
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

**5. Run**
```powershell
$env:PYTHONIOENCODING="utf-8"
streamlit run app.py
```

First run downloads the dataset (~15MB) and builds ChromaDB 
(~8 minutes). Every subsequent run loads in ~5 seconds.

---

## Run Evaluation

```bash
python evaluation/evaluator.py
```

Outputs:
- `evaluation/eval_report.md` — readable markdown report
- `evaluation/eval_results.json` — raw scores per test case

---

## What I Learned Building This

- Chunking strategy is the biggest factor in RAG retrieval quality
  — naive splitting loses scheme names from content
- Encoding corruption (mojibake) in source data feeds into LLM 
  prompts and causes confident-sounding fallbacks on relevant queries
- MMR retrieval consistently outperforms pure similarity for 
  document corpora with redundant content
- Cross-lingual embeddings eliminate translation overhead for 
  bilingual RAG — Hindi queries find English documents natively
- Evaluation without a test set is just vibes — 
  a 20-case structured eval revealed issues invisible during manual testing
- 100% fallback accuracy matters more than overall score 
  for civic/government information tools

---

## Limitations and Known Issues

- **PM Kisan short-form query**: "What is PM Kisan?" ranks 
  near-homonym schemes above the correct entry due to embedding 
  model sensitivity. Fixed for longer phrasings.
- **Dataset coverage**: BharatSchemes covers 2,029 schemes but not 
  all 500+ central schemes. Manual entries added for highest-priority gaps.
- **State scheme coverage**: Central schemes are better covered 
  than state-specific schemes.
- **Run-to-run variance**: Evaluation scores vary ±2-4% due to 
  non-determinism in MMR retrieval ordering.

---

## Roadmap

- [ ] Add RAGAS evaluation for faithfulness + context precision
- [ ] Replace ChromaDB with Pinecone for production scale
- [ ] Add Tamil and Telugu query support
- [ ] Integrate live MyScheme API for current benefit amounts
- [ ] Add re-ranking layer (cross-encoder) for better retrieval precision
- [ ] Build WhatsApp bot interface for rural accessibility
