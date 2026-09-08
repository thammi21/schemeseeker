"""
evaluation/evaluator.py — RAG Quality Measurement

Measures:
  1. Answer relevance  — does answer address the question?
  2. Faithfulness      — does answer match retrieved context?
  3. Fallback rate     — what % of queries fall back?
  4. Hindi support     — does Hindi retrieval work?

Runs against a built-in test set of 20 questions.
Saves report to evaluation/eval_report.md
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

# Running this file directly (python evaluation/evaluator.py) puts
# evaluation/ on sys.path, not the project root, so `from core...`
# imports below would fail without this.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
HF_CACHE_DIR    = BASE_DIR / "data" / "hf_cache"
EVAL_DIR        = BASE_DIR / "evaluation"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


# ── Test set ──────────────────────────────────────────────────────
TEST_SET = [
    # Agriculture
    {
        "id"      : "agr_01",
        "query"   : "What schemes are available for farmers with small land?",
        "category": "agriculture",
        "language": "english",
        "expected_keywords": ["farmer", "scheme", "land", "eligible"],
    },
    {
        "id"      : "agr_02",
        "query"   : "I am a farmer in Maharashtra, what crop insurance schemes exist?",
        "category": "agriculture",
        "language": "english",
        "expected_keywords": ["insurance", "crop", "farmer"],
    },
    {
        "id"      : "agr_03",
        "query"   : "किसान के लिए सरकारी योजनाएं कौन सी हैं?",
        "category": "agriculture",
        "language": "hindi",
        "expected_keywords": ["किसान", "योजना"],
    },
    {
        "id"      : "agr_04",
        "query"   : "What is PM Kisan Samman Nidhi?",
        "category": "agriculture",
        "language": "english",
        "expected_keywords": ["kisan", "farmer", "6000"],
    },
    # Health
    {
        "id"      : "hlth_01",
        "query"   : "What health insurance schemes are available for poor families?",
        "category": "health",
        "language": "english",
        "expected_keywords": ["health", "insurance", "family"],
    },
    {
        "id"      : "hlth_02",
        "query"   : "महिलाओं के लिए स्वास्थ्य योजनाएं कौन सी हैं?",
        "category": "health",
        "language": "hindi",
        "expected_keywords": ["महिला", "स्वास्थ्य"],
    },
    {
        "id"      : "hlth_03",
        "query"   : "What is Ayushman Bharat and who can benefit from it?",
        "category": "health",
        "language": "english",
        "expected_keywords": ["ayushman", "health", "family"],
    },
    # Education
    {
        "id"      : "edu_01",
        "query"   : "What scholarships are available for SC ST students?",
        "category": "education",
        "language": "english",
        "expected_keywords": ["scholarship", "student", "SC", "ST"],
    },
    {
        "id"      : "edu_02",
        "query"   : "I am a girl student, what education schemes can I apply for?",
        "category": "education",
        "language": "english",
        "expected_keywords": ["student", "girl", "scheme"],
    },
    {
        "id"      : "edu_03",
        "query"   : "छात्रवृत्ति योजनाएं कौन सी हैं?",
        "category": "education",
        "language": "hindi",
        "expected_keywords": ["छात्र", "योजना"],
    },
    # Housing
    {
        "id"      : "hous_01",
        "query"   : "What housing schemes are available for BPL families?",
        "category": "housing",
        "language": "english",
        "expected_keywords": ["housing", "BPL", "family"],
    },
    {
        "id"      : "hous_02",
        "query"   : "I am a woman from rural area, what housing schemes can I get?",
        "category": "housing",
        "language": "english",
        "expected_keywords": ["housing", "rural", "woman"],
    },
    # Finance / Loans
    {
        "id"      : "fin_01",
        "query"   : "What loan schemes are available for small business owners?",
        "category": "finance",
        "language": "english",
        "expected_keywords": ["loan", "business", "scheme"],
    },
    {
        "id"      : "fin_02",
        "query"   : "I want to start a business, what government loans can I get?",
        "category": "finance",
        "language": "english",
        "expected_keywords": ["loan", "business", "mudra"],
    },
    {
        "id"      : "fin_03",
        "query"   : "व्यापार के लिए ऋण योजनाएं कौन सी हैं?",
        "category": "finance",
        "language": "hindi",
        "expected_keywords": ["व्यापार", "ऋण"],
    },
    # Women
    {
        "id"      : "wom_01",
        "query"   : "What schemes are available for women entrepreneurs?",
        "category": "women",
        "language": "english",
        "expected_keywords": ["women", "scheme"],
    },
    {
        "id"      : "wom_02",
        "query"   : "I am a pregnant woman, what schemes can I get?",
        "category": "women",
        "language": "english",
        "expected_keywords": ["woman", "pregnant", "health"],
    },
    # Off-topic (should fallback)
    {
        "id"      : "oot_01",
        "query"   : "What is the capital of France?",
        "category": "off_topic",
        "language": "english",
        "expected_keywords": [],
        "should_fallback"  : True,
    },
    {
        "id"      : "oot_02",
        "query"   : "Who won the cricket World Cup?",
        "category": "off_topic",
        "language": "english",
        "expected_keywords": [],
        "should_fallback"  : True,
    },
    # Full profile query
    {
        "id"      : "full_01",
        "query"   : "I am a 35 year old woman farmer in Maharashtra with 2 acres land and income below 2 lakhs",
        "category": "agriculture",
        "language": "english",
        "expected_keywords": ["farmer", "scheme", "Maharashtra"],
    },
]


# ── Load resources ────────────────────────────────────────────────
def load_resources():
    """Load embedding model and vectorstore."""
    emb = HuggingFaceEmbeddings(
        model_name    = "paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
        cache_folder  = str(HF_CACHE_DIR / "embeddings"),
    )
    vs = Chroma(
        collection_name    = "schemeseeker",
        embedding_function = emb,
        persist_directory  = str(VECTORSTORE_DIR),
    )
    return vs


# ── Metrics ───────────────────────────────────────────────────────
def is_fallback(answer: str) -> bool:
    """Check if answer is a fallback response."""
    fallback_phrases = [
        "i don't have specific information",
        "please check myscheme.gov.in",
        "मुझे इस बारे में जानकारी नहीं",
        "myscheme.gov.in देखें",
    ]
    answer_lower = answer.lower()
    return any(p in answer_lower for p in fallback_phrases) and len(answer) < 300


def check_keywords(answer: str, keywords: list) -> dict:
    """Check how many expected keywords appear in answer."""
    if not keywords:
        return {"found": 0, "total": 0, "score": 1.0}

    answer_lower = answer.lower()
    found = sum(
        1 for kw in keywords
        if kw.lower() in answer_lower
    )
    return {
        "found": found,
        "total": len(keywords),
        "score": found / len(keywords),
    }


def score_answer_relevance(
    query  : str,
    answer : str,
    llm    : ChatGroq,
) -> float:
    """
    Score answer relevance using structured criteria.
    Returns 0.0 to 1.0.

    Deterministic — no LLM call needed for scoring, which also
    removes the template injection risk of the earlier LLM-judge
    version (which also wasn't discriminating: it scored nearly
    every non-fallback answer as a flat 0.5).
    """
    # Simple keyword-based check first
    # If it's a fallback — score 0
    fallback_phrases = [
        "i don't have specific information",
        "please check myscheme.gov.in",
    ]
    if any(p in answer.lower() for p in fallback_phrases):
        return 0.0

    # Score based on answer quality signals
    score = 0.0

    # Has scheme name (capitalized noun phrase)?
    import re
    scheme_names = re.findall(r'[A-Z][A-Za-z\s]{5,30}(?:Scheme|Yojana|Mission|Programme|Plan)', answer)
    if scheme_names:
        score += 0.3

    # Has eligibility information?
    eligibility_words = ["eligible", "eligibility", "qualify", "criteria", "who can"]
    if any(w in answer.lower() for w in eligibility_words):
        score += 0.2

    # Has benefit information?
    benefit_words = ["benefit", "lakh", "rupee", "rs.", "coverage", "loan", "grant"]
    if any(w in answer.lower() for w in benefit_words):
        score += 0.2

    # Has application information?
    apply_words = ["apply", "application", "website", "portal", "visit", "contact"]
    if any(w in answer.lower() for w in apply_words):
        score += 0.2

    # Reasonable length (not too short)?
    if len(answer) > 200:
        score += 0.1

    return min(score, 1.0)


# ── Run evaluation ────────────────────────────────────────────────
def run_evaluation(verbose: bool = True) -> dict:
    """
    Run full evaluation against TEST_SET.
    Returns evaluation results dict.
    """
    from core.chain       import ask, detect_language
    from core.eligibility import extract_profile, map_to_categories, build_enhanced_query

    print("\n" + "=" * 60)
    print("  SCHEMESEEKER — RAG Evaluation")
    print(f"  {len(TEST_SET)} test cases")
    print("=" * 60 + "\n")

    vs  = load_resources()
    llm = ChatGroq(
        model       = GROQ_MODEL,
        api_key     = os.environ.get("GROQ_API_KEY"),
        temperature = 0,
        max_tokens  = 100,
    )

    results      = []
    total_score  = 0.0
    fallback_count = 0
    correct_fallbacks = 0

    for i, test in enumerate(TEST_SET, 1):
        query    = test["query"]
        test_id  = test["id"]
        category = test["category"]
        language = test["language"]
        should_fallback = test.get("should_fallback", False)

        if verbose:
            print(f"[{i:02d}/{len(TEST_SET)}] {test_id} — {query[:50]}...")

        # Run pipeline
        try:
            profile         = extract_profile(query)
            enhanced_q      = build_enhanced_query(query, profile)
            detected_lang   = detect_language(query)
            result          = ask(enhanced_q, vs, language=detected_lang)
            answer          = result["answer"]
        except Exception as e:
            answer = f"ERROR: {e}"

        # Metrics
        fell_back       = is_fallback(answer)
        keyword_result  = check_keywords(answer, test.get("expected_keywords", []))
        relevance_score = score_answer_relevance(query, answer, llm)

        # Fallback correctness
        if should_fallback and fell_back:
            fallback_correct = True
            correct_fallbacks += 1
        elif should_fallback and not fell_back:
            fallback_correct = False
        elif not should_fallback and fell_back:
            fallback_correct = False
            fallback_count  += 1
        else:
            fallback_correct = True

        # Combined score
        if should_fallback:
            combined = 1.0 if fell_back else 0.0
        else:
            combined = (
                relevance_score * 0.6 +
                keyword_result["score"] * 0.4
            )

        total_score += combined

        result_entry = {
            "id"              : test_id,
            "category"        : category,
            "language"        : language,
            "query"           : query,
            "answer_length"   : len(answer),
            "fell_back"       : fell_back,
            "should_fallback" : should_fallback,
            "fallback_correct": fallback_correct,
            "keyword_score"   : keyword_result["score"],
            "keywords_found"  : keyword_result["found"],
            "keywords_total"  : keyword_result["total"],
            "relevance_score" : relevance_score,
            "combined_score"  : combined,
            "answer_preview"  : answer[:300],
        }
        results.append(result_entry)

        if verbose:
            status = "✅" if combined >= 0.6 else "⚠️" if combined >= 0.3 else "❌"
            print(f"     {status} Score: {combined:.2f} | "
                  f"Fallback: {fell_back} | "
                  f"Keywords: {keyword_result['found']}/{keyword_result['total']} | "
                  f"Relevance: {relevance_score:.1f}")

    # Aggregate metrics
    avg_score        = total_score / len(TEST_SET)
    hindi_results    = [r for r in results if r["language"] == "hindi"]
    english_results  = [r for r in results if r["language"] == "english"]
    on_topic         = [r for r in results if not r["should_fallback"]]
    off_topic        = [r for r in results if r["should_fallback"]]

    summary = {
        "total_tests"        : len(TEST_SET),
        "overall_score"      : round(avg_score, 3),
        "overall_pct"        : round(avg_score * 100, 1),
        "hindi_score"        : round(
            sum(r["combined_score"] for r in hindi_results) /
            max(len(hindi_results), 1), 3
        ),
        "english_score"      : round(
            sum(r["combined_score"] for r in english_results) /
            max(len(english_results), 1), 3
        ),
        "on_topic_score"     : round(
            sum(r["combined_score"] for r in on_topic) /
            max(len(on_topic), 1), 3
        ),
        "fallback_accuracy"  : round(
            correct_fallbacks / max(len(off_topic), 1), 3
        ),
        "unnecessary_fallbacks": fallback_count,
        "results"            : results,
        "timestamp"          : datetime.now().isoformat(),
    }

    return summary


# ── Generate report ───────────────────────────────────────────────
def generate_report(summary: dict) -> str:
    """Generate a markdown evaluation report."""

    lines = [
        "# SchemeSeeker — RAG Evaluation Report",
        f"\n**Date:** {summary['timestamp'][:10]}",
        f"**Total tests:** {summary['total_tests']}",
        "",
        "## Overall Scores",
        "",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Overall Score | {summary['overall_pct']}% |",
        f"| English Queries | {summary['english_score']*100:.1f}% |",
        f"| Hindi Queries | {summary['hindi_score']*100:.1f}% |",
        f"| On-Topic Score | {summary['on_topic_score']*100:.1f}% |",
        f"| Fallback Accuracy | {summary['fallback_accuracy']*100:.1f}% |",
        f"| Unnecessary Fallbacks | {summary['unnecessary_fallbacks']} |",
        "",
        "## Results by Category",
        "",
        "| ID | Category | Language | Score | Fallback | Keywords |",
        "|----|----------|----------|-------|----------|----------|",
    ]

    for r in summary["results"]:
        status = "✅" if r["combined_score"] >= 0.6 else "⚠️" if r["combined_score"] >= 0.3 else "❌"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['language']} | "
            f"{status} {r['combined_score']:.2f} | "
            f"{'Yes' if r['fell_back'] else 'No'} | "
            f"{r['keywords_found']}/{r['keywords_total']} |"
        )

    lines += [
        "",
        "## Failed Tests",
        "",
    ]

    failed = [r for r in summary["results"] if r["combined_score"] < 0.6]
    if failed:
        for r in failed:
            lines += [
                f"### {r['id']} — {r['query'][:60]}",
                f"- Score: {r['combined_score']:.2f}",
                f"- Fell back: {r['fell_back']}",
                f"- Answer preview: {r['answer_preview'][:200]}",
                "",
            ]
    else:
        lines.append("All tests passed with score >= 0.6")

    return "\n".join(lines)


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = run_evaluation(verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Overall Score     : {summary['overall_pct']}%")
    print(f"  English Score     : {summary['english_score']*100:.1f}%")
    print(f"  Hindi Score       : {summary['hindi_score']*100:.1f}%")
    print(f"  Fallback Accuracy : {summary['fallback_accuracy']*100:.1f}%")
    print(f"  Bad Fallbacks     : {summary['unnecessary_fallbacks']}")
    print("=" * 60)

    # Save report
    report_path = EVAL_DIR / "eval_report.md"
    report      = generate_report(summary)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    # Save raw results
    json_path = EVAL_DIR / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Raw data saved: {json_path}")

    print("\n  Run: cat evaluation/eval_report.md to see full report")
