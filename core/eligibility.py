# core/eligibility.py
"""
SCRIPT 4 — Eligibility Extraction

Extracts structured user profile from natural language queries.
Uses LangChain's PydanticOutputParser — same concept as Week 3.

Flow:
  "I am a 35 year old farmer in UP with 2 acres of land"
       ↓ LLM with structured output
  UserProfile(
      occupation="farmer",
      age=35,
      state="Uttar Pradesh",
      land_acres=2.0,
      gender=None,
      annual_income_lakhs=None
  )
       ↓ map_to_categories()
  ["agriculture", "finance"]
       ↓ used to filter ChromaDB before search

Why extract before searching?
  Without filter: search all 17,865 chunks (lots of noise)
  With filter   : search only relevant category chunks (precise)
"""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


# ── Pydantic model for user profile ──────────────────────────────
class UserProfile(BaseModel):
    """
    Structured profile extracted from user's natural language query.

    Every field is Optional — user may not mention all details.
    None means "not mentioned" — not "not applicable."
    """
    occupation: Optional[str] = Field(
        default=None,
        description="User's occupation: farmer, student, woman, entrepreneur, worker, unemployed, etc."
    )
    age: Optional[int] = Field(
        default=None,
        description="User's age as integer. None if not mentioned."
    )
    gender: Optional[str] = Field(
        default=None,
        description="User's gender: male, female, other. None if not mentioned."
    )
    state: Optional[str] = Field(
        default=None,
        description="Indian state name in full. E.g. 'Uttar Pradesh' not 'UP'."
    )
    annual_income_lakhs: Optional[float] = Field(
        default=None,
        description="Annual income in lakhs. E.g. 2.5 for ₹2.5 lakh. None if not mentioned."
    )
    land_acres: Optional[float] = Field(
        default=None,
        description="Land holding in acres. None if not mentioned."
    )
    category: Optional[str] = Field(
        default=None,
        description="Social category: SC, ST, OBC, General. None if not mentioned."
    )
    education: Optional[str] = Field(
        default=None,
        description="Education level: 10th, 12th, graduate, postgraduate, etc."
    )
    is_bpl: Optional[bool] = Field(
        default=None,
        description="True if user mentions BPL/Below Poverty Line. None if not mentioned."
    )
    specific_need: Optional[str] = Field(
        default=None,
        description="Specific need mentioned: housing, health, education, loan, insurance, pension, skill training, etc."
    )


# ── Category mapping ──────────────────────────────────────────────
# Maps user profile fields to scheme categories
# Used to build ChromaDB metadata filter

OCCUPATION_TO_CATEGORIES = {
    "farmer"        : ["agriculture", "finance", "insurance"],
    "student"       : ["education", "scholarship"],
    "woman"         : ["women", "health", "housing"],
    "entrepreneur"  : ["finance", "startup", "msme"],
    "worker"        : ["labour", "pension", "insurance"],
    "unemployed"    : ["skill", "finance", "employment"],
    "fisherman"     : ["agriculture", "finance"],
    "artisan"       : ["skill", "finance", "msme"],
    "disabled"      : ["disability", "pension", "health"],
    "senior"        : ["pension", "health"],
    "pregnant"      : ["health", "women"],
}

NEED_TO_CATEGORIES = {
    "housing"       : ["housing"],
    "health"        : ["health", "insurance"],
    "education"     : ["education", "scholarship"],
    "loan"          : ["finance", "msme"],
    "insurance"     : ["insurance"],
    "pension"       : ["pension"],
    "skill"         : ["skill", "employment"],
    "food"          : ["food", "ration"],
    "water"         : ["infrastructure"],
    "electricity"   : ["infrastructure"],
}


def map_to_categories(profile: UserProfile) -> List[str]:
    """
    Map extracted profile to relevant scheme categories.

    Priority order:
    1. specific_need (most direct signal)
    2. occupation (strong signal)
    3. gender = female (adds women category)
    4. is_bpl = True (adds welfare categories)

    Returns list of relevant category strings.
    Empty list means no filter — search everything.
    """
    categories = set()

    # Map specific need
    if profile.specific_need:
        need_lower = profile.specific_need.lower()
        for need, cats in NEED_TO_CATEGORIES.items():
            if need in need_lower:
                categories.update(cats)

    # Map occupation
    if profile.occupation:
        occ_lower = profile.occupation.lower()
        for occ, cats in OCCUPATION_TO_CATEGORIES.items():
            if occ in occ_lower:
                categories.update(cats)

    # Women always get women category added
    if profile.gender == "female":
        categories.add("women")
    if profile.occupation and "woman" in profile.occupation.lower():
        categories.add("women")

    # BPL users get welfare categories
    if profile.is_bpl:
        categories.update(["housing", "health", "food"])

    # Students get education
    if profile.education and profile.occupation == "student":
        categories.update(["education", "scholarship"])

    return list(categories)


# ── LLM setup ─────────────────────────────────────────────────────
def load_extraction_llm() -> ChatGroq:
    """
    Load LLM for profile extraction.
    temperature=0 — we want consistent, deterministic extraction.
    """
    return ChatGroq(
        model       = GROQ_MODEL,
        api_key     = os.environ.get("GROQ_API_KEY"),
        temperature = 0,
        max_tokens  = 500,
    )


# ── Build extraction chain ────────────────────────────────────────
def build_extraction_chain():
    """
    Build LangChain chain for profile extraction.

    Uses PydanticOutputParser — same as Week 3.
    Parser injects format instructions into prompt.
    LLM returns JSON → parser validates → UserProfile object.
    """
    parser = PydanticOutputParser(pydantic_object=UserProfile)
    llm    = load_extraction_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an information extraction assistant.\n"
            "Extract user profile information from the query below.\n"
            "Only extract what is explicitly mentioned.\n"
            "Return null for anything not mentioned.\n\n"
            "EXAMPLES for income extraction:\n"
            "- 'income below 2 lakhs' → annual_income_lakhs: 2.0\n"
            "- 'earning under 3 lakh' → annual_income_lakhs: 3.0\n"
            "- 'less than 1.5 lakh income' → annual_income_lakhs: 1.5\n"
            "- 'annual income 5 lakhs' → annual_income_lakhs: 5.0\n"
            "- 'BPL family' → is_bpl: true\n\n"
            "{format_instructions}"
        ),
        (
            "user",
            """Extract profile from this query:
"{query}"

Return only the JSON object."""
        )
    ]).partial(format_instructions=parser.get_format_instructions())

    # LCEL chain — same pattern as Week 3
    chain = prompt | llm | parser
    return chain


# ── Main extraction function ──────────────────────────────────────
def extract_profile(query: str) -> UserProfile:
    """
    Extract structured profile from a natural language query.

    Uses LLM with PydanticOutputParser for structured output.
    Falls back to empty profile if extraction fails.

    Args:
        query: User's natural language question

    Returns:
        UserProfile Pydantic object with extracted fields
    """
    try:
        chain   = build_extraction_chain()
        profile = chain.invoke({"query": query})
        return profile

    except Exception as e:
        # Graceful fallback — return empty profile
        # RAG still works without profile, just less filtered
        print(f"  ⚠️  Profile extraction failed: {e}")
        print(f"  Continuing with unfiltered search...")
        return UserProfile()


# ── Build enhanced query ──────────────────────────────────────────
def build_enhanced_query(query: str, profile: UserProfile) -> str:
    """
    Build enhanced query from profile fields ONLY.
    Does NOT repeat original query to avoid duplication.
    If no profile extracted, returns original query unchanged.
    """
    parts = []

    if profile.occupation:
        parts.append(f"for {profile.occupation}")
    if profile.specific_need:
        parts.append(f"{profile.specific_need} scheme")
    if profile.state:
        parts.append(f"in {profile.state}")
    if profile.gender:
        parts.append(f"for {profile.gender}")
    if profile.age:
        parts.append(f"aged {profile.age}")
    if profile.land_acres:
        parts.append(f"with {profile.land_acres} acres land")
    if profile.annual_income_lakhs:
        parts.append(f"income below {profile.annual_income_lakhs} lakh")
    if profile.is_bpl:
        parts.append("BPL below poverty line")
    if profile.category:
        parts.append(f"{profile.category} category")
    if profile.education:
        parts.append(f"{profile.education} education")

    if parts:
        return " ".join(parts)

    return query


# ── Format profile for display ────────────────────────────────────
def format_profile_display(profile: UserProfile) -> dict:
    """
    Convert UserProfile to a clean dict for Streamlit display.
    Filters out None values — only show what we know.
    """
    display = {}
    field_labels = {
        "occupation"          : "Occupation",
        "age"                 : "Age",
        "gender"              : "Gender",
        "state"               : "State",
        "annual_income_lakhs" : "Annual Income (Lakhs)",
        "land_acres"          : "Land Holding (Acres)",
        "category"            : "Social Category",
        "education"           : "Education",
        "is_bpl"              : "BPL Status",
        "specific_need"       : "Specific Need",
    }

    for field, label in field_labels.items():
        value = getattr(profile, field, None)
        if value is not None:
            if field == "is_bpl":
                display[label] = "Yes — Below Poverty Line" if value else "No"
            else:
                display[label] = str(value)

    return display


# ── Run directly ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SCHEMESEEKER — Eligibility Extraction Test")
    print("=" * 60 + "\n")

    test_queries = [
        # Simple
        "I am a farmer in UP",
        # Detailed English
        "I am a 35 year old woman farmer in Maharashtra "
        "with 2 acres of land and annual income below 2 lakhs",
        # Student
        "I am a 20 year old SC student looking for scholarship",
        # BPL family
        "We are a BPL family in Bihar, need housing assistance",
        # Entrepreneur
        "I want to start a small business, need loan under 10 lakhs",
        # Vague — should return empty profile
        "What schemes are available?",
        # Hindi
        "मैं एक किसान हूँ, मुझे सरकारी सहायता चाहिए",
    ]

    for query in test_queries:
        print(f"{'─' * 60}")
        print(f"Query   : {query}")

        profile    = extract_profile(query)
        categories = map_to_categories(profile)
        enhanced   = build_enhanced_query(query, profile)
        display    = format_profile_display(profile)

        print(f"\nExtracted Profile:")
        if display:
            for k, v in display.items():
                print(f"  {k:<25}: {v}")
        else:
            print("  (no profile details extracted)")

        print(f"\nCategories : {categories if categories else ['none — search all']}")
        print(f"Enhanced Q : {enhanced[:100]}")
        print()

    print("=" * 60)
    print("  ✅ Eligibility extraction working!")
    print("  Next → python utils/helpers.py")
    print("=" * 60 + "\n")