# utils/helpers.py
"""
SCRIPT 5 — Helpers and Formatting

Utility functions used by app.py:
  - Format RAG answers for display
  - Format user profile for sidebar
  - Format source citations
  - Language-aware display helpers
  - Session state helpers
"""

from datetime import datetime
from core.eligibility import UserProfile


# ── Language helpers ──────────────────────────────────────────────
def get_language_label(language: str) -> str:
    """Return emoji + label for detected language."""
    return "🇮🇳 Hindi" if language == "hindi" else "🇬🇧 English"


def get_ui_text(language: str) -> dict:
    """
    Return UI labels in the correct language.
    Used to make the entire UI bilingual.

    When user asks in Hindi → sidebar labels, placeholders
    and captions also appear in Hindi.
    """
    if language == "hindi":
        return {
            "placeholder"  : "अपना प्रश्न यहाँ लिखें... (जैसे: मैं एक किसान हूँ, मुझे योजनाएं बताएं)",
            "profile_title": "आपकी प्रोफ़ाइल",
            "sources_title": "स्रोत",
            "no_profile"   : "प्रश्न पूछें — प्रोफ़ाइल अपने आप बनेगी",
            "thinking"     : "उत्तर ढूंढा जा रहा है...",
            "categories"   : "संबंधित श्रेणियाँ",
        }
    return {
        "placeholder"  : "Ask about schemes... (e.g. I am a farmer in UP, what schemes can I apply for?)",
        "profile_title": "Your Profile",
        "sources_title": "Sources",
        "no_profile"   : "Ask a question — profile builds automatically",
        "thinking"     : "Finding relevant schemes...",
        "categories"   : "Relevant Categories",
    }


# ── Profile formatting ────────────────────────────────────────────
def format_profile_for_sidebar(profile_display: dict) -> list[dict]:
    """
    Format profile dict into list of label-value pairs
    with icons for Streamlit sidebar display.

    Args:
        profile_display: dict from eligibility.format_profile_display()

    Returns:
        List of {icon, label, value} dicts
    """
    icon_map = {
        "Occupation"           : "💼",
        "Age"                  : "🎂",
        "Gender"               : "👤",
        "State"                : "📍",
        "Annual Income (Lakhs)": "💰",
        "Land Holding (Acres)" : "🌾",
        "Social Category"      : "🏷️",
        "Education"            : "🎓",
        "BPL Status"           : "📋",
        "Specific Need"        : "🎯",
    }

    items = []
    for label, value in profile_display.items():
        items.append({
            "icon" : icon_map.get(label, "•"),
            "label": label,
            "value": value,
        })
    return items


# ── Source formatting ─────────────────────────────────────────────
def format_sources(sources: list[dict]) -> list[dict]:
    """
    Clean and deduplicate sources for display.

    Each source shows:
    - Preview of the retrieved chunk
    - Original question from the dataset
    - Source attribution

    Args:
        sources: list of source dicts from chain.ask()

    Returns:
        Cleaned, deduplicated list for display
    """
    seen     = set()
    cleaned  = []

    for src in sources:
        preview = src.get("preview", "")[:200]

        # Skip exact duplicates
        if preview in seen:
            continue
        seen.add(preview)

        cleaned.append({
            "preview" : preview,
            "question": src.get("question", "")[:150],
            "source"  : src.get("source", "myscheme.gov.in"),
        })

    return cleaned


# ── Category badge formatting ─────────────────────────────────────
def format_categories(categories: list[str]) -> list[dict]:
    """
    Format category list into badge dicts with colors.

    Args:
        categories: list of category strings

    Returns:
        List of {label, color, icon} dicts
    """
    category_styles = {
        "agriculture" : {"color": "#166534", "bg": "#dcfce7", "icon": "🌾"},
        "health"      : {"color": "#991b1b", "bg": "#fee2e2", "icon": "🏥"},
        "housing"     : {"color": "#1e3a5f", "bg": "#dbeafe", "icon": "🏠"},
        "finance"     : {"color": "#713f12", "bg": "#fef3c7", "icon": "💰"},
        "education"   : {"color": "#312e81", "bg": "#ede9fe", "icon": "📚"},
        "women"       : {"color": "#831843", "bg": "#fce7f3", "icon": "👩"},
        "insurance"   : {"color": "#134e4a", "bg": "#ccfbf1", "icon": "🛡️"},
        "pension"     : {"color": "#374151", "bg": "#f3f4f6", "icon": "👴"},
        "skill"       : {"color": "#1e40af", "bg": "#dbeafe", "icon": "🎯"},
        "scholarship" : {"color": "#5b21b6", "bg": "#ede9fe", "icon": "🎓"},
        "msme"        : {"color": "#92400e", "bg": "#fef3c7", "icon": "🏭"},
        "startup"     : {"color": "#064e3b", "bg": "#d1fae5", "icon": "🚀"},
        "food"        : {"color": "#7c2d12", "bg": "#ffedd5", "icon": "🍚"},
        "labour"      : {"color": "#1f2937", "bg": "#f9fafb", "icon": "⚒️"},
    }

    badges = []
    for cat in categories:
        style = category_styles.get(
            cat.lower(),
            {"color": "#374151", "bg": "#f3f4f6", "icon": "📋"}
        )
        badges.append({
            "label": cat.title(),
            "icon" : style["icon"],
            "color": style["color"],
            "bg"   : style["bg"],
        })
    return badges


# ── Chat history helpers ──────────────────────────────────────────
def format_chat_history(history: list[dict]) -> list[dict]:
    """
    Format chat history for display.

    Each entry:
    {role, content, language, timestamp, sources}

    Args:
        history: list of message dicts from session_state

    Returns:
        Formatted list ready for st.chat_message()
    """
    formatted = []
    for msg in history:
        formatted.append({
            "role"     : msg.get("role", "user"),
            "content"  : msg.get("content", ""),
            "language" : msg.get("language", "english"),
            "timestamp": msg.get("timestamp", ""),
            "sources"  : msg.get("sources", []),
        })
    return formatted


def add_to_history(
    history  : list,
    role     : str,
    content  : str,
    language : str = "english",
    sources  : list = None,
) -> list:
    """
    Add a message to chat history with timestamp.

    Args:
        history : existing history list
        role    : "user" or "assistant"
        content : message text
        language: detected language
        sources : list of source dicts (for assistant messages)

    Returns:
        Updated history list
    """
    history.append({
        "role"     : role,
        "content"  : content,
        "language" : language,
        "timestamp": datetime.now().strftime("%H:%M"),
        "sources"  : sources or [],
    })
    return history


# ── Session stats ─────────────────────────────────────────────────
def get_session_stats(history: list) -> dict:
    """
    Calculate stats from chat history.
    Displayed in sidebar.
    """
    total_msgs    = len(history)
    user_msgs     = sum(1 for m in history if m["role"] == "user")
    hindi_queries = sum(
        1 for m in history
        if m["role"] == "user" and m.get("language") == "hindi"
    )

    return {
        "Total Questions"  : user_msgs,
        "Hindi Queries"    : hindi_queries,
        "English Queries"  : user_msgs - hindi_queries,
        "Total Messages"   : total_msgs,
    }


# ── Starter questions ─────────────────────────────────────────────
STARTER_QUESTIONS = [
    "I am a farmer in UP with 2 acres of land. What schemes am I eligible for?",
    "I am a woman from rural Maharashtra. What housing schemes can I apply for?",
    "I am a 20 year old SC student. What scholarships are available?",
    "We are a BPL family. What health insurance schemes can we get?",
    "I want to start a small business. What loans are available?",
    "किसान के लिए कौन सी योजनाएं हैं?",
]


def get_starter_questions() -> list[str]:
    """Return starter questions for empty chat state."""
    return STARTER_QUESTIONS