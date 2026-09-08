# app.py
"""
SCHEMESEEKER — Main Streamlit App

Ties everything together:
  core/ingestion.py   → loads ChromaDB
  core/eligibility.py → extracts user profile
  core/chain.py       → RAG chain + streaming
  utils/helpers.py    → formatting + display

UI Structure:
  Sidebar → user profile (auto-built) + session stats + settings
  Main    → chat interface with streaming responses + source citations
"""

import streamlit as st
import time
from pathlib import Path

# ── Page config — must be first ───────────────────────────────────
st.set_page_config(
    page_title = "SchemeSeeker — Indian Govt Schemes",
    page_icon  = "🇮🇳",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    #MainMenu { visibility: hidden; }
    header   { visibility: hidden; }
    footer   { visibility: hidden; }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #1e3a5f, #1d4ed8);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #1e293b;
        color: #e2e8f0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        border-left: 3px solid #10b981;
    }

    /* Profile card */
    .profile-item {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid #1e293b;
        font-size: 13px;
    }
    .profile-label { color: #64748b; }
    .profile-value { color: #e2e8f0; font-weight: 500; }

    /* Category badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
        font-weight: 500;
    }

    /* Source box */
    .source-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        color: #94a3b8;
        margin: 4px 0;
    }

    /* Stat metric */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px;
    }

    /* Divider */
    hr { border-color: #1e293b; }

    /* Input */
    .stChatInput input {
        background: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Imports (after page config) ───────────────────────────────────
from core.ingestion   import run_ingestion, vectorstore_exists
from core.chain       import ask_streaming, load_vectorstore
from core.eligibility import extract_profile, map_to_categories, format_profile_display, build_enhanced_query
from utils.helpers    import (
    get_language_label,
    get_ui_text,
    format_profile_for_sidebar,
    format_sources,
    format_categories,
    get_starter_questions,
    add_to_history,
    get_session_stats,
)


# ── Session state init ────────────────────────────────────────────
def init_state():
    defaults = {
        "vectorstore"     : None,   # ChromaDB instance
        "chat_history"    : [],     # list of message dicts
        "profile"         : None,   # latest UserProfile
        "profile_display" : {},     # formatted profile for sidebar
        "categories"      : [],     # current relevant categories
        "language"        : "english",
        "initialized"     : False,  # vectorstore loaded flag
        "total_searches"  : 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ── Load vectorstore once ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """
    Load ChromaDB — cached across all reruns.

    @st.cache_resource caches the return value permanently
    for the lifetime of the Streamlit session.
    The vectorstore is only loaded ONCE even though
    Streamlit reruns the script on every interaction.

    If vectorstore doesn't exist → runs full ingestion.
    If it exists → loads in ~2 seconds.
    """
    if not vectorstore_exists():
        with st.spinner("🚀 First time setup — building scheme database... (~5 min)"):
            return run_ingestion()
    else:
        with st.spinner("📂 Loading scheme database..."):
            return load_vectorstore()


# ── Sidebar ───────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:

        # Header
        st.markdown("""
        <div style='text-align:center; padding:10px 0 16px;'>
            <div style='font-size:40px;'>🇮🇳</div>
            <h2 style='color:#10b981; margin:4px 0;'>SchemeSeeker</h2>
            <p style='color:#475569; font-size:12px; margin:0;'>
                Indian Government Schemes Assistant
            </p>
            <p style='color:#334155; font-size:11px; margin:4px 0;'>
                Powered by Groq · RAG · 2000+ Schemes
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── User Profile ──────────────────────────────────────────
        lang = st.session_state.language
        ui   = get_ui_text(lang)

        st.markdown(f"### 👤 {ui['profile_title']}")

        profile_display = st.session_state.profile_display

        if profile_display:
            items = format_profile_for_sidebar(profile_display)
            for item in items:
                st.markdown(f"""
                <div class='profile-item'>
                    <span class='profile-label'>{item['icon']} {item['label']}</span>
                    <span class='profile-value'>{item['value']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"<p style='color:#475569; font-size:13px;'>{ui['no_profile']}</p>",
                unsafe_allow_html=True
            )

        # ── Categories ────────────────────────────────────────────
        if st.session_state.categories:
            st.markdown(f"\n**{ui['categories']}:**")
            badges = format_categories(st.session_state.categories)
            badge_html = " ".join([
                f"<span class='badge' style='background:{b['bg']}; color:{b['color']};'>"
                f"{b['icon']} {b['label']}</span>"
                for b in badges
            ])
            st.markdown(badge_html, unsafe_allow_html=True)

        st.divider()

        # ── Session Stats ─────────────────────────────────────────
        st.markdown("### 📊 Session Stats")
        stats = get_session_stats(st.session_state.chat_history)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", stats["Total Questions"])
            st.metric("Hindi", stats["Hindi Queries"])
        with col2:
            st.metric("Searches", st.session_state.total_searches)
            st.metric("English", stats["English Queries"])

        st.divider()

        # ── Language indicator ────────────────────────────────────
        st.markdown("### 🌐 Language")
        st.markdown(
            f"<p style='color:#e2e8f0;'>{get_language_label(lang)}</p>",
            unsafe_allow_html=True
        )
        st.caption("Auto-detected from your query")

        st.divider()

        # ── Actions ───────────────────────────────────────────────
        st.markdown("### ⚙️ Actions")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.profile      = None
            st.session_state.profile_display = {}
            st.session_state.categories   = []
            st.session_state.total_searches = 0
            st.rerun()

        # About
        with st.expander("ℹ️ About SchemeSeeker"):
            st.markdown("""
            **SchemeSeeker** helps Indian citizens discover
            government welfare schemes they are eligible for.

            **How it works:**
            1. Ask your question in Hindi or English
            2. AI extracts your profile automatically
            3. Searches 2000+ scheme Q&A pairs
            4. Answers based only on official scheme data

            **Data source:** BharatSchemes dataset covering
            Central + State schemes across all 28 states.

            ⚠️ Always verify scheme details at official
            government websites before applying.
            """)


# ── Main chat area ────────────────────────────────────────────────
def render_chat(vectorstore):
    """Render the main chat interface."""

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <h1 style='color:#10b981; margin:0; font-size:26px;'>
            🇮🇳 SchemeSeeker
        </h1>
        <p style='color:#475569; font-size:14px; margin:4px 0;'>
            Ask about Indian government schemes in Hindi or English
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Starter questions (empty state) ───────────────────────────
    if not st.session_state.chat_history:
        st.markdown("""
        <p style='color:#64748b; font-size:14px; margin:16px 0 8px;'>
        💡 Try one of these:
        </p>
        """, unsafe_allow_html=True)

        starters = get_starter_questions()
        col1, col2 = st.columns(2)

        for i, q in enumerate(starters):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.button(q, key=f"starter_{i}", use_container_width=True):
                    process_query(q, vectorstore)
                    st.rerun()

        st.divider()

    # ── Chat history ──────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        role = msg["role"]

        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg["content"])

        else:
            with st.chat_message("assistant", avatar="🇮🇳"):
                st.markdown(msg["content"])

                # Show sources in expander
                sources = msg.get("sources", [])
                if sources:
                    clean_sources = format_sources(sources)
                    with st.expander(
                        f"📚 Sources ({len(clean_sources)} retrieved)",
                        expanded=False
                    ):
                        for i, src in enumerate(clean_sources[:3], 1):
                            st.markdown(f"""
                            <div class='source-box'>
                                <strong>[{i}]</strong> {src['preview']}...
                            </div>
                            """, unsafe_allow_html=True)
                        st.caption(f"Source: {clean_sources[0]['source']}")

    # ── Chat input ─────────────────────────────────────────────────
    lang = st.session_state.language
    ui   = get_ui_text(lang)

    if prompt := st.chat_input(ui["placeholder"]):
        process_query(prompt, vectorstore)


# ── Process user query ────────────────────────────────────────────
def process_query(query: str, vectorstore):
    """
    Handle a user query end-to-end:
      1. Display user message
      2. Extract profile + update sidebar
      3. Build enhanced query
      4. Stream response from RAG chain
      5. Save to history
    """
    # Step 1: Show user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # Step 2: Extract profile (background)
    with st.spinner("🔍 Analysing your query..."):
        profile    = extract_profile(query)
        categories = map_to_categories(profile)
        enhanced_q = build_enhanced_query(query, profile)
        prof_display = format_profile_display(profile)

    # Update session state profile
    st.session_state.profile         = profile
    st.session_state.profile_display = prof_display
    st.session_state.categories      = categories

    # Detect language from query
    from core.chain import detect_language
    language = detect_language(query)
    st.session_state.language = language

    # Add user message to history
    st.session_state.chat_history = add_to_history(
        st.session_state.chat_history,
        role     = "user",
        content  = query,
        language = language,
    )

    # Step 3: Stream response — single retrieval, single LLM call
    with st.chat_message("assistant", avatar="🇮🇳"):
        response_placeholder = st.empty()
        full_response        = ""
        sources              = []
        language             = st.session_state.language

        for chunk in ask_streaming(enhanced_q, vectorstore, language=language):
            if isinstance(chunk, dict) and "__sources__" in chunk:
                sources  = chunk["__sources__"]
                language = chunk["__language__"]
                st.session_state.language = language
            else:
                full_response += chunk
                response_placeholder.markdown(
                    full_response + "▊",
                    unsafe_allow_html=False
                )

        # Final response without cursor
        response_placeholder.markdown(full_response)

        # Show sources
        if sources:
            clean_sources = format_sources(sources)
            with st.expander(
                f"📚 Sources ({len(clean_sources)} retrieved)",
                expanded=False
            ):
                for i, src in enumerate(clean_sources[:3], 1):
                    st.markdown(f"""
                    <div class='source-box'>
                        <strong>[{i}]</strong> {src['preview']}...
                    </div>
                    """, unsafe_allow_html=True)
                if clean_sources:
                    st.caption(f"Source: {clean_sources[0]['source']}")

    # Step 4: Save to history
    st.session_state.chat_history = add_to_history(
        st.session_state.chat_history,
        role     = "assistant",
        content  = full_response,
        language = language,
        sources  = sources,
    )
    st.session_state.total_searches += 1

  
# ── Main ──────────────────────────────────────────────────────────
def main():
    # Load vectorstore once
    vectorstore = get_vectorstore()

    if vectorstore is None:
        st.error("Failed to load scheme database. Please check your setup.")
        return

    # Render UI
    render_sidebar()
    render_chat(vectorstore)


if __name__ == "__main__":
    main()