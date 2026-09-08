# test_helpers.py
from utils.helpers import (
    get_language_label,
    get_ui_text,
    format_categories,
    get_starter_questions,
    add_to_history,
    get_session_stats,
)

print("Test 1 — Language labels:")
print(f"  {get_language_label('hindi')}")
print(f"  {get_language_label('english')}")

print("\nTest 2 — UI text (Hindi):")
ui = get_ui_text("hindi")
for k, v in ui.items():
    print(f"  {k}: {v}")

print("\nTest 3 — Category badges:")
badges = format_categories(["agriculture", "health", "women", "finance"])
for b in badges:
    print(f"  {b['icon']} {b['label']}")

print("\nTest 4 — Starter questions:")
for q in get_starter_questions():
    print(f"  → {q}")

print("\nTest 5 — Chat history:")
history = []
history = add_to_history(history, "user", "I am a farmer", "english")
history = add_to_history(history, "assistant", "Here are schemes...", "english", [])
stats = get_session_stats(history)
print(f"  Stats: {stats}")

print("\n✅ All helpers working!")
print("   Next → python app.py")