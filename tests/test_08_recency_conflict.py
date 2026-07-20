"""
Test 08 — Recency vs. relevance conflict (episodic memory)

Gap this closes:
    search_episodes ranks purely by embedding distance. It has no concept of
    "which fact is current" — if a client's rate changed, the OLD and NEW
    statements are both "about the rate," and cosine distance alone cannot
    tell them apart. Until this test, episodes weren't even timestamped in
    Chroma (added created_at to store_episode/search_episodes for this).

Goal:
1) Plant an OLD fact (rate = 4.2%, backdated ~90 days) and a NEWER fact
   (rate changed to 5.1%, backdated ~2 days) for client_a.
2) Step A — show the raw gap: query "what's my current rate?" with plain
   search_episodes(). Print both hits' distance AND created_at side by side.
   Whichever ranks first by distance is not guaranteed to be the newer one —
   that's the bug. (We print this either way, we don't assert on it, because
   which one happens to win by distance depends on wording, not on a real bug
   we control — the point is distance alone never even LOOKS at the date.)
3) Step B — the fix: format BOTH hits into the prompt, each tagged with its
   date, and rely on the updated AGENT_SYSTEM_PROMPT_WITH_MEMORY rule
   ("trust the entry with the LATER date"). Run the real agent and check the
   reply states 5.1% as current, not 4.2%.

Requires:
- Ollama running locally with the model from .env (default: llama3.2)

Run (from project root):
    uv run python -m tests.test_08_recency_conflict
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.agent.agent import build_messages, generate_reply
from app.memory.episodic import clear_episodes, search_episodes, store_episode
from tests.helpers import print_header, print_subheader, show_profile_summary

TEST_CHROMA_PATH = "./chroma_data_test"
MAX_DISTANCE = 0.9

NOW = datetime.now(timezone.utc)
OLD_DATE = (NOW - timedelta(days=90)).isoformat()
NEW_DATE = (NOW - timedelta(days=2)).isoformat()


def format_dated_context(hits: list[dict]) -> str | None:
    """Sort oldest -> newest so the conflict is easy for the model to read
    chronologically, and label each entry with its date explicitly."""
    if not hits:
        return None
    ordered = sorted(hits, key=lambda h: h["created_at"] or "")
    lines = []
    for h in ordered:
        date_only = (h["created_at"] or "unknown")[:10]
        lines.append(f"- [as of {date_only}] {h['text']}")
    return "\n".join(lines)


def main() -> None:
    print_header("TEST 08 — RECENCY VS. RELEVANCE CONFLICT")

    clear_episodes(path=TEST_CHROMA_PATH)
    print(f"OLD_DATE (90 days ago): {OLD_DATE[:10]}")
    print(f"NEW_DATE (2 days ago):  {NEW_DATE[:10]}")

    print_header("STEP 1 — Plant conflicting facts for client_a")
    store_episode(
        "client_a",
        "session_old",
        "My home loan rate is 4.2 percent with Maybank.",
        "Noted — 4.2 percent with Maybank.",
        created_at=OLD_DATE,
        path=TEST_CHROMA_PATH,
    )
    store_episode(
        "client_a",
        "session_new",
        "Quick update — I refinanced and my new rate is 5.1 percent.",
        "Got it, updating your rate to 5.1 percent.",
        created_at=NEW_DATE,
        path=TEST_CHROMA_PATH,
    )
    print("Stored: OLD fact (4.2%, 90 days ago) + NEW fact (5.1%, 2 days ago)")

    query = "What's my current home loan rate?"

    print_header("STEP 2 — Raw retrieval, no date-awareness (shows the gap)")
    hits = search_episodes(query, "client_a", n_results=3, max_distance=MAX_DISTANCE, path=TEST_CHROMA_PATH)
    for h in hits:
        print(f"  distance={h['distance']:.4f}  created_at={h['created_at'][:10]}  text={h['text'][:70]}")
    if hits and hits[0]["created_at"] == OLD_DATE:
        print("\n  -> Top hit by distance is the OLDER fact. Distance alone got this backwards.")
    elif hits and hits[0]["created_at"] == NEW_DATE:
        print("\n  -> Top hit by distance happens to be the newer one here, but that's luck of")
        print("     the wording, not something distance ranking guarantees in general.")

    print_header("STEP 3 — Fix: date-tagged context + recency-aware prompt rule")
    profile = show_profile_summary("client_a")
    episodic_context = format_dated_context(hits)

    print_subheader("EXACT PROMPT SENT TO THE MODEL")
    for index, message in enumerate(build_messages(profile, [], episodic_context), start=1):
        print(f"\n[{index}] role={message['role']}")
        print(message["content"])

    reply = generate_reply(profile, [{"role": "user", "content": query}], episodic_context)
    print("\nAGENT REPLY:")
    print(reply)

    print_header("CHECK — does the reply correctly say 5.1% is current?")
    lowered = reply.lower()
    has_new = "5.1" in lowered
    has_old_as_current = "4.2" in lowered and "5.1" not in lowered
    if has_new:
        print("PASS (soft check): reply states 5.1% — the newer fact was preferred.")
    elif has_old_as_current:
        print("FAIL-ish: reply only mentions 4.2% — looks like it used the stale fact.")
        print("          Read the reply above to confirm.")
    else:
        print("WARN: neither rate found verbatim — read the reply above yourself.")

    print_header("DONE")
    print("This tests prompt-level recency handling (date tags + instruction),")
    print("not a distance/recency scoring blend in search_episodes itself —")
    print("that would be the next step up if this proves unreliable at scale.")


if __name__ == "__main__":
    main()
