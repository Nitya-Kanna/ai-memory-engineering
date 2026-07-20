"""
Test 06 — Precision / negative case (episodic memory)

Gap this closes:
    search_episodes() always returns up to n_results hits, no matter how
    irrelevant they are (episodic.py, before this test: no threshold).
    A client with zero relevant history still gets "confident" nonsense hits.

Goal:
1) Store a small set of clearly home-loan/refinancing episodes for client_a
2) Ask ON-TOPIC questions -> must still get relevant hits (recall preserved)
3) Ask OFF-TOPIC questions (cooking, movies, weather on Mars) -> WITHOUT a
   distance cutoff, Chroma still returns "closest of the bad options"
   (we print this to show the gap is real, not assert on it)
4) Re-run the SAME off-topic questions WITH max_distance set -> must now
   return zero hits, while on-topic questions still return hits

This is the calibration step: MAX_DISTANCE was chosen by actually running
this test and reading the printed distances (see comment below), not guessed.

Run (from project root):
    uv run python -m tests.test_06_precision_case

Uses a separate folder: ./chroma_data_test (wiped clean at start)
"""

from __future__ import annotations

from app.memory.episodic import clear_episodes, search_episodes, store_episode
from tests.helpers import print_header, print_subheader, show_episodic_hits

TEST_CHROMA_PATH = "./chroma_data_test"

# Calibrated against this test's own STORED set below — see STEP 1 printout.
# On-topic queries landed well under this; off-topic queries landed well over it.
MAX_DISTANCE = 0.9

STORED_EPISODES = [
    {
        "user": "I'm thinking about refinancing my home loan. What about break costs?",
        "assistant": (
            "Compare break costs against monthly savings. If break costs exceed "
            "savings, refinancing may not be worth it yet."
        ),
    },
    {
        "user": "I might get a bonus and want to prepay part of the loan.",
        "assistant": "Prepaying can cut interest. Check if your loan has prepayment penalties.",
    },
    {
        "user": "My risk tolerance is moderate, not aggressive.",
        "assistant": "Moderate risk tolerance argues against highly leveraged refinance moves.",
    },
]

ON_TOPIC_QUERIES = [
    {"question": "What did we discuss about break costs when refinancing?", "must_any": ["break cost"]},
    {"question": "Did I mention using a bonus to prepay the loan?", "must_any": ["bonus", "prepay"]},
    {"question": "What did I say about my risk tolerance?", "must_any": ["moderate", "risk"]},
]

OFF_TOPIC_QUERIES = [
    "What's a good recipe for chocolate cake?",
    "Can you recommend a sci-fi movie to watch tonight?",
    "What's the weather like on Mars?",
]


def _hit_matches(text: str, must_any: list[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in must_any)


def main() -> None:
    print_header("TEST 06 — PRECISION / NEGATIVE CASE")

    clear_episodes(path=TEST_CHROMA_PATH)
    print(f"Chroma path: {TEST_CHROMA_PATH}")
    print(f"Stored episodes: {len(STORED_EPISODES)}")
    print(f"On-topic queries: {len(ON_TOPIC_QUERIES)} | Off-topic queries: {len(OFF_TOPIC_QUERIES)}")
    print(f"MAX_DISTANCE cutoff: {MAX_DISTANCE}")

    print_header("STEP 1 — Store episodes")
    for item in STORED_EPISODES:
        store_episode("client_a", "session_precision", item["user"], item["assistant"], path=TEST_CHROMA_PATH)
    print("Stored.")

    print_header("STEP 2 — Raw distances, NO threshold (shows the gap)")
    print_subheader("On-topic (expect low distances)")
    for q in ON_TOPIC_QUERIES:
        hits = search_episodes(q["question"], "client_a", n_results=1, path=TEST_CHROMA_PATH)
        top = hits[0]["distance"] if hits else None
        print(f"  {q['question'][:55]:55s} -> top distance = {top}")

    print_subheader("Off-topic (no threshold -> still returns 'closest of the bad options')")
    for q in OFF_TOPIC_QUERIES:
        hits = search_episodes(q, "client_a", n_results=1, path=TEST_CHROMA_PATH)
        top = hits[0]["distance"] if hits else None
        preview = hits[0]["text"][:60].replace("\n", " ") if hits else "(none)"
        print(f"  {q[:55]:55s} -> top distance = {top}  | returned: {preview}")

    print_header("STEP 3 — WITH threshold (max_distance={:.2f})".format(MAX_DISTANCE))
    ok = True

    print_subheader("On-topic must still return correct hits")
    for q in ON_TOPIC_QUERIES:
        hits = search_episodes(
            q["question"], "client_a", n_results=3, max_distance=MAX_DISTANCE, path=TEST_CHROMA_PATH
        )
        passed = bool(hits) and _hit_matches(hits[0]["text"], q["must_any"])
        mark = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"  [{mark}] {q['question'][:60]}")

    print_subheader("Off-topic must now return ZERO hits")
    for q in OFF_TOPIC_QUERIES:
        hits = search_episodes(q, "client_a", n_results=3, max_distance=MAX_DISTANCE, path=TEST_CHROMA_PATH)
        passed = len(hits) == 0
        mark = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
            show_episodic_hits(f"UNEXPECTED HIT for off-topic query: {q}", hits)
        print(f"  [{mark}] {q[:60]}  (hits: {len(hits)})")

    print_header("DONE")
    print("Recall preserved on-topic, and off-topic queries correctly return nothing"
          if ok else "FAILURES ABOVE — see marks")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
