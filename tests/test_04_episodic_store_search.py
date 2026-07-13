"""
Test 04 — Episodic memory store + search (Chroma)

Goal:
1) Store turns for client_a and client_b
2) Search for client_a → get A's refinance episode (actual words from Chroma)
3) Search for client_b → must NOT return A's episode
4) Show you do NOT need SQLite to get the recalled sentences

This is Step 2 only: store/search work.
Not wired into the agent / working memory yet.

Run (from project root):
    uv run python -m tests.test_04_episodic_store_search

Uses a separate folder: ./chroma_data_test (so real chroma_data stays clean)
"""

from app.memory.episodic import clear_episodes, search_episodes, store_episode
from tests.helpers import print_header, show_episodic_hits

TEST_CHROMA_PATH = "./chroma_data_test"


def main() -> None:
    print_header("TEST 04 — EPISODIC STORE + SEARCH")

    clear_episodes(path=TEST_CHROMA_PATH)
    print(f"Using Chroma path: {TEST_CHROMA_PATH} (wiped for a clean test)")

    # --- Store episodes (like Session 1 turns) ---
    print_header("STEP 1 — Store episodes")

    id_a = store_episode(
        client_id="client_a",
        session_id="session_a_1",
        user_message="I'm thinking about refinancing my home loan. What about break costs?",
        assistant_message=(
            "Compare break costs against monthly savings. "
            "If break costs exceed savings, refinancing may not be worth it yet."
        ),
        path=TEST_CHROMA_PATH,
    )
    print(f"stored client_a episode: {id_a}")

    id_a2 = store_episode(
        client_id="client_a",
        session_id="session_a_1",
        user_message="I might get a bonus and want to prepay part of the loan.",
        assistant_message="Prepaying can cut interest. Check if your loan has prepayment penalties.",
        path=TEST_CHROMA_PATH,
    )
    print(f"stored client_a episode: {id_a2}")

    id_b = store_episode(
        client_id="client_b",
        session_id="session_b_1",
        user_message="I am self-employed and prefer conservative investments.",
        assistant_message="Given your conservative risk tolerance, focus on capital preservation.",
        path=TEST_CHROMA_PATH,
    )
    print(f"stored client_b episode: {id_b}")

    # --- Search for client_a ---
    print_header("STEP 2 — Search as client_a (should recall refinance)")
    query_a = "What did we discuss about my home loan and break costs?"
    print(f"query: {query_a}")

    hits_a = search_episodes(
        query_a,
        client_id="client_a",
        n_results=3,
        path=TEST_CHROMA_PATH,
    )
    show_episodic_hits("EPISODIC HITS for client_a (text comes from Chroma)", hits_a)

    print_header("EXPECTED for client_a")
    print("- Top hit should mention refinancing / break costs")
    print("- All hits must have client_id=client_a")
    print("- Actual sentences are returned from Chroma (no SQLite lookup)")

    # --- Search for client_b ---
    print_header("STEP 3 — Search as client_b (isolation)")
    query_b = "What did we discuss about my home loan and break costs?"
    print(f"query: {query_b}  (same words, different client)")

    hits_b = search_episodes(
        query_b,
        client_id="client_b",
        n_results=3,
        path=TEST_CHROMA_PATH,
    )
    show_episodic_hits("EPISODIC HITS for client_b", hits_b)

    print_header("EXPECTED for client_b")
    print("- Must NOT return client_a's refinance episode")
    print("- If anything returns, it should be client_b only (conservative / self-employed)")

    # --- Simple pass/fail checks ---
    print_header("CHECKS")
    ok = True

    if not hits_a:
        print("FAIL: client_a got zero hits")
        ok = False
    else:
        top = hits_a[0]["text"].lower()
        if "refinanc" in top or "break cost" in top:
            print("PASS: client_a top hit looks like the refinance turn")
        else:
            print("WARN: client_a top hit may not be the refinance turn — read the printout")
            print(f"      top text preview: {hits_a[0]['text'][:120]}...")

        if any(h["client_id"] != "client_a" for h in hits_a):
            print("FAIL: client_a search leaked another client")
            ok = False
        else:
            print("PASS: all client_a hits are scoped to client_a")

    if any(h["client_id"] == "client_a" for h in hits_b):
        print("FAIL: client_b search returned a client_a episode")
        ok = False
    else:
        print("PASS: client_b search did not leak client_a")

    print_header("DONE")
    print("Step 2 complete: store + search work.")
    print("Next: wire store into save_turn, and inject search hits into working memory.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
