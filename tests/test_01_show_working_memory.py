"""
Test 01 — Show working memory

Goal: see exactly what the agent sees BEFORE we run Scenario 0.

Run (from project root):
    uv run python -m tests.test_01_show_working_memory
"""

from tests.helpers import print_header, show_profile_summary, show_working_memory


def main() -> None:
    print_header("TEST 01 — SHOW WORKING MEMORY")

    profile = show_profile_summary("client_a")

    # Simulate one turn in a session
    session_messages = [
        {
            "role": "user",
            "content": "I'm thinking about refinancing my home loan. What should I consider?",
        }
    ]

    show_working_memory(profile, session_messages)

    print_header("KEY TAKEAWAY")
    print("Working memory = system rules + profile + THIS session messages only.")
    print("history_seed is stored in JSON but intentionally excluded in Phase 0.")
    print("Cross-session chat is NOT included here.")


if __name__ == "__main__":
    main()
