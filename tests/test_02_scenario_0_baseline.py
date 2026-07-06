"""
Test 02 — Scenario 0 baseline (no cross-session memory)

Goal:
1) Run session 1 and session 2 for client_a
2) Show that session 1 IS stored in SQLite
3) Show that session 2 working memory does NOT include session 1
4) Show client_b isolation

Run (from project root):
    uv run python -m tests.test_02_scenario_0_baseline

Requires:
- Ollama running locally
- Model from .env (default: llama3.2)
"""

from app.agent.agent import generate_reply
from app.db.database import SessionLocal, create_session, get_session_messages, init_db, save_turn
from tests.helpers import print_header, show_profile_summary, show_sqlite_storage, show_working_memory


def run_chat_turn(db, session_id: str, client_id: str, user_message: str) -> str:
    profile = show_profile_summary(client_id)

    stored_messages = get_session_messages(db, session_id)
    session_messages = [{"role": m.role, "content": m.content} for m in stored_messages]
    session_messages.append({"role": "user", "content": user_message})

    show_working_memory(profile, session_messages)

    reply = generate_reply(profile, session_messages)
    save_turn(db, session_id, client_id, user_message, reply)

    print("\nAGENT REPLY:")
    print(reply)
    return reply


def main() -> None:
    print_header("TEST 02 — SCENARIO 0 BASELINE")

    init_db()
    db = SessionLocal()

    try:
        # Step 1: Session 1 (client_a)
        print_header("STEP 1 — Session 1 (client_a): refinancing")
        session_1 = create_session(db, "client_a")
        print(f"session_id: {session_1.id}")
        run_chat_turn(
            db,
            session_1.id,
            "client_a",
            "I'm thinking about refinancing my home loan. What should I consider?",
        )

        # Step 2: Session 2 (client_a) recall test
        print_header("STEP 2 — Session 2 (client_a): recall test")
        session_2 = create_session(db, "client_a")
        print(f"session_id: {session_2.id}")
        run_chat_turn(
            db,
            session_2.id,
            "client_a",
            "What did we discuss about my home loan last time?",
        )

        print_header("CHECK — SQLite still has Session 1 data")
        show_sqlite_storage(db, "client_a")

        print_header("EXPECTED RESULT FOR STEP 2")
        print("- Session 1 messages should still exist in SQLite")
        print("- Session 2 working memory should NOT include Session 1 messages")
        print("- Agent should say it cannot recall prior session")

        # Step 3: Client B isolation
        print_header("STEP 3 — Session 1 (client_b): isolation test")
        session_b = create_session(db, "client_b")
        print(f"session_id: {session_b.id}")
        run_chat_turn(
            db,
            session_b.id,
            "client_b",
            "What is my risk tolerance and income situation?",
        )

        print_header("EXPECTED RESULT FOR STEP 3")
        print("- Reply should describe client_b only (conservative, self-employed)")
        print("- No client_a refinancing context should appear")

        print_header("DONE")
        print("Record your observations in README.md or your own notes.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
