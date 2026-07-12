"""
Test 03 — Session 1 interactive (see working memory live)

Goal:
1) Start ONE session (Session 1) for client_a
2) You type prompts yourself
3) Before each reply, print working memory (what the agent sees)
4) After each reply, save the turn to SQLite

This is still Phase 0: working memory = profile + THIS session only.

Run (from project root):
    uv run python -m tests.test_03_session1_interactive

Requires:
- Ollama running locally
- Model from .env (default: llama3.2)

Commands while chatting:
- empty line or 'quit' / 'exit' / 'q' → end session
"""

from app.agent.agent import generate_reply
from app.db.database import SessionLocal, create_session, get_session_messages, init_db, save_turn
from tests.helpers import print_header, show_profile_summary, show_sqlite_storage, show_working_memory


def main() -> None:
    print_header("TEST 03 — SESSION 1 INTERACTIVE")
    print("Type a message to chat. Type 'quit' to stop.")
    print("Before every reply, you will see WORKING MEMORY.")

    init_db()
    db = SessionLocal()

    try:
        profile = show_profile_summary("client_a")
        session = create_session(db, "client_a")
        print(f"\nsession_id: {session.id}  (this is Session 1)")

        turn = 0
        while True:
            print()
            user_message = input("YOU > ").strip()
            if not user_message or user_message.lower() in {"quit", "exit", "q"}:
                break

            turn += 1
            print_header(f"TURN {turn}")

            stored_messages = get_session_messages(db, session.id)
            session_messages = [{"role": m.role, "content": m.content} for m in stored_messages]
            session_messages.append({"role": "user", "content": user_message})

            # Show exactly what the agent will see THIS turn
            show_working_memory(profile, session_messages)

            print("\nCalling Ollama...")
            reply = generate_reply(profile, session_messages)
            save_turn(db, session.id, "client_a", user_message, reply)

            print("\nAGENT REPLY:")
            print(reply)

        print_header("CHECK — what was saved in SQLite for this Session 1")
        show_sqlite_storage(db, "client_a")

        print_header("KEY TAKEAWAY")
        print("- Working memory grew with each turn IN this session")
        print("- Everything above was still Session 1 only")
        print("- A new session later would NOT include these messages (Phase 0)")

    finally:
        db.close()


if __name__ == "__main__":
    main()
