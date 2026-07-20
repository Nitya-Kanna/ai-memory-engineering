"""
Test 07 — End-to-end episodic recall (real agent, real LLM)

Gap this closes:
    test_04/05/06 only ever tested Chroma directly (store_episode/search_episodes
    called by hand). Nothing proved that retrieval actually changes what the
    real agent SAYS. This test wires the two halves together and runs the real
    LLM (Ollama), mirroring test_02's exact scenario so the result is a direct
    A/B comparison:
        test_02 (no episodic memory) -> agent CANNOT recall session 1 in session 2
        test_07 (episodic memory)    -> agent CAN recall session 1 in session 2

Goal:
1) Session 1 (client_a): state a concrete fact (rate, bank, balance). Save the
   turn to SQLite (transcript) AND Chroma (episodic index).
2) Session 2 (client_a), brand-new session_id -> working memory is empty again,
   same as test_02. BEFORE calling the LLM, search Chroma for this new
   question and inject whatever comes back into the system prompt.
3) Print: the retrieved hits, the exact assembled prompt, and the real reply.
4) Client_b: same question style, prove episodic search never leaks client_a.

Requires:
- Ollama running locally with the model from .env (default: llama3.2)

Run (from project root):
    uv run python -m tests.test_07_episodic_end_to_end
"""

from __future__ import annotations

from app.agent.agent import build_messages, generate_reply
from app.db.database import SessionLocal, create_session, get_session_messages, init_db, save_turn
from app.memory.episodic import search_episodes, store_episode
from tests.helpers import print_header, print_subheader, show_profile_summary, show_sqlite_storage

TEST_CHROMA_PATH = "./chroma_data_test"
MAX_DISTANCE = 0.9  # same cutoff calibrated in test_06


def format_episodic_context(hits: list[dict]) -> str | None:
    """Turn Chroma hits into the text block injected into the system prompt."""
    if not hits:
        return None
    return "\n".join(f"- {h['text']}" for h in hits)


def run_chat_turn(db, session_id: str, client_id: str, user_message: str) -> str:
    profile = show_profile_summary(client_id)

    # --- Retrieval happens BEFORE the LLM is called ---
    hits = search_episodes(
        user_message, client_id, n_results=3, max_distance=MAX_DISTANCE, path=TEST_CHROMA_PATH
    )
    print_subheader("EPISODIC RETRIEVAL for this turn")
    if hits:
        for h in hits:
            preview = h["text"][:90].replace("\n", " ")
            print(f"  distance={h['distance']:.4f}  session={h['session_id']}  text={preview}")
    else:
        print("  (no relevant episodes found)")

    episodic_context = format_episodic_context(hits)

    stored_messages = get_session_messages(db, session_id)
    session_messages = [{"role": m.role, "content": m.content} for m in stored_messages]
    session_messages.append({"role": "user", "content": user_message})

    print_subheader("EXACT PROMPT SENT TO THE MODEL")
    for index, message in enumerate(build_messages(profile, session_messages, episodic_context), start=1):
        print(f"\n[{index}] role={message['role']}")
        print(message["content"])

    reply = generate_reply(profile, session_messages, episodic_context)

    # --- Persist AFTER the LLM replies: transcript (SQLite) + index (Chroma) ---
    save_turn(db, session_id, client_id, user_message, reply)
    store_episode(client_id, session_id, user_message, reply, path=TEST_CHROMA_PATH)

    print("\nAGENT REPLY:")
    print(reply)
    return reply


def main() -> None:
    print_header("TEST 07 — END-TO-END EPISODIC RECALL")

    init_db()
    db = SessionLocal()

    try:
        # Step 1: Session 1 (client_a) - plant the fact
        print_header("STEP 1 — Session 1 (client_a): state the fact")
        session_1 = create_session(db, "client_a")
        print(f"session_id: {session_1.id}")
        run_chat_turn(
            db,
            session_1.id,
            "client_a",
            "I'm thinking about refinancing my home loan. My current rate is 4.2 percent "
            "with Maybank, and my outstanding balance is around 420000.",
        )

        # Step 2: Session 2 (client_a) - recall test, fresh working memory
        print_header("STEP 2 — Session 2 (client_a): recall test (NEW session)")
        session_2 = create_session(db, "client_a")
        print(f"session_id: {session_2.id}")
        reply_2 = run_chat_turn(
            db,
            session_2.id,
            "client_a",
            "What did we discuss about my home loan last time?",
        )

        print_header("CHECK — did the reply recall the planted fact?")
        must_any = ["4.2", "maybank", "420000", "420,000"]
        lowered = reply_2.lower()
        found = [kw for kw in must_any if kw.lower() in lowered]
        if found:
            print(f"PASS (soft check): reply contains {found}")
        else:
            print("WARN: none of the expected keywords found in the reply text.")
            print("      This does NOT necessarily mean recall failed — the model may have")
            print("      paraphrased (e.g. 'just above four percent'). Read the reply above yourself.")

        print_header("CHECK — SQLite still holds both sessions separately")
        show_sqlite_storage(db, "client_a")

        # Step 3: client_b isolation
        print_header("STEP 3 — Session 1 (client_b): isolation test")
        session_b = create_session(db, "client_b")
        print(f"session_id: {session_b.id}")
        run_chat_turn(
            db,
            session_b.id,
            "client_b",
            "What is my risk tolerance and income situation?",
        )

        print_header("CHECK — episodic isolation (hard check)")
        leak_hits = search_episodes(
            "What did we discuss about my home loan last time?",
            "client_b",
            n_results=3,
            max_distance=MAX_DISTANCE,
            path=TEST_CHROMA_PATH,
        )
        leaked = any(h["client_id"] != "client_b" for h in leak_hits)
        if leaked:
            print("FAIL: client_b search returned a client_a episode")
            raise SystemExit(1)
        print(f"PASS: client_b search returned {len(leak_hits)} hit(s), none from client_a")

        print_header("DONE")
        print("Compare reply_2 above against test_02's session-2 reply:")
        print("  test_02 (no episodic memory) -> agent said it could not recall")
        print("  test_07 (episodic memory)    -> agent's reply is printed above; read it")

    finally:
        db.close()


if __name__ == "__main__":
    main()
