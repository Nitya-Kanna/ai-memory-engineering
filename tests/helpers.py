"""Shared helpers for printing memory state during tests."""

from app.agent.agent import build_messages
from app.bank_client_profiles import BankClientProfile, load_bank_client_profile
from app.db.database import Message, Session


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def show_working_memory(
    profile: BankClientProfile,
    session_messages: list[dict[str, str]],
    episodic_context: str | None = None,
) -> None:
    """Print exactly what the agent sees this turn (working memory)."""
    print_subheader("WORKING MEMORY (what the agent sees this turn)")
    messages = build_messages(profile, session_messages, episodic_context)

    for index, message in enumerate(messages, start=1):
        print(f"\n[{index}] role={message['role']}")
        print(message["content"])

    print_subheader("NOT in working memory (Phase 0)")
    print("- Prior sessions (even if saved in SQLite)")
    print("- history_seed from profile JSON:")
    for item in profile.history_seed:
        print(f"  - {item}")


def show_sqlite_storage(db, client_id: str) -> None:
    """Print what is stored in SQLite (session transcript storage)."""
    print_subheader(f"SQLITE STORAGE for client_id={client_id}")

    sessions = (
        db.query(Session)
        .filter(Session.client_id == client_id)
        .order_by(Session.created_at)
        .all()
    )

    if not sessions:
        print("(no sessions yet)")
        return

    for session in sessions:
        print(f"\nSession: {session.id}")
        print(f"  created_at: {session.created_at}")

        messages = (
            db.query(Message)
            .filter(Message.session_id == session.id)
            .order_by(Message.created_at)
            .all()
        )

        if not messages:
            print("  (no messages)")
            continue

        for message in messages:
            preview = message.content.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  - [{message.role}] {preview}")


def show_profile_summary(client_id: str) -> BankClientProfile:
    profile = load_bank_client_profile(client_id)
    print_subheader(f"BANK CLIENT PROFILE ({client_id})")
    print(f"name: {profile.name}")
    print(f"risk_tolerance: {profile.risk_tolerance}")
    print(f"income_type: {profile.income_type}")
    print(f"goals: {profile.goals}")
    return profile


def show_episodic_hits(title: str, hits: list[dict]) -> None:
    """Print Chroma search results (episodic memory)."""
    print_subheader(title)
    if not hits:
        print("(no episodes found)")
        return

    for index, hit in enumerate(hits, start=1):
        print(f"\n[{index}] id={hit['id']}")
        print(f"    client_id={hit['client_id']}  session_id={hit['session_id']}")
        print(f"    distance={hit['distance']:.4f}  (lower = closer)")
        print(f"    text:\n{hit['text']}")
