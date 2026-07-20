"""
Episodic memory — searchable past turns in Chroma.

SQLite keeps the exact transcript.
Chroma keeps the same turn text + an embedding so we can find relevant past moments.

Phase 1 Step 2: store + search only (not wired into the agent yet).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import chromadb

from app.config import settings


def format_episode_text(user_message: str, assistant_message: str) -> str:
    """One episode = one turn (user + assistant), as plain text for embedding."""
    return f"User: {user_message}\nAssistant: {assistant_message}"


def _client(path: str | None = None) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=path or settings.chroma_path)


def get_episodes_collection(path: str | None = None):
    """Get or create the episodes collection on disk."""
    client = _client(path)
    return client.get_or_create_collection(
        name=settings.episodic_collection,
        metadata={"hnsw:space": "cosine"},
    )


def clear_episodes(path: str | None = None) -> None:
    """Delete and recreate the episodes collection (useful for tests)."""
    client = _client(path)
    name = settings.episodic_collection
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def store_episode(
    client_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
    *,
    episode_id: str | None = None,
    created_at: str | None = None,
    path: str | None = None,
) -> str:
    """
    Save one turn into Chroma.

    created_at: ISO 8601 string. Defaults to now (UTC). Tests may pass an
    explicit backdated value to simulate an older turn.

    Returns the episode id.
    """
    collection = get_episodes_collection(path)
    text = format_episode_text(user_message, assistant_message)
    eid = episode_id or str(uuid4())
    stamp = created_at or datetime.now(timezone.utc).isoformat()

    collection.add(
        ids=[eid],
        documents=[text],
        metadatas=[
            {
                "client_id": client_id,
                "session_id": session_id,
                "created_at": stamp,
            }
        ],
    )
    return eid


def search_episodes(
    query: str,
    client_id: str,
    *,
    n_results: int = 3,
    max_distance: float | None = None,
    path: str | None = None,
) -> list[dict]:
    """
    Find past turns for THIS client that are closest in meaning to query.

    max_distance: if set, hits with cosine distance above this are dropped.
    Without it, this always returns up to n_results hits regardless of how
    irrelevant they are — there is no "nothing relevant" case.

    Returns a list of dicts:
      - id
      - text          (actual words stored in Chroma)
      - client_id
      - session_id
      - created_at    (ISO 8601 string; ranking below is by distance only,
                        NOT by this — recency is not considered here)
      - distance      (lower = closer match; cosine distance)
    """
    collection = get_episodes_collection(path)

    # Avoid asking for more neighbors than exist
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count),
        where={"client_id": client_id},
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, eid in enumerate(ids):
        meta = metadatas[i] or {}
        hits.append(
            {
                "id": eid,
                "text": documents[i],
                "client_id": meta.get("client_id"),
                "session_id": meta.get("session_id"),
                "created_at": meta.get("created_at"),
                "distance": distances[i],
            }
        )

    if max_distance is not None:
        hits = [h for h in hits if h["distance"] <= max_distance]

    return hits
