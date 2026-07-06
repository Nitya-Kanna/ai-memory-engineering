# AI Memory Engineering

Financial advisory agent for multiple synthetic bank clients. Built to study how memory works across sessions in AI.

**Stack:** FastAPI · SQLite · Ollama · Chroma (Phase 1+)

## Memory types

| Type | What it stores | Status |
|------|----------------|--------|
| **Working** | What the model sees this turn (profile + current session messages) | Phase 0 |
| **Episodic** | Past conversations and events | Phase 1 |
| **Semantic** | Stable facts about the client | Phase 2 |
| **Procedural** | How to run recurring tasks | Phase 3 |

## Roadmap

- [x] **Phase 0** — Baseline agent, bank client profiles, SQLite sessions, Ollama, memory tests
- [ ] **Phase 1** — Episodic memory (Chroma)
- [ ] **Phase 2** — Semantic memory (structured facts + conflict resolution)
- [ ] **Phase 3** — Procedural memory (workflows)

**Now:** Phase 0. Agent has no cross-session recall by design.

## Setup

```bash
uv sync
cp .env.example .env
ollama pull llama3.2
```

## Run tests

```bash
uv run python -m tests.test_01_show_working_memory
uv run python -m tests.test_02_scenario_0_baseline
```

## Layout

```
bank_client_profiles/   # synthetic client JSON
app/agent/              # prompts + Ollama agent
app/db/                 # SQLite sessions/messages
tests/                  # memory inspection tests
```
