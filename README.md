# PaneLLM

A panel of LLMs that debate, critique, score, and judge. API-first backend that takes a prompt,
sends it to several LLMs, has them critique and score each other across fixed dimensions over one
or more rounds, then has a judge model synthesize a final answer with reasoning, contributions,
disagreements, and a confidence score.

Not "pick the best model" — a structured brainstorm / verify / correct / score / judge pipeline
whose every intermediate step is stored and queryable.

## Status

Built in milestones. **Current: M3 — persistence layer complete.**
M4 (background worker + LLM fan-out) is next.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # fill in LLM keys if you want real providers
```

**Run without Docker (API only, no worker):**
```bash
uvicorn app.main:app --reload
```

**Run full stack (API + Postgres + Redis):**
```bash
docker compose up -d
docker compose exec api alembic upgrade head
```

Endpoints:
- Health: `curl localhost:8000/healthz`
- Readiness (checks DB): `curl localhost:8000/readyz`
- Create debate session:
  ```bash
  curl -X POST localhost:8000/v1/debates \
    -H "x-api-key: dev-local-key" \
    -H "content-type: application/json" \
    -d '{"prompt":"Is a hot dog a sandwich?"}'
  # → {"session_id": "...", "status": "queued"}
  ```
- Read session: `curl localhost:8000/v1/debates/<session_id> -H "x-api-key: dev-local-key"`
- API docs (auto-generated): `http://localhost:8000/docs`

Run tests (no API keys, no Docker needed):
```bash
pytest --cov=app --cov-report=term-missing
# 55 tests, 96% coverage
```

## Architecture

Modular monolith, two process types sharing `app/`:

```
Client → FastAPI (api) → Postgres (sessions)
                       → Redis (job queue)
                             ↓
                        arq worker → LLM providers (OpenAI / Anthropic / Gemini / mock)
                                   → Postgres (answers, critiques, scores, judgments)
```

- **api process** — validates requests, writes sessions to DB, enqueues jobs. Never calls LLMs.
- **worker process** (M4+) — runs debates: parallel LLM fan-out, critique rounds, judge synthesis.
- **Providers** — adapter pattern: all three vendors behind one `LLMProvider` interface.
- **Repositories** — all SQL in one file (`app/storage/repositories.py`). Nothing else touches the DB.

## File structure

```
app/
  main.py                 ← app entry point, wires middleware and routers
  config.py               ← all settings from environment variables
  api/                    ← HTTP endpoints (thin — no business logic)
  core/                   ← @timer, @retry, structured logging
  providers/              ← OpenAI / Anthropic / Gemini / mock adapters + registry
  storage/                ← ORM models, DB session, all SQL
  workers/                ← (M4) arq job that runs the debate engine
  services/               ← (M4+) debate engine, scoring, judge
tests/                    ← 55 tests, SQLite in-memory, no real API calls
alembic/                  ← database migration scripts
.github/workflows/ci.yml  ← lint + types + tests + integration on every push
Dockerfile                ← single image, command chosen at runtime (api or worker)
docker-compose.yml        ← api + postgres + redis
```

## Security assumptions (v1)

- Secrets only via environment (`.env`, never committed); missing required secrets fail fast.
- API-key header gate on all debate endpoints; wrong key → 401.
- User prompt is always a separate `user` message, never concatenated into system instructions.
- Logs carry correlation IDs, never API keys or full prompts at INFO level.
- All DB access through SQLAlchemy ORM — no raw SQL string construction.
