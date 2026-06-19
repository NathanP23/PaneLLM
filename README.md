# PaneLLM

A panel of LLMs that debate, critique, score, and judge. API-first backend that takes a prompt, sends it to several LLMs, has them critique and score
each other across fixed dimensions over one or more rounds, then has a judge model synthesize a
final answer with reasoning, contributions, disagreements, and a confidence score.

Not "pick the best model" — a structured brainstorm / verify / correct / score / judge pipeline
whose every intermediate step is stored and queryable.

## Status

Built in milestones. See [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) for the changelog and what
currently works. **Current: M1 — runnable skeleton.**

## Quick start (M1)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

Then:
- Health: `curl localhost:8000/healthz`
- Readiness: `curl localhost:8000/readyz`
- Create a (stub) debate: `curl -X POST localhost:8000/v1/debates -H "x-api-key: dev-local-key" -H "content-type: application/json" -d '{"prompt":"Is a hot dog a sandwich?"}'`
- API docs: open `http://localhost:8000/docs`

Run tests (no API keys needed): `pytest`

## Architecture

Modular monolith, two process types (FastAPI `api` + `arq` worker) sharing `app/`, backed by
Postgres (durable record) and Redis (queue + cache + rate-limit + SSE pub/sub). Provider quirks
live behind a single adapter interface. Full design and rationale are in the approved plan;
day-to-day conventions are in [CLAUDE.md](CLAUDE.md).

## Security assumptions (v1)

- Secrets only via environment (`.env`, never committed); missing required secrets fail fast.
- API-key header gate + per-key rate limiting (real auth comes later).
- User prompt is always a separate message, never concatenated into system instructions.
- Logs carry correlation IDs, never API keys or full prompts at INFO.
