# Shared Postgres

The single PostgreSQL server used by every agent in the pipeline. One database (`mva_pipeline`), one container — each agent gets its own schema inside it, so table names can never collide and no agent's migrations can touch another agent's tables.

```
mva_pipeline (database)
├── agent1  (Schema Intelligence Layer's tables, owned by postgres)
├── agent2  (MVA Data Profiling Engine's tables, owned by mva_user)
└── agent3  (Analytics Agent's conversation memory, owned by postgres)
```

This intentionally replaced an earlier setup where each agent ran its own separate Postgres server — consolidated into one server, then further consolidated from one-database-per-agent into one shared database with per-agent schemas.

## Usage

```bash
docker compose up -d
```

Starts a `postgres:16-alpine` container on `localhost:5433` (not the default `5432`, to avoid clashing with any local Postgres install you might already have) with a persistent named volume. On first startup, `init/01-create-agent-schemas.sql` runs automatically and creates all three schemas and the `mva_user` role.

That init script only runs once, on a *fresh* volume — it won't re-run against an already-initialized instance. Agent 3's own `init_db()` (`Analytics-Agent/app/services/database.py`) creates its schema/table idempotently at every service startup instead, which is why it doesn't need a dedicated role the way Agent 2 does: `CREATE SCHEMA IF NOT EXISTS` needs no elevated one-time setup, unlike `CREATE USER`.

Every other repo in this pipeline (`Schema-Intelligence-Layer`, `MVA-use-case-latest-one`, `Analytics-Agent`) points at this server — none of them define their own Postgres container. Their own `docker-compose.yml` files are comment-only pointers back to this one.

## Connection details

| | |
|---|---|
| Host | `localhost` |
| Port | `5433` |
| Database | `mva_pipeline` |
| Agent 1 role | `postgres` / `postgres` — schema `agent1` |
| Agent 2 role | `mva_user` / `mva_password` — schema `agent2` |
| Agent 3 role | `postgres` / `postgres` — schema `agent3` |

These are local development defaults, not production credentials — this container isn't exposed beyond `localhost`.

## Stopping / resetting

```bash
docker compose down          # stop, keep data
docker compose down -v       # stop and wipe the volume (full reset)
```
