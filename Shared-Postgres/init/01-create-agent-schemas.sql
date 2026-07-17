-- One database (mva_pipeline, created automatically by POSTGRES_DB) shared by all
-- agents, namespaced per agent via Postgres schemas so table names can never
-- collide and no agent's migrations can touch another agent's tables.
CREATE SCHEMA IF NOT EXISTS agent1 AUTHORIZATION postgres;

CREATE USER mva_user WITH PASSWORD 'mva_password';
CREATE SCHEMA IF NOT EXISTS agent2 AUTHORIZATION mva_user;

-- Agent 3 (Analytics Agent) — conversation memory only. Uses the postgres
-- superuser like agent1, not a dedicated role like agent2: this schema is
-- created idempotently by Agent 3's own init_db() at every startup too
-- (this script only runs once, on a fresh volume), so there's no benefit
-- to a separate role at this project's scale.
CREATE SCHEMA IF NOT EXISTS agent3 AUTHORIZATION postgres;
