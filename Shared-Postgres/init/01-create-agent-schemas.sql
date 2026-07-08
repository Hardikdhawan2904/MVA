-- One database (mva_pipeline, created automatically by POSTGRES_DB) shared by all
-- agents, namespaced per agent via Postgres schemas so table names can never
-- collide and no agent's migrations can touch another agent's tables.
CREATE SCHEMA IF NOT EXISTS agent1 AUTHORIZATION postgres;

CREATE USER mva_user WITH PASSWORD 'mva_password';
CREATE SCHEMA IF NOT EXISTS agent2 AUTHORIZATION mva_user;
