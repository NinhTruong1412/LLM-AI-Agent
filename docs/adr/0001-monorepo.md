# ADR 0001: One monorepo with shared interfaces

**Status:** Accepted, 2026-11-19

## Context
The twelve projects depend on each other: agentic RAG needs retrieval, the
swarm needs the MCP server and the state machine, the eval gate grades the
agents. Separate repos would duplicate glue code and hide those links.

## Decision
One Python package per project under `src/`, plus `agentcore` for shared
pieces (LLM client protocol, tracing). Projects depend on protocols
(`LLMClient`, `Embedder`, `Reranker`), never on a concrete SDK.

## Consequences
- Tests run offline with fakes; CI needs no API keys.
- Later projects (routing gateway, semantic cache) can wrap any client.
- One CI pipeline and one dependency set; heavy model dependencies stay in
  optional extras.
