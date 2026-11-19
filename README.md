# LLM AI Agent: twelve production-style LLM systems

This repo builds twelve LLM systems as **one connected codebase**, not twelve
separate demos. Each project reuses the ones before it: the retrieval stack
feeds the context assembler, the agentic RAG and the GraphRAG pipeline; the
MCP server and the durable state machine carry the agent swarm; the eval gate
grades the agents on every pull request.

Every project ships with tests, a reproducible benchmark, and a write-up in
[`docs/projects/`](docs/projects/).

## Roadmap

| # | Project | Phase | Status |
|---|---|---|---|
| 2 | [Hybrid Reranking Retrieval Stack](docs/projects/02-hybrid-retrieval.md) | 1. Foundations | ✅ Built |
| 1 | Context Assembler Engine | 1. Foundations | Next |
| 4 | Self-Correcting Agentic RAG | 2. Reasoning over data | Planned |
| 3 | GraphRAG Knowledge Pipeline | 2. Reasoning over data | Planned |
| 7 | 3-Tier Memory System | 2. Reasoning over data | Planned |
| 6 | Raw MCP Tool Server | 3. Agent infrastructure | Planned |
| 5 | Durable Agent State Machine | 3. Agent infrastructure | Planned |
| 8 | Typed Multi-Agent Swarm | 3. Agent infrastructure | Planned |
| 10 | Inter-Agent Security Proxy | 3. Agent infrastructure | Planned |
| 9 | Trajectory Eval CI Gate | 4. Production | Planned |
| 11 | Semantic Cache + Routing Gateway | 4. Production | Planned |
| 12 | Public Architecture Teardown | 4. Production | Planned |

Projects are numbered by theme and built in the order above, so each one
has its dependencies in place.

## Layout

```
src/
  agentcore/          shared: LLM client interface, tracing
  hybrid_retrieval/   Project 2
benchmarks/           reproducible benchmark scripts; results/ holds JSON output
docs/
  projects/           one write-up per project
  adr/                architecture decision records
tests/
```

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
pytest -q
python benchmarks/bench_retrieval.py      # downloads SciFact (~3 MB) on first run
```

Optional extras: `.[llm]` for the Anthropic SDK, `.[models]` for pretrained
embedders and cross-encoders from Hugging Face.

## Engineering rules for every project

1. **Numbers or it didn't happen.** Each claim in a write-up links to a
   benchmark script anyone can rerun.
2. **Depend on interfaces.** Projects talk to `agentcore.LLMClient`,
   `Embedder` and `Reranker` protocols, so tests run offline and later
   projects (the routing gateway, the cache) can wrap them.
3. **Trace every stage.** Latency is recorded by `agentcore.Tracer` on the
   same code path that serves requests.
