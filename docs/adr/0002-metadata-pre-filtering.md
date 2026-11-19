# ADR 0002: Metadata filters are applied before top-k

**Status:** Accepted, 2026-11-19

## Context
Filtering after retrieval (take top 100, then drop non-matching) is simple
but can return too few results, or none, when a filter is selective.

## Decision
Compile each filter to a boolean mask over the corpus and apply it inside
each retriever before top-k selection.

## Consequences
- A query always returns up to k matching documents when they exist.
- Building the mask is O(corpus) per query in Python. At 5k documents this
  is negligible; past roughly 1M documents it should move to per-field
  column indexes or the vector database's native filtering.
