

# Pipelines

```mermaid
flowchart TD
  A[Raw Logs JSONL] --> B[Ingest → Events]
  A2[Raw Sessions JSONL] --> C[Ingest → Sessions]
  B --> D[Unitize → Cohorts (day or session grouping)]
  C --> D
  D --> E[Add Bags → pairbag / tagbag]
  E --> F[L2 Build → MDX digests]
  F --> G[Index L2 (windows)]
  G --> H[Publish]
```

**Bags**:

* **pairbag**: high-signal co-tag pairs (e.g., `free:python + stage:execute`) over a long window
* **tagbag**: single tags with enough support (e.g., `free:automation`)

These bags create coherent strands across time, perfect for recurring digests like “This week in Python”.

