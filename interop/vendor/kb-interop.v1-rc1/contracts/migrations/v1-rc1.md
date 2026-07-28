# KB interoperability v1 RC1 migration guidance

This release introduces the first machine-readable knowledge interoperability surface. Historical artifacts remain valid under their declared historical schemas and must not be rewritten solely to satisfy v1.

Producers adopting Profile 1 add a `kb.module` descriptor, publish versioned knowledge schemas, emit one `kb.knowledge_artifact_manifest` per promoted product, preserve producer-assigned identifiers exactly, and expose a deterministic offline validation command.

Optional fields and namespaced extensions are compatible when they do not change identity or interpretation. Required additions, removals, renames, semantic changes, checksum changes, identity/normalization changes, enum removals, cardinality changes, and document-meaning changes require a new schema version.

Canonical/legacy aliases must declare precedence. Conflicting canonical and legacy values fail validation. Consumers should support the current and immediately previous final versions unless they explicitly declare a longer window.

Run records, run-bundle manifests, lifecycle statuses, operational errors, environment evidence, and promotion mechanics remain producer-owned and are not migrated by this release.
