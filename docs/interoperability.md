# Interoperability boundary

KB Artifacts owns the identity of its canonical selected-evidence product. The
canonical payload is the exact finalized bytes of `selected.jsonl`. CSV and
Markdown are compatibility views, and the existing `manifest.json` is a
producer-local operational manifest rather than a shared artifact manifest.

## Selected-evidence artifact identity

Algorithm identifier:

```text
selected-evidence-sha256.v1
```

For exact finalized `selected.jsonl` bytes:

```text
payload_sha256 = lowercase hexadecimal SHA-256(payload bytes)
artifact_id = "selected-evidence.sha256." + payload_sha256
```

The ID is content-addressed and does not depend on a timestamp, output directory,
absolute path, CSV rendering, Markdown rendering, or the producer-local
`manifest.json`. A one-byte change to `selected.jsonl` changes the ID. Consumers
must treat the published ID as an opaque, case-preserving reference; they do not
need to recompute it.

This algorithm does not change evidence-record identity. Record IDs retain their
existing precedence and semantics.

## Contract dependency gate

The producer will publish `interop/module.v1.json` and
`selected-evidence.artifact-manifest.json` only after the exact approved
`kb_interop_release.v1-rc1` bundle is locally available and its release manifest,
source commit, schema identifiers, file hashes, stable-reference vectors, and
offline validation expectations have been verified.

KB Artifacts does not infer schemas from documentation and does not import
runtime implementation from a sibling checkout. Producer-owned schemas may be
declared through a future shared verification protocol, but no such declaration
is claimed until an approved release explicitly supports it.

When an immutable bundle is supplied locally, verify its identity, complete file
inventory, hashes, schema identities, migration reference, test-vector reference,
and offline/deterministic declaration with:

```bash
make contract-release-verify \
  CONTRACT_RELEASE_MANIFEST=/path/to/release.json \
  CONTRACT_RELEASE_ROOT=/path/to/release-root
```

The verifier is offline and does not fetch or import code from KB Contracts. A
release manifest without the files whose hashes it declares is intentionally
insufficient for contract binding.
