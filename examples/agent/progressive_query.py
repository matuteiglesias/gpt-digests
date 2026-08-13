"""Offline deterministic agent-like corpus exploration and selection example."""

from pathlib import Path
import tempfile

from kb_artifacts import (
    SelectionRequest,
    count_corpus,
    describe_corpus,
    facet_corpus,
    load_corpus_profiles,
    sample_corpus,
    select,
)

profiles = load_corpus_profiles("examples/agent/corpus.toml")
source = {"corpus": "sanitized-playbooks", "profiles": profiles}
print("available:", [item["id"] for item in profiles.list()["corpora"]])
print("records:", describe_corpus(**source)["counts"]["records_observed"])
print("domains:", facet_corpus(field="domain", **source)["values"])

candidate = {"gte": {"field": "reusability_score", "value": 4}}
print("candidate count:", count_corpus(query=candidate, **source)["counts"]["records_matching"])
print("sample refs:", [item["record_id"] for item in sample_corpus(query=candidate, limit=5, **source)["samples"]])

refined = {
    "all": [
        candidate,
        {"contains": {"field": "tags", "value": "automation"}},
        {"not": {"eq": {"field": "stage", "value": "reflection"}}},
    ]
}
with tempfile.TemporaryDirectory(prefix="kb-artifacts-agent-example-") as temporary:
    output = Path(temporary) / "selected"
    manifest = select(
        SelectionRequest(corpus="sanitized-playbooks", query=refined),
        profiles=profiles,
        output=output,
    )
    print("selected:", manifest["counts"]["selected"])
    print("source ids:", [item["source_id"] for item in manifest["matched_partitions"]])
