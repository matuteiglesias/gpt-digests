"""Select the runbook record from the bundled basic example."""

from kb_artifacts import SelectionRequest, select


request = SelectionRequest(
    chunk_globs=("examples/basic/evidence.jsonl",),
    tags=("runbook",),
)

result = select(request, output="selected-python")
print(f"Selected {result['counts']['selected']} record into selected-python/")
