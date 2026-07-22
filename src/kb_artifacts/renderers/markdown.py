"""Plain Markdown renderer for selected evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from kb_artifacts.contracts import ArtifactRecipe, EvidenceRecord, SelectionDecision


def render(recipe: ArtifactRecipe, selected: Iterable[tuple[EvidenceRecord, SelectionDecision]]) -> str:
    groups: dict[str, list[tuple[EvidenceRecord, SelectionDecision]]] = defaultdict(list)
    for record, decision in selected:
        groups[recipe.group(record)].append((record, decision))
    lines = [f"# {recipe.title}", "", "Generated from read-only governed bus records.", ""]
    for group in sorted(groups, key=str.casefold):
        lines.extend((f"## {group}", ""))
        for record, decision in groups[group]:
            title = record.title or record.summary or (record.text or "Untitled evidence").splitlines()[0][:120]
            lines.extend((f"### {title}", ""))
            if record.summary:
                lines.extend((record.summary, ""))
            elif record.text:
                lines.extend((record.text, ""))
            lines.append(f"- Score: {decision.score:g}")
            lines.append(f"- Reasons: {', '.join(decision.reasons)}")
            lines.append(f"- Source: `{record.provenance.source_ref or record.record_id}`")
            lines.append(f"- Partition: `{record.provenance.partition}:{record.provenance.line_number}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
