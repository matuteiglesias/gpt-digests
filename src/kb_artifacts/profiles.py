"""Local corpus profile configuration and deterministic source resolution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import tomllib


class CorpusProfileError(ValueError):
    """Raised when corpus profile configuration is missing or malformed."""


@dataclass(frozen=True)
class CorpusProfile:
    """Approved local source roles associated with a stable corpus name."""

    corpus_id: str
    chunk_globs: tuple[str, ...] = ()
    summary_globs: tuple[str, ...] = ()
    description: str | None = None
    annotations: Mapping[str, str] | None = None
    excerpts_permitted_by_default: bool = False


@dataclass(frozen=True)
class CorpusProfiles:
    """Validated collection of named local corpus profiles."""

    profiles: Mapping[str, CorpusProfile]

    def get(self, corpus_id: str) -> CorpusProfile:
        try:
            return self.profiles[corpus_id]
        except KeyError as error:
            raise CorpusProfileError(f"Unknown corpus profile: {corpus_id}") from error

    def list(self) -> dict[str, object]:
        """Return discovery metadata without exposing configured filesystem paths."""
        return {
            "corpora": [
                {
                    "id": profile.corpus_id,
                    "description": profile.description,
                    "source_roles": {
                        "chunk": bool(profile.chunk_globs),
                        "summary": bool(profile.summary_globs),
                    },
                    "annotations": dict(sorted((profile.annotations or {}).items())),
                    "excerpts_permitted_by_default": profile.excerpts_permitted_by_default,
                }
                for profile in sorted(self.profiles.values(), key=lambda item: item.corpus_id)
            ]
        }


def _strings(value: object, location: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise CorpusProfileError(f"{location} must be an array of non-empty strings")
    return tuple(value)


def load_corpus_profiles(path: str | Path) -> CorpusProfiles:
    """Load profiles from a local TOML file."""
    source = Path(path)
    try:
        with source.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CorpusProfileError(f"Could not load corpus profiles: {error}") from error
    corpora = document.get("corpora")
    if not isinstance(corpora, dict):
        raise CorpusProfileError("Corpus profile file requires a [corpora] table")
    profiles: dict[str, CorpusProfile] = {}
    for corpus_id, raw in corpora.items():
        if not isinstance(corpus_id, str) or not corpus_id or not corpus_id.replace("-", "a").replace("_", "a").isalnum():
            raise CorpusProfileError(f"Invalid corpus profile id: {corpus_id!r}")
        if not isinstance(raw, dict):
            raise CorpusProfileError(f"corpora.{corpus_id} must be a table")
        allowed = {"description", "chunk_globs", "summary_globs", "annotations", "excerpts_permitted_by_default"}
        unknown = set(raw) - allowed
        if unknown:
            raise CorpusProfileError(f"corpora.{corpus_id} has unknown keys: {', '.join(sorted(unknown))}")
        description = raw.get("description")
        annotations = raw.get("annotations", {})
        excerpts = raw.get("excerpts_permitted_by_default", False)
        if description is not None and not isinstance(description, str):
            raise CorpusProfileError(f"corpora.{corpus_id}.description must be a string")
        if not isinstance(annotations, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in annotations.items()):
            raise CorpusProfileError(f"corpora.{corpus_id}.annotations must contain string values")
        if not isinstance(excerpts, bool):
            raise CorpusProfileError(f"corpora.{corpus_id}.excerpts_permitted_by_default must be boolean")
        chunk_globs = _strings(raw.get("chunk_globs"), f"corpora.{corpus_id}.chunk_globs")
        summary_globs = _strings(raw.get("summary_globs"), f"corpora.{corpus_id}.summary_globs")
        if not chunk_globs and not summary_globs:
            raise CorpusProfileError(f"corpora.{corpus_id} must declare at least one source glob")
        profiles[corpus_id] = CorpusProfile(corpus_id, chunk_globs, summary_globs, description, annotations, excerpts)
    return CorpusProfiles(profiles)


def resolve_corpus_sources(
    *,
    chunk_globs: Iterable[str],
    summary_globs: Iterable[str],
    corpus: str | None = None,
    profiles: CorpusProfiles | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], CorpusProfile | None]:
    """Resolve exactly one source mode; profiles and explicit globs never merge."""
    chunks, summaries = tuple(chunk_globs), tuple(summary_globs)
    if corpus is None:
        return chunks, summaries, None
    if chunks or summaries:
        raise CorpusProfileError("Corpus profile cannot be combined with explicit source globs")
    if profiles is None:
        raise CorpusProfileError("Corpus profile configuration is required when --corpus is used")
    profile = profiles.get(corpus)
    return profile.chunk_globs, profile.summary_globs, profile
