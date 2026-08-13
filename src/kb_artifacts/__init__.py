"""Public API for inspecting and selecting JSONL evidence collections."""

from .contracts import EvidenceRecord, SelectionDecision, SourceReference
from .corpus import count_corpus, describe_corpus, facet_corpus, sample_corpus
from .inspection import inspect_source
from .profiles import CorpusProfileError, CorpusProfiles, load_corpus_profiles
from .query import QueryExpression, QueryValidationError, evaluate_query, parse_query
from .selection import SelectionRequest, select

__all__ = (
    "CorpusProfileError",
    "CorpusProfiles",
    "EvidenceRecord",
    "QueryExpression",
    "QueryValidationError",
    "SelectionDecision",
    "SelectionRequest",
    "SourceReference",
    "count_corpus",
    "describe_corpus",
    "evaluate_query",
    "facet_corpus",
    "inspect_source",
    "load_corpus_profiles",
    "parse_query",
    "sample_corpus",
    "select",
)
