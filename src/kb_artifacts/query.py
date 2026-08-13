"""Typed, deterministic predicates over normalized evidence records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from typing import Mapping, Sequence

from kb_artifacts.contracts import EvidenceRecord
from kb_artifacts.normalization import normalize_value, tag_lexeme


class QueryValidationError(ValueError):
    """Raised when a query expression is not structurally valid."""


_OPERATORS = frozenset({"eq", "in", "contains", "exists", "gte", "lte", "regex", "all", "any", "not"})
_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_COMMON_FIELDS = {
    "record_id", "source_kind", "text", "summary", "title", "timestamp",
    "conversation_id", "message_id", "tags",
}
_MISSING = object()


@dataclass(frozen=True)
class QueryExpression:
    """A validated JSON-compatible query expression."""

    expression: Mapping[str, object]

    def __post_init__(self) -> None:
        canonical = _parse_node(self.expression, path="query")
        object.__setattr__(self, "expression", canonical)

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible representation."""
        return _copy_node(self.expression)


def parse_query(value: QueryExpression | Mapping[str, object]) -> QueryExpression:
    """Validate a mapping, or return an already validated expression."""
    return value if isinstance(value, QueryExpression) else QueryExpression(value)


def _copy_node(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _copy_node(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_node(item) for item in value]
    return value


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise QueryValidationError(f"{path} must be an object")
    return value


def _field(value: object, path: str) -> str:
    if not isinstance(value, str) or not _FIELD.fullmatch(value):
        raise QueryValidationError(f"{path} must be a non-empty field name")
    return value


def _scalar(value: object, path: str) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise QueryValidationError(f"{path} must be a JSON scalar")


def _exact_keys(payload: Mapping[str, object], required: set[str], path: str) -> None:
    if set(payload) != required:
        raise QueryValidationError(f"{path} requires exactly: {', '.join(sorted(required))}")


def _parse_node(value: object, path: str) -> dict[str, object]:
    node = _mapping(value, path)
    if len(node) != 1:
        raise QueryValidationError(f"{path} must contain exactly one operator")
    operator, raw = next(iter(node.items()))
    if operator not in _OPERATORS:
        raise QueryValidationError(f"{path} has unknown operator {operator!r}")
    if operator in {"all", "any"}:
        if not isinstance(raw, (list, tuple)) or not raw:
            raise QueryValidationError(f"{path}.{operator} must be a non-empty array")
        return {operator: tuple(_parse_node(item, f"{path}.{operator}[{index}]") for index, item in enumerate(raw))}
    if operator == "not":
        return {operator: _parse_node(raw, f"{path}.not")}
    payload = _mapping(raw, f"{path}.{operator}")
    required = {"field"} if operator == "exists" else {"target", "pattern"} if operator == "regex" else {"field", "values"} if operator == "in" else {"field", "value"}
    _exact_keys(payload, required, f"{path}.{operator}")
    if operator == "regex":
        target = _field(payload["target"], f"{path}.regex.target")
        pattern = payload["pattern"]
        if not isinstance(pattern, str):
            raise QueryValidationError(f"{path}.regex.pattern must be a string")
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as error:
            raise QueryValidationError(f"{path}.regex.pattern is invalid: {error}") from error
        return {operator: {"target": target, "pattern": pattern}}
    field = _field(payload["field"], f"{path}.{operator}.field")
    if operator == "exists":
        return {operator: {"field": field}}
    if operator == "in":
        values = payload["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise QueryValidationError(f"{path}.in.values must be a non-empty array")
        return {operator: {"field": field, "values": tuple(_scalar(item, f"{path}.in.values[{index}]") for index, item in enumerate(values))}}
    operand = _scalar(payload["value"], f"{path}.{operator}.value")
    if operator in {"gte", "lte"} and not _number(operand):
        raise QueryValidationError(f"{path}.{operator}.value must be numeric")
    return {operator: {"field": field, "value": operand}}


def _value(record: EvidenceRecord, field: str) -> object:
    if field in _COMMON_FIELDS:
        value = getattr(record, field)
        return value.isoformat() if isinstance(value, datetime) else value
    return record.annotations.get(field, _MISSING)


def _number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _equal(actual: object, wanted: object, *, tags: bool = False) -> bool:
    if tags and isinstance(actual, str) and isinstance(wanted, str):
        return tag_lexeme(actual) == normalize_value(wanted)
    if isinstance(actual, str) and isinstance(wanted, str):
        return normalize_value(actual) == normalize_value(wanted)
    if type(actual) is not type(wanted):
        return False
    return actual == wanted


def _evaluate(record: EvidenceRecord, node: Mapping[str, object]) -> bool:
    operator, raw = next(iter(node.items()))
    if operator == "all":
        return all(_evaluate(record, item) for item in raw)  # type: ignore[union-attr]
    if operator == "any":
        return any(_evaluate(record, item) for item in raw)  # type: ignore[union-attr]
    if operator == "not":
        return not _evaluate(record, raw)  # type: ignore[arg-type]
    payload = raw  # validated internal mapping
    if operator == "regex":
        actual = _value(record, payload["target"])  # type: ignore[index]
        return isinstance(actual, str) and re.search(payload["pattern"], actual, re.IGNORECASE) is not None  # type: ignore[index]
    field = payload["field"]  # type: ignore[index]
    actual = _value(record, field)
    if operator == "exists":
        return actual is not _MISSING and actual is not None
    if actual is _MISSING:
        return False
    if operator in {"gte", "lte"}:
        wanted = payload["value"]  # type: ignore[index]
        return _number(actual) and (actual >= wanted if operator == "gte" else actual <= wanted)  # type: ignore[operator]
    actual_values: Sequence[object] = actual if isinstance(actual, (list, tuple, set)) else (actual,)
    if operator == "eq":
        return any(_equal(item, payload["value"], tags=field == "tags") for item in actual_values)  # type: ignore[index]
    if operator == "in":
        return any(_equal(item, wanted, tags=field == "tags") for item in actual_values for wanted in payload["values"])  # type: ignore[index]
    wanted = payload["value"]  # type: ignore[index]
    if isinstance(actual, str) and isinstance(wanted, str):
        return normalize_value(wanted) in normalize_value(actual)
    return any(_equal(item, wanted, tags=field == "tags") for item in actual_values)


def evaluate_query(record: EvidenceRecord, query: QueryExpression | Mapping[str, object]) -> bool:
    """Evaluate one expression without ranking, scoring, or external state."""
    return _evaluate(record, parse_query(query).expression)
