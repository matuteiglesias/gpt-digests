"""Small, dependency-free tag normalization derived from useful legacy behavior."""

from __future__ import annotations

import ast
import unicodedata


def normalize_value(value: object) -> str:
    """Casefold, accent-fold, and normalize whitespace, underscores, and hyphens."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.replace("_", " ").replace("-", " ").split())


def parse_tag_values(value: object) -> list[str]:
    """Accept common list and comma-separated bus tag representations safely."""
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed if str(item).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def canonical_tag(value: object) -> str:
    """Return a deterministic ``namespace:value`` tag, defaulting to ``free``."""
    raw = str(value or "").strip()
    if not raw:
        return "free:unknown"
    namespace, separator, tag_value = raw.partition(":")
    if not separator:
        namespace, tag_value = "free", raw
    normalized_namespace = normalize_value(namespace).replace(" ", "_") or "free"
    normalized_value = normalize_value(tag_value).replace(" ", "_") or "unknown"
    return f"{normalized_namespace}:{normalized_value}"


def tag_lexeme(value: object) -> str:
    """Return a namespace-independent normalized tag value for recipe matching."""
    return canonical_tag(value).split(":", 1)[1].replace("_", " ")


def normalized_tags(value: object) -> tuple[str, ...]:
    """Canonicalize and deterministically deduplicate parsed tag values."""
    return tuple(sorted(set(canonical_tag(item) for item in parse_tag_values(value))))
