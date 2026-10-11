"""Explicit span policy. Verifications are attestations, never inferred from labels."""
from dataclasses import dataclass
from .text import tokens


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    sense: str


@dataclass(frozen=True)
class Alias:
    start: int
    end: int
    text: str
    sense: str
    verified: bool = False
    source: str = ""
    verifier: str = ""


def validate_spans(query, spans):
    end = 0
    for s in sorted(spans, key=lambda s: s.start):
        if (s.kind not in {"immutable_constraint", "alias_expandable_semantic"}
                or not s.sense or not 0 <= s.start < s.end <= len(query) or s.start < end):
            raise ValueError("invalid or overlapping protected spans")
        if (s.start and query[s.start - 1].isalnum()) or (s.end < len(query) and query[s.end].isalnum()):
            raise ValueError("span cuts a word")
        if not tokens(query[s.start:s.end]):
            raise ValueError("empty span")
        end = s.end


def accept_aliases(query, spans, proposals):
    """Do not edit original text. Return approved proposals and rejection reasons."""
    validate_spans(query, spans)
    by_range = {(s.start, s.end): s for s in spans}
    accepted, rejected, seen = [], [], set()
    words = 0
    for p in proposals:
        s = by_range.get((p.start, p.end))
        key = (p.start, p.end, tokens(p.text))
        reason = None
        if s is None:
            reason = "source_span_missing"
        elif s.kind == "immutable_constraint":
            reason = "immutable_constraint"
        elif p.sense != s.sense:
            reason = "sense_changed"
        elif not p.verified or not p.source or not p.verifier:
            reason = "unverified"
        elif not tokens(p.text):
            reason = "empty_alias"
        elif key in seen or tokens(p.text) == tokens(query[p.start:p.end]):
            reason = "duplicate"
        elif len(accepted) >= 3 or words + len(p.text.split()) > 16:
            reason = "budget_exceeded"
        if reason:
            rejected.append({"proposal": p, "reason": reason})
        else:
            accepted.append(p)
            seen.add(key)
            words += len(p.text.split())
    return accepted, rejected


def query_groups(query, spans, proposals=()):
    """O/D/L share original groups. Unannotated text is immutable to expansion."""
    accepted, rejected = accept_aliases(query, spans, proposals)
    groups, cursor = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        groups.extend((t, ()) for t in tokens(query[cursor:s.start]))
        alternatives = tuple(p.text for p in accepted if (p.start, p.end) == (s.start, s.end))
        groups.append((query[s.start:s.end], alternatives))
        cursor = s.end
    groups.extend((t, ()) for t in tokens(query[cursor:]))
    # Repeated original groups contribute once, with stable order.
    merged = {}
    for original, aliases in groups:
        key = tokens(original)
        if key not in merged:
            merged[key] = [original, []]
        merged[key][1].extend(a for a in aliases if a not in merged[key][1])
    return [(o, tuple(a)) for o, a in merged.values()], rejected
