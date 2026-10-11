"""Experimental scoring ONLY. Not equivalent to Elasticsearch synonym/dis_max.

Pure functions accept term statistics. No corpus loading, indexing, or search.
Future ES adapter must compare analyzer/positions, phrase tf/df, norms, rewrite,
boost and tie handling against fixtures rather than assuming equivalence.
"""
from dataclasses import dataclass, asdict
import math
from .text import tokens


@dataclass(frozen=True)
class ScoringConfig:
    implementation: str = "experimental_phrase_max_v1"
    k1: float = 1.2
    b: float = 0.75
    original_boost: float = 2.0
    alias_boost: float = 1.0
    phrase_slop: int = 0
    tie_contribution: float = 0.0

    def manifest(self):
        return asdict(self)


def phrase_tf(document_tokens, phrase):
    phrase = tokens(phrase)
    if not phrase:
        raise ValueError("empty phrase")
    n = len(phrase)
    return sum(tuple(document_tokens[i:i+n]) == phrase for i in range(len(document_tokens) - n + 1))


def bm25(tf, df, n_docs, doc_length, avg_length, config=ScoringConfig()):
    if not (n_docs > 0 and 0 <= df <= n_docs and tf >= 0 and doc_length >= 0
            and avg_length > 0 and config.k1 > 0 and 0 <= config.b <= 1):
        raise ValueError("invalid BM25 statistics")
    if tf > 0 and df == 0:
        raise ValueError("positive tf requires positive df")
    if tf == 0:
        return 0.0
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return idf * tf * (config.k1 + 1) / (tf + config.k1 * (1 - config.b + config.b * doc_length / avg_length))


def synonym_or(group_scores, config=ScoringConfig()):
    """group_scores: iterable[(original B score, iterable[alias B scores])]."""
    if config.phrase_slop != 0 or config.tie_contribution != 0:
        raise ValueError("only slop=0 and max/tie=0 implemented")
    total = 0.0
    for original, aliases in group_scores:
        scores = [config.original_boost * original, *(config.alias_boost * x for x in aliases)]
        if any(not math.isfinite(x) or x < 0 for x in scores):
            raise ValueError("invalid group score")
        total += max(scores)
    return total
