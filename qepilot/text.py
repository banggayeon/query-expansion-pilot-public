"""Frozen English tokenizer and lexical diagnostics; no semantic predictions."""
import re

TOKENIZER_VERSION = "english-preserve-numeric-v1"
TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*|<=|>=|!=|[<>=/%]", re.I)


def tokens(text):
    return tuple(m.group().lower() for m in TOKEN.finditer(text))


def overlap(a, b):
    left, right = set(tokens(a)), set(tokens(b))
    a, b = a.lower(), b.lower()
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = nxt
    return {"token_jaccard": len(left & right) / len(left | right) if left | right else 1.0,
            "character_similarity": 1 - row[-1] / max(len(a), len(b)) if a or b else 1.0,
            "tokenizer_version": TOKENIZER_VERSION}
