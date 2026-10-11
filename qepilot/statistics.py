"""Family-paired descriptive summaries. No significance claims."""
import random
import statistics
from collections import defaultdict


def paired_summary(rows, metric="recall", repetitions=10000, seed=20261010):
    """rows: family_id, variant, before, after, snapshot_id, comparison.

    Exactly q0/qINV/qDIR required per family. Unknown excludes whole family
    rather than silently changing its internal denominator.
    """
    rows = list(rows)
    if repetitions <= 0:
        raise ValueError("positive repetitions required")
    if len({(r["snapshot_id"], r["comparison"]) for r in rows}) > 1:
        raise ValueError("mixed snapshots or comparisons")
    if len({(r["before"].get("k"), r["before"].get("qrels_status"), r["before"].get("interpretation")) for r in rows}) > 1:
        raise ValueError("mixed K, qrels status or interpretation")
    grouped = defaultdict(dict)
    for r in rows:
        for key in ("snapshot_id", "k", "qrels_status", "interpretation"):
            if key not in r["before"] or key not in r["after"] or r["before"][key] != r["after"][key]:
                raise ValueError("paired metrics require matching metadata")
        if r["before"]["snapshot_id"] != r["snapshot_id"]:
            raise ValueError("row/metric snapshot mismatch")
        if r["variant"] in grouped[r["family_id"]]:
            raise ValueError("duplicate family variant")
        grouped[r["family_id"]][r["variant"]] = r
    differences, excluded = {}, []
    variant_deltas = defaultdict(list)
    for family, group in sorted(grouped.items()):
        if set(group) != {"q0", "qINV", "qDIR"}:
            raise ValueError("unpaired family")
        if any(r["before"].get(metric) is None or r["after"].get(metric) is None for r in group.values()):
            excluded.append(family)
            continue
        vals = []
        for variant, r in group.items():
            delta = r["after"][metric] - r["before"][metric]
            vals.append(delta)
            variant_deltas[variant].append(delta)
        differences[family] = statistics.mean(vals)
    values = list(differences.values())
    out = {"n_families": len(values), "excluded_unknown": excluded, "paired_deltas": differences,
           "variant_mean_deltas": {k: statistics.mean(v) for k, v in variant_deltas.items()},
           "ci_interpretation": "descriptive_only; excludes model, generation and qrels uncertainty"}
    if not values:
        return {**out, "mean_delta": None, "ci95": None}
    mean = statistics.mean(values)
    ci, loo = None, None
    if len(values) >= 5:
        rng = random.Random(seed)
        boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
        ci = [boot[int((repetitions - 1) * .025)], boot[int((repetitions - 1) * .975)]]
    if len(values) > 1:
        deleted = [statistics.mean(values[:i] + values[i+1:]) for i in range(len(values))]
        loo = [min(deleted), max(deleted)]
    return {**out, "mean_delta": mean, "median_delta": statistics.median(values), "ci95": ci,
            "leave_one_family_out": loo, "improved": sum(v > 0 for v in values),
            "tied": sum(v == 0 for v in values), "worsened": sum(v < 0 for v in values)}


def by_type(rows, key, **kwargs):
    rows = list(rows)
    groups = defaultdict(list)
    family_tags = {}
    for row in rows:
        tag = row[key]
        if row["family_id"] in family_tags and family_tags[row["family_id"]] != tag:
            raise ValueError("type changes within family")
        family_tags[row["family_id"]] = tag
        groups[tag].append(row)
    return {"overall": paired_summary(rows, **kwargs),
            "by_type": {tag: paired_summary(group, **kwargs) for tag, group in sorted(groups.items())}}


def drift_summary(records):
    """drift must be independently reviewed True/False or None, not loss-derived."""
    reviewed = [r for r in records if r["drift"] is not None]
    if any(type(r["drift"]) is not bool for r in reviewed):
        raise ValueError("invalid drift label")
    expanded = [r for r in reviewed if r["expanded"]]
    table = {f"drift_{d}_loss_{l}": 0 for d in (False, True) for l in (False, True)}
    for r in reviewed:
        if r["delta"] is not None:
            table[f"drift_{r['drift']}_loss_{r['delta'] < 0}"] += 1
    return {"reviewed_n": len(reviewed), "unknown_n": len(records) - len(reviewed),
            "drift_rate": sum(r["drift"] for r in reviewed) / len(reviewed) if reviewed else None,
            "expanded_reviewed_n": len(expanded),
            "expanded_drift_rate": sum(r["drift"] for r in expanded) / len(expanded) if expanded else None,
            "drift_loss_table": table}
