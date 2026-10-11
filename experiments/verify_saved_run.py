"""Check saved artifacts and source hashes without model loading or searching."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qepilot.audit import read_jsonl


def main():
    import numpy as np
    run = ROOT / "runs/cpu_minimal_20261011"
    manifest = json.loads((run / "manifest.json").read_text())
    for section in ("data_and_original_config_sha256", "frozen_code_sha256"):
        for filename, expected in manifest[section].items():
            assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    rows = read_jsonl(run / "rankings.jsonl")
    rank = {(r["method"], r["query_id"]): [d["doc_id"] for d in r["ranked_docs"]] for r in rows}
    assert len(rank) == len(rows) == 180
    for r in rows:
        assert len(set(rank[r["method"], r["query_id"]])) == 60
        assert [d["rank"] for d in r["ranked_docs"]] == list(range(1, 61))
        assert r["ranked_docs"] == sorted(r["ranked_docs"], key=lambda d: (-d["score"], d["doc_id"]))
    top = read_jsonl(run / "topk.jsonl")
    assert len(top) == 720
    for r in top:
        if r["k"] == 100:
            assert r["status"] == "N/A" and r["doc_ids"] is None
        else:
            assert len(r["doc_ids"]) == r["k"] and r["doc_ids"] == rank[r["method"], r["query_id"]][:r["k"]]
    vectors = {}
    for name, n in (("document", 60), ("query", 36)):
        v = np.load(run / (name + "_embeddings.npy"), allow_pickle=False)
        assert v.shape == (n, 384) and v.dtype == np.float32 and np.isfinite(v).all()
        error = float(np.max(np.abs(np.linalg.norm(v, axis=1) - 1)))
        assert error < 1e-5
        vectors[name] = {"shape": list(v.shape), "dtype": str(v.dtype), "max_unit_norm_error": error}
    print(json.dumps({"status": "passed", "checks": ["frozen_data_and_code_hashes", "180_full_rankings", "720_topk_records", "K100_NA", "embedding_shape_dtype_norm"],
                      "inference_rerun": False, "vectors": vectors}, indent=2))


if __name__ == "__main__":
    main()
