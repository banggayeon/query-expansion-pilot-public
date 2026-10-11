"""Summarize stored results only. Does not load a model or rerun search."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qepilot.audit import read_jsonl


RUN = ROOT / "runs/cpu_minimal_20261011"


def avg(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def table(headers, rows):
    def fmt(v):
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows])


def main():
    m = json.loads((RUN / "manifest.json").read_text())
    grouped = read_jsonl(RUN / "grouped_metrics.jsonl")
    contrast = read_jsonl(RUN / "contrast.jsonl")
    dirs = read_jsonl(RUN / "dir.jsonl")
    invs = read_jsonl(RUN / "inv.jsonl")
    changes = read_jsonl(RUN / "rank_changes.jsonl")
    summary = {"status": "provisional_AI_not_human", "contrast": [], "dir_ai_qc": [], "rank_changes": [], "inv": []}
    for method in m["methods"]:
        rows = [r for r in contrast if r["method"] == method]
        variants = [r["raw"]["variants"][v] for r in rows for v in ("q0", "qINV", "qDIR")]
        summary["contrast"].append({"method": method, "families": 4, "queries": 12, "positive_negative_pairs": 24,
                                    "pair_accuracy": avg(v["pair_accuracy"] for v in variants),
                                    "top1_positive": avg(v["top1_positive"] for v in variants),
                                    "hard_negative_before_positive": avg(v["hard_negative_before_positive"] for v in variants),
                                    "strict_switch_families": sum(r["raw"]["strict_positive_switch"] == 1 for r in rows),
                                    "sensitivity_excluding_f04_pair_accuracy": avg(v["pair_accuracy"] for r in rows if r["family_id"] != "f04" for v in r["raw"]["variants"].values())})
        dr = [r["raw"] for r in dirs if r["method"] == method and r["label_layer"] == "ai_qc_36"]
        summary["dir_ai_qc"].append({"method": method,
                                    **{name: {"one": sum(r[name] == 1 for r in dr), "zero": sum(r[name] == 0 for r in dr), "unknown": sum(r[name] is None for r in dr)} for name in ("old_only_hit", "new_only_success", "joint_success")}})
        for layer in ("ai_qc_36", "author_draft_180"):
            iv = [r["raw"] for r in invs if r["method"] == method and r["label_layer"] == layer and r["k"] == 10]
            summary["inv"].append({"method": method, "label_layer": layer, "families": len(iv),
                                   "mean_delta_recall_INV_minus_original": avg(v["delta_recall"] for v in iv),
                                   "mean_delta_ndcg_INV_minus_original": avg(v["delta_ndcg"] for v in iv),
                                   "recall_decreased_families": sum(v["delta_recall"] is not None and v["delta_recall"] < 0 for v in iv)})
    for comparison in ("bm25_D-bm25_O", "hybrid_DO-hybrid_OO"):
        for k in (10, 20, 50):
            rows = [r for r in changes if r["comparison"] == comparison and r["k"] == k]
            summary["rank_changes"].append({"comparison": comparison, "k": k, "queries": len(rows),
                                           "order_changed": sum(not r["same_order"] for r in rows),
                                           "membership_changed": sum(bool(r["entered"]) for r in rows),
                                           "mean_overlap_count": avg(r["topk_overlap_count"] for r in rows)})
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    sections = ["# 최소 CPU 실행 수치 부록", "", "모든 수치는 provisional이다. AI QC와 인간 검수는 다르며 human-reviewed qrels는 0개다.", "",
                "## Contrast-only (AI QC 4 family)", "",
                table(["방법", "Pair accuracy", "Top1 positive", "Hard-negative 우선", "Strict switch /4", "f04 제외 pair accuracy"],
                      [[r[k] for k in ("method", "pair_accuracy", "top1_positive", "hard_negative_before_positive", "strict_switch_families", "sensitivity_excluding_f04_pair_accuracy")] for r in summary["contrast"]]),
                "", "Positive-negative 동점은 0.5. query별 두 pair, 총 24 pair. f04 제외는 QC ambiguity에 따른 label 민감도이며 추가 검색이 아니다.", ""]
    for layer, note in (("ai_qc_36", "AI가 재검토한 36 label / 12질의·4 family; 다른 문서는 미판정"),
                        ("author_draft_180", "작성상 draft 180 label / 36질의·12 family; 144 label은 이번 AI 재검토도 하지 않음")):
        sections += ["## Known-qrels: " + layer, "", note + ". 완전 recall/nDCG가 아니다. 인간 검수 coverage는 모두 0%.", "",
                     table(["방법", "K", "Known Recall", "Known nDCG", "Known-label coverage", "평균 미판정/K"],
                           [[r[k] for k in ("method", "k", "recall", "ndcg", "judged_coverage", "unjudged")] for r in grouped if r["label_layer"] == layer and r["group"] == "all"]), ""]
    sections += ["## DIR@10: AI QC subset", "", "각 칸은 1 / 0 / unknown family 수이다. unknown을 0으로 바꾸지 않는다.", "",
                 table(["방법", "OldOnly Hit", "NewOnly Success", "Joint Success"],
                       [[r["method"]] + [" / ".join(str(r[k][v]) for v in ("one", "zero", "unknown")) for k in ("old_only_hit", "new_only_success", "joint_success")] for r in summary["dir_ai_qc"]]), "",
                 "## INV@10", "", table(["방법", "Label 층", "Family N", "ΔRecall", "ΔnDCG", "Recall 하락 family"],
                                              [[r[k] for k in ("method", "label_layer", "families", "mean_delta_recall_INV_minus_original", "mean_delta_ndcg_INV_minus_original", "recall_decreased_families")] for r in summary["inv"]]), "",
                 "## Full-corpus 후보 변화", "", table(["비교", "K", "질의 N", "순서 변경", "구성 변경", "평균 공통 문서"],
                                                                [[r[k] for k in ("comparison", "k", "queries", "order_changed", "membership_changed", "mean_overlap_count")] for r in summary["rank_changes"]]), "",
                 "## 유형별 @10: author draft 진단", "", "세 변형별·DIR 유형별·INV 유형별·도메인별·split별 모두 저장한다. 평가 label의 정확성은 미확정이다.", "",
                 table(["그룹", "방법", "질의 N", "Family N", "Known Recall", "Known nDCG", "Known-label coverage"],
                       [[r[k] for k in ("group", "method", "n_queries", "n_families", "recall", "ndcg", "judged_coverage")] for r in grouped if r["label_layer"] == "author_draft_180" and r["k"] == 10 and r["group"] != "all"]), "",
                 "## CPU 단계별 실측", "", table(["단계", "wall sec", "CPU sec", "종료 RSS MiB", "누적 ru_maxrss MiB"],
                                                       [[name, t["wall_seconds"], t["cpu_seconds"], t["rss_mib_at_end"], t["process_peak_rss_mib_so_far"]] for name, t in m["timings"].items()]), "",
                 "전체 검색 파이프라인 wall %.4fs, CPU %.4fs. 단일 실행이며 download·dependency 설치·평가는 제외. " % (m["search_pipeline_wall_seconds"], m["search_pipeline_cpu_seconds"]),
                 "ru_maxrss는 첫 단계부터 같은 높은 값으로 관측되었다. launcher/초기화 영향이 분리되지 않은 프로세스 high-water 값이므로 개별 모델의 순수 peak라고 해석하지 않는다. RSS도 단계 종료 관측값이며 순간 peak가 아니다.", ""]
    (RUN / "TABLES.md").write_text("\n".join(sections))
    print(json.dumps({"contrast": summary["contrast"], "rank_changes": summary["rank_changes"]}, indent=2))


if __name__ == "__main__":
    main()
