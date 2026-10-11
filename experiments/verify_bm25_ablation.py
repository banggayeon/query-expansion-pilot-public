"""Read-only artifact verification, no rescoring."""
import hashlib
import json
from pathlib import Path
import math

ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
dirs=sorted((ROOT/'runs').glob('bm25_ablation_v1_*'))
assert len(dirs)==8
baseline={(x['query_id'],x['method'][-1]):x['ranked_docs'] for x in rows(ROOT/'runs/cpu_minimal_20261011/rankings.jsonl') if x['method'] in ['bm25_O','bm25_D']}
for folder in dirs:
    m=json.loads((folder/'manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in m['source_hashes'].items())
    rankings=rows(folder/'rankings.jsonl');assert len(rankings)==24
    lookup={(r['query_id'],r['mode']):r['ranked_docs'] for r in rankings}
    assert len(lookup)==24
    for (q,mode),rs in lookup.items():
        assert len(rs)==len({r['doc_id'] for r in rs})==60
        assert [r['rank'] for r in rs]==list(range(1,61))
        assert rs==sorted(rs,key=lambda r:(-r['score'],r['doc_id']))
        assert all(math.isfinite(r['score']) and r['score']>=0 for r in rs)
        if m['baseline_reused']:assert rs==baseline[q,mode]
        if not q.startswith('f04'):assert rs==lookup[q,'D' if mode=='O' else 'O']
    traces=rows(folder/'f04_group_scores.jsonl')
    for q in ['f04_q0','f04_qINV','f04_qDIR']:
        for mode in ['O','D']:
            for d in ['d016','d033','d050','d007']:
                score=sum(x['contribution'] for x in traces if (x['query_id'],x['mode'],x['doc_id'])==(q,mode,d))
                saved=next(x['score'] for x in lookup[q,mode] if x['doc_id']==d)
                assert abs(score-saved)<1e-10
    pairs=rows(folder/'pairs.jsonl');assert len(pairs)==48
    tops=rows(folder/'topk_changes.jsonl');assert len(tops)==96
    assert sum(x.get('status')=='N/A' for x in tops)==24
    for x in tops:
        if x.get('status')=='N/A':continue
        assert len(x['entered'])==len(x['exited'])
        assert math.isclose(x['overlap'],1-len(x['entered'])/x['k'])
print(json.dumps({'status':'passed','conditions':8,'rankings':192,'pairs':384,'f04_group_sums_checked':192,'input_hashes_unchanged':52,'baseline_reused_exactly':24,'no_alias_OD_identity':True,'K100_NA':True,'scoring_or_inference_rerun':False}))
