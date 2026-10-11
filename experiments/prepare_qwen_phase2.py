"""Build only label-free input pairs from the approved Phase 1 snapshot."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
plan=json.loads((ROOT/'reports/reranker_phase1_candidates.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in plan['source_hashes'].items())
docs={d['doc_id']:d for d in rows(ROOT/'data/corpus.jsonl')}
qs={q['query_id']:q for q in rows(ROOT/'data/queries.jsonl')}
pairs=[]
for s in plan['snapshots']:
    if s['origin']!='contrast_diagnostic':continue
    for d in s['ordered_doc_ids']:
        query=qs[s['query_id']]['text'];document=docs[d]['title']+' '+docs[d]['text']
        pairs.append({'query_id':s['query_id'],'doc_id':d,'query':query,'document':document,'query_hash':sha(query),'document_hash':sha(document)})
assert len(pairs)==36
out=ROOT/'runs/qwen_contrast_phase2_v1';out.mkdir(exist_ok=False)
(out/'pairs.json').write_text(json.dumps({'pairs':pairs},ensure_ascii=False,indent=2)+'\n')
(out/'source_hashes.json').write_text(json.dumps(plan['source_hashes'],indent=2)+'\n')
print(json.dumps({'pairs':len(pairs),'run_id':out.name,'no_model_inference':True}))
