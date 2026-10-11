"""Read-only scoring inputs; evaluate saved contrast output, no inference."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qepilot.contracts import CandidateSnapshot,validate_same_candidates
from qepilot.evaluation import Qrels,contrast_diagnostic,evaluate
RUN=ROOT/'runs/qwen_contrast_phase2_v1'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def relation(a,b):return 'above' if a>b else 'below' if a<b else 'tie'
def transition(a,b):
    if a==b:return 'unchanged_'+a
    if b=='tie':return 'to_tie'
    if a=='tie':return 'from_tie_'+b
    return 'correction' if b=='above' else 'regression'

def main():
    hashes=json.loads((RUN/'source_hashes.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    input_pairs=json.loads((RUN/'pairs.json').read_text())['pairs']
    rr=rows(RUN/'gpu/scores.jsonl');expected={(x['query_id'],x['doc_id']):x for x in input_pairs}
    assert len(rr)==len(expected)==36 and {(x['query_id'],x['doc_id']) for x in rr}==set(expected)
    for r in rr:
        src=expected[r['query_id'],r['doc_id']]
        assert r['query_hash']==src['query_hash'] and r['document_hash']==src['document_hash']
        assert not r['truncated'] and r['tokens']<=512 and all(math.isfinite(r[k]) for k in ['score','logit_no','logit_yes','p_yes'])
        assert math.isclose(r['score'],r['logit_yes']-r['logit_no'],abs_tol=1e-6)
    plans=json.loads((ROOT/'reports/reranker_phase1_candidates.json').read_text())
    qs={};scores={};rankings={}
    qc=rows(ROOT/'qc/cpu_minimal_20261011/relevance.ai.jsonl');corpus=frozenset(x['doc_id'] for x in rows(ROOT/'data/corpus.jsonl'))
    base={(r['query_id'],r['method']):r['ranked_docs'] for r in rows(ROOT/'runs/cpu_minimal_20261011/rankings.jsonl')}
    for s in plans['snapshots']:
        if s['origin']!='contrast_diagnostic':continue
        q=s['query_id'];scores[q]={r['doc_id']:r['score'] for r in rr if r['query_id']==q}
        ids=sorted(scores[q],key=lambda d:(-scores[q][d],d));rankings[q]=ids
        snap=CandidateSnapshot(s['candidate_set_id'],q,tuple(s['ordered_doc_ids']),s['set_hash'],s['order_hash'],s['origin'],s['budget'],hashes['data/corpus.jsonl'],'AI_QC_36_draft')
        validate_same_candidates(snap,ids)
        qs[q]=Qrels({r['doc_id']:r['qc_grade'] for r in qc if r['query_id']==q},corpus,'draft','AI_QC_36_draft')
        assert evaluate(ids,qs[q],3,allow_draft=True)['recall']==evaluate(list(s['ordered_doc_ids']),qs[q],3,allow_draft=True)['recall']
    diagnostics=[];transitions=[];details=[];invs=[]
    methods=['bm25_O','bm25_D','dense_O','hybrid_OO','hybrid_DO']
    for c in rows(ROOT/'data/contrast_sets.jsonl'):
        for variant,q in c['query_ids'].items():
            pos=next(d for d,g in qs[q].grades.items() if g==2);neg=[d for d,g in qs[q].grades.items() if g==0]
            for method in methods+['qwen']:
                sc=scores[q] if method=='qwen' else {r['doc_id']:r['score'] for r in base[q,method]}
                diag=contrast_diagnostic(sc,qs[q],c['roles'].values(),c['roles']['hard_negative'],allow_draft=True)
                diagnostics.append({'query_id':q,'family':c['family_id'],'method':method,**diag})
            for d in rankings[q]:
                raw=next(r for r in rr if (r['query_id'],r['doc_id'])==(q,d))
                details.append({**raw,'grade':qs[q].grades[d],'contrast_rank':rankings[q].index(d)+1,'role':next(k for k,v in c['roles'].items() if v==d),'baseline':{m:next(r for r in base[q,m] if r['doc_id']==d) for m in methods}})
            for method in methods:
                sc={r['doc_id']:r['score'] for r in base[q,method]}
                for d in neg:
                    a=relation(sc[pos],sc[d]);b=relation(scores[q][pos],scores[q][d])
                    transitions.append({'query_id':q,'family':c['family_id'],'baseline':method,'positive':pos,'negative':d,'hard_negative':d==c['roles']['hard_negative'],'before':a,'after':b,'transition':transition(a,b),'reranker_margin':scores[q][pos]-scores[q][d]})
        a,b=c['query_ids']['q0'],c['query_ids']['qINV'];ids=rankings[a]
        concord=sum((rankings[a].index(x)<rankings[a].index(y))==(rankings[b].index(x)<rankings[b].index(y)) for i,x in enumerate(ids) for y in ids[i+1:])
        invs.append({'family':c['family_id'],'q0_order':rankings[a],'qINV_order':rankings[b],'rank_order_concordance':concord/3,'positive_top1_both':rankings[a][0]==rankings[b][0]==c['roles']['original_relevant']})
    summary={m:{k:statistics.mean(r[k] for r in diagnostics if r['method']==m) for k in ['pair_accuracy','top1_positive','hard_negative_before_positive']} for m in methods+['qwen']}
    counts={m:dict(Counter(r['transition'] for r in transitions if r['baseline']==m)) for m in methods}
    report={'status':'provisional_AI_QC_not_human','scope':'contrast_only_3_documents','summary':summary,'transitions':counts,'family_qwen':{f:statistics.mean(r['pair_accuracy'] for r in diagnostics if r['method']=='qwen' and r['family']==f) for f in ['f01','f02','f03','f04']},'inv':invs,'unchanged_source_hashes':True,'candidate_preservation':True,'recall_at3_invariant':True,'top20_or_at10_claims':'not evaluated: only three contrast documents scored'}
    write(RUN/'evaluation.json',report);write(RUN/'pair_transitions.json',transitions);write(RUN/'document_scores.json',details)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
