"""Read saved ablation results only. No scoring or inference."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports'
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def mean(x):return statistics.mean(x)
def accuracy(ps):return mean({'above':1,'tie':.5,'below':0}[p['relation']] for p in ps)
def table(head,rows):return ['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows],'']
data={}
for folder in sorted((ROOT/'runs').glob('bm25_ablation_v1_*')):
    cfg=json.loads((folder/'manifest.json').read_text())['config'];key=cfg['condition']
    data[key]={name:read(folder/(name+'.jsonl')) for name in ['pairs','rankings','metrics','dir','inv','topk_changes','f04_group_scores']}
    data[key]['manifest']=json.loads((folder/'manifest.json').read_text())

out=['# BM25 Ablation: 전체 수치 부록','','모든 label은 provisional AI QC / draft이며 인간 검수는 0건이다. P=token_sum_replacement. F는 P와 무관하게 보호되지 않은 독립 unigram 기능어 그룹만 제거한다. 모든 score는 실험용 scoring 값이며 ES 또는 표준 BM25 성능이 아니다.','']
summary=[];family=[];trans=[];inv=[];tops=[];dirt=[];inter=[];changes=[]
for key,d in data.items():
    for mode in ['O','D']:
        ps=[p for p in d['pairs'] if p['mode']==mode];hs=[p for p in ps if p['hard_negative']]
        mm=[m['raw'] for m in d['metrics'] if m['mode']==mode and m['raw']['k']==10]
        summary.append([key,mode,f'{accuracy(ps):.6f}',f'{accuracy([p for p in ps if p["family"]!="f04"]):.6f}',f'{mean(m["recall"] for m in mm):.6f}',f'{mean(m["ndcg"] for m in mm):.6f}',f'{mean(m["judged_coverage"] for m in mm):.6f}'])
        family.append([key,mode,*[f'{accuracy([p for p in ps if p["family"]==f]):.6f}' for f in ['f01','f02','f03','f04']]])
        for scope,rs in [('P-H',hs),('all24',ps)]:
            c=Counter(p['transition'] for p in rs)
            trans.append([key,mode,scope,*[c[k] for k in ['correction','regression','to_tie','from_tie_above','from_tie_below','unchanged_above','unchanged_below','unchanged_tie']]])
        iv=[x for x in d['inv'] if x['mode']==mode];den=sum(x['q0_hit'] for x in iv);num=sum(x['q0_hit'] and x['qINV_hit'] for x in iv)
        pr=[p for x in iv for p in x['pair_relations']];pd=sum(p['q0']=='above' for p in pr);pn=sum(p['q0']==p['qINV']=='above' for p in pr)
        inv.append([key,mode,f'{num}/{den}',f'{sum(x["q0_hit"] for x in iv)}/4',f'{sum(x["qINV_hit"] for x in iv)}/4',f'{pn}/{pd}' if pd else 'N/A',','.join(x['family'] for x in iv if x['q0_hit'] and not x['qINV_hit']) or '없음'])
        for k in [10,20,50]:
            ts=[x for x in d['topk_changes'] if x['mode']==mode and x['k']==k]
            tops.append([key,mode,k,sum(x['order_changed'] for x in ts),sum(bool(x['entered']) for x in ts),sum(len(x['entered']) for x in ts),f'{mean(x["overlap"] for x in ts):.6f}'])
        for x in [x for x in d['dir'] if x['mode']==mode]:
            old=x['documents']['original_relevant'];new=x['documents']['changed_relevant']
            dirt.append([key,mode,x['family'],f'{old["q0_rank"]}→{old["qDIR_rank"]}',f'{old["baseline_qDIR_rank"]}→{old["qDIR_rank"]}',f'{new["baseline_qDIR_rank"]}→{new["qDIR_rank"]}',x['raw']['old_only_hit'],x['raw']['new_only_success']])
        base=data['F0N0P0']['rankings'];baseidx={(x['query_id'],x['mode']):{r['doc_id']:r for r in x['ranked_docs']} for x in base}
        delta=[p['positive_result']['rank']-baseidx[p['query_id'],mode][p['positive']]['rank'] for p in hs]
        changes.append([key,mode,sum(v<0 for v in delta),sum(v>0 for v in delta),sum(v==0 for v in delta)])

out+=['## 전체 contrast 및 known-qrels @10','']+table(['조건','mode','Pair accuracy','f04 제외','Known Recall','Known nDCG','QC-label coverage'],summary)
out+=['## Family별 pair accuracy','']+table(['조건','mode','f01','f02','f03','f04'],family)
out+=['## Baseline 대비 관계 전이','', 'P-H는 12 pair, all24는 P-X까지 포함한 24 pair. score 동점은 doc_id 순서와 분리한다.','']+table(['조건','mode','범위','correction','regression','to tie','tie→above','tie→below','동일 above','동일 below','동일 tie'],trans)
out+=['## Known contrast positive 순위 변화','']+table(['조건','mode','개선','악화','불변'],changes)
out+=['## INV 유지','', '조건부 회수 유지= q0 Top10 성공 중 qINV Top10 성공. pair 유지=q0 strict win 중 qINV strict win; 동점은 strict win이 아니다.','']+table(['조건','mode','회수 유지','q0 hit','qINV hit','pair 유지','회수 실패 family'],inv)
out+=['## Top-K 구성·순서 변화 대 baseline','', '모든 질의 포함, 각 행 분모 12. 진입합계는 query-document 조합 수. K100은 전 조건 N/A. 질의별 진입·이탈·공통 문서 rank delta는 각 run의 topk_changes.jsonl에 모두 저장했다.','']+table(['조건','mode','K','순서 변경 질의','구성 변경 질의','진입합계','평균 교집합/K'],tops)
out+=['## DIR OldOnly/NewOnly 순위','', 'None은 partial qrels의 unknown이며 0이 아니다. AI contrast에는 Shared가 없다.','']+table(['조건','mode','family','Old q0→qDIR','Old qDIR baseline→조건','New qDIR baseline→조건','Old Hit10','New Success10'],dirt)
for mode in ['O','D']:
    val={tuple(int(key[i]) for i in [1,3,5]):accuracy([p for p in d['pairs'] if p['mode']==mode]) for key,d in data.items()}
    b=val[0,0,0]
    effects={'F':val[1,0,0]-b,'N':val[0,1,0]-b,'P':val[0,0,1]-b,
      'FN':val[1,1,0]-val[1,0,0]-val[0,1,0]+b,
      'FP':val[1,0,1]-val[1,0,0]-val[0,0,1]+b,
      'NP':val[0,1,1]-val[0,1,0]-val[0,0,1]+b,
      'FNP':val[1,1,1]-val[1,1,0]-val[1,0,1]-val[0,1,1]+val[1,0,0]+val[0,1,0]+val[0,0,1]-b}
    for name,e in effects.items():inter.append([mode,name,f'{e:+.6f}'])
    for axis,name in enumerate('FNP'):
        ds=[]
        for bits in itertools.product((0,1),repeat=3):
            if bits[axis]:continue
            changed=list(bits);changed[axis]=1;ds.append(val[tuple(changed)]-val[bits])
        inter.append([mode,'평균 marginal '+name,f'{mean(ds):+.6f}'])
out+=['## 단독 효과·상호작용 (pair accuracy 차이)','', 'F/N/P는 다른 요인 0에서의 효과, FN/FP/NP는 세 번째 요인 0에서의 차이의 차이, FNP는 3차 차분이다. 평균 marginal은 다른 요인 4조합의 평균 효과이며 통계적 유의성을 뜻하지 않는다.','']+table(['mode','효과','차이'],inter)

detail=['# Contrast 전체 pair: rank·score·margin·baseline 관계 비교','','총 384 pair. H=hard negative, X=다른 의미 문서. label은 provisional이며 각 rank는 전체 60문서 기준이다.','']
for key,d in data.items():
    detail+=['## '+key,'']
    rr=[]
    for p in d['pairs']:
        a,b=p['positive_result'],p['negative_result']
        rr.append([p['mode'],p['query_id'],p['positive'],p['negative']+(' H' if p['hard_negative'] else ' X'),f'{a["rank"]}/{b["rank"]}',f'{a["score"]:.9f}/{b["score"]:.9f}',f'{p["margin"]:+.9f}',p['baseline_relation']+'→'+p['relation'],p['transition'],p['rank_order_reversed']])
    detail+=table(['mode','query','P','negative','rank P/N','score P/N','margin','관계','전이','ID순위역전'],rr)
for name,body in [('BM25_ABLATION_TABLES.md',out),('BM25_ABLATION_PAIRS.md',detail)]:
    path=R/name
    if path.exists():raise RuntimeError('refuse overwrite '+str(path))
    path.write_text('\n'.join(body)+'\n')
print(json.dumps({'summary':summary,'interactions':inter,'rank_changes':changes,'all_input_hashes_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in data['F0N0P0']['manifest']['source_hashes'].items())},ensure_ascii=False))
