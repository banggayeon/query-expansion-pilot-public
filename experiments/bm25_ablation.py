"""Small CPU-only factorial ablation. Never imports a model or edits baseline."""
import hashlib
import itertools
import json
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qepilot.text import tokens
from qepilot.scoring import bm25
from qepilot.evaluation import Qrels, evaluate, evaluate_dir, contrast_diagnostic

STOPS = frozenset('a an the how what when where who whom which why do does did is are was were be been being am have has had i me my we us our you your he him his she her it its they them their at by for from in into of on to with as and or'.split())
ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
NUMBERS = {v: i for i, v in enumerate(ONES)}
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
NUMBERS.update(TENS)
for word, value in TENS.items():
    for i in range(1, 10):
        NUMBERS[word + '-' + ONES[i]] = value + i
UNITS = {u + suffix: u for u in ('second', 'minute', 'hour') for suffix in ('', 's')}


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def lines(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def normalize(seq):
    """Integers 0..99, optional hyphenated time unit. Never split decimal versions."""
    expanded = []
    for t in seq:
        prefix, sep, last = t.rpartition('-')
        if sep and last in UNITS and (prefix in NUMBERS or prefix.isdigit() and int(prefix) <= 99):
            expanded.extend((prefix, UNITS[last]))
        else:
            expanded.append(t)
    out, i = [], 0
    while i < len(expanded):
        t = expanded[i]
        if t in TENS and i + 1 < len(expanded) and expanded[i + 1] in ONES[1:10]:
            out.append(str(TENS[t] + NUMBERS[expanded[i + 1]])); i += 2
            continue
        out.append(str(NUMBERS[t]) if t in NUMBERS else UNITS.get(t, t))
        i += 1
    return tuple(out)


def tf(doc, term):
    return sum(doc[i:i+len(term)] == term for i in range(len(doc)-len(term)+1)) if term else 0


def units(term, n, p):
    seq = normalize(tokens(term)) if n else tokens(term)
    # Repeated tokens count repeatedly within and across groups: no implicit dedup.
    return [(t,) for t in seq] if p else [seq]


def removable(original, protected):
    key = tokens(original)
    return len(key) == 1 and key[0] in STOPS and key not in protected


def relation(a, b):
    return 'above' if a > b else 'below' if a < b else 'tie'


def transition(before, after):
    if before == after: return 'unchanged_' + before
    if after == 'tie': return 'to_tie'
    if before == 'tie': return 'from_tie_' + after
    return 'correction' if after == 'above' else 'regression'


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('data', 'qepilot', 'configs', 'qc/cpu_minimal_20261011', 'runs/cpu_minimal_20261011')
            for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}


def plan():
    folder = ROOT / 'configs/bm25_ablation_v1'
    folder.mkdir(exist_ok=False)
    for f, n, p in itertools.product((0, 1), repeat=3):
        key = f'F{f}N{n}P{p}'
        save(folder / (key + '.json'), {
            'run_id': 'bm25_ablation_v1_' + key, 'condition': key, 'F': f, 'N': n, 'P': p,
            'P_name': 'token_sum_replacement', 'query_modes': ['O', 'D'],
            'families': ['f01', 'f02', 'f03', 'f04'], 'k': [10, 20, 50, 100],
            'F_policy': 'Only unprotected independent original unigram groups; retain all phrase-internal tokens for P=0 AND P=1',
            'stopwords': sorted(STOPS), 'protected_exclusions': ['all annotated spans', 'negation', 'only', 'numbers', 'versions', 'units'],
            'normalization': 'Integer words 0..99 incl hyphens; number-time-unit hyphens to tokens; second/minute/hour singular; no conversion; decimals unchanged',
            'P_accumulation': 'sum BM25 for every token occurrence in each expression; duplicates retained within/across groups; max original*2 vs alias*1 per group',
            'P_limit': 'phrase IDF/TF and score accumulation change together, NOT a single adjacency intervention or standard BM25/ES equivalence',
            'order': ['N representation normalization', 'P expression scoring', 'F independent-group zeroing'],
            'bm25': {'k1': 1.2, 'b': 0.75, 'original_boost': 2, 'alias_boost': 1},
            'labels': 'provisional AI QC; draft status retained', 'tie_policy': 'exact score equality; doc_id sorting only for deterministic ranks',
            'baseline': 'reuse frozen rankings; arithmetic parity check only', 'tuning': False})


def run():
    before = hashes()
    docs = read(ROOT / 'data/corpus.jsonl')
    queries = {q['query_id']: q for q in read(ROOT / 'data/queries.jsonl') if q['family_id'] in ('f01','f02','f03','f04')}
    exps = {e['query_id']: e for e in read(ROOT / 'runs/cpu_minimal_20261011/expansions.jsonl') if e['query_id'] in queries}
    baseline = {(x['query_id'], x['method'][-1]): x['ranked_docs'] for x in read(ROOT / 'runs/cpu_minimal_20261011/rankings.jsonl') if x['query_id'] in queries and x['method'] in ('bm25_O','bm25_D')}
    contrasts = read(ROOT / 'data/contrast_sets.jsonl')
    qc = read(ROOT / 'qc/cpu_minimal_20261011/relevance.ai.jsonl')
    corpus = frozenset(d['doc_id'] for d in docs)
    qrels = {q: Qrels({r['doc_id']: r['qc_grade'] for r in qc if r['query_id']==q}, corpus, 'draft', 'ai_qc_36_fixed') for q in queries}
    all_results = []
    for cfgfile in sorted((ROOT / 'configs/bm25_ablation_v1').glob('*.json')):
        cfg = json.loads(cfgfile.read_text()); f,n,p = (cfg[x] for x in 'FNP')
        out = ROOT / 'runs' / cfg['run_id']; out.mkdir(exist_ok=False)
        started = time.perf_counter(); cpu = time.process_time()
        dt = {d['doc_id']: (normalize(tokens(d['title']+' '+d['text'])) if n else tokens(d['title']+' '+d['text'])) for d in docs}
        avg = statistics.mean(map(len, dt.values())); cache = {}
        def base(docid, term):
            if term not in cache:
                fs = {d: tf(t, term) for d,t in dt.items()}
                cache[term] = fs, sum(v>0 for v in fs.values())
            fs,df = cache[term]
            return bm25(fs[docid], df, len(dt), len(dt[docid]), avg)
        rankings=[]; traces=[]; pairs=[]; metrics=[]; topchanges=[]; dirs=[]; inv=[]; maps={}
        for qid,q in queries.items():
            protected = {tokens(q['text'][s['start']:s['end']]) for s in q['spans']}
            for mode in ('O','D'):
                scores = {}; groups=exps[qid]['groups_'+mode]
                for d in dt:
                    total=0
                    for gi,(original,aliases) in enumerate(groups):
                        removed=bool(f and removable(original,protected))
                        terms=[]; values=[]
                        for term in [original,*aliases]:
                            us=units(term,n,p); vals=[base(d,u) for u in us]; values.append(sum(vals))
                            terms.append({'expression':term,'terms':[{'tokens':u,'tf':cache[u][0][d],'df':cache[u][1],'B':v} for u,v in zip(us,vals)]})
                        contribution=0 if removed else max([2*values[0],*values[1:]])
                        total+=contribution
                        if q['family_id']=='f04' and d in ('d016','d033','d050','d007'):
                            traces.append({'query_id':qid,'mode':mode,'doc_id':d,'group':gi,'removed_F':removed,'expressions':terms,'contribution':contribution,'doc_length':len(dt[d]),'avg_length':avg})
                    scores[d]=total
                if not (f or n or p):
                    for old in baseline[qid,mode]:
                        assert abs(scores[old['doc_id']]-old['score'])<1e-12
                    ranked=baseline[qid,mode]
                else:
                    ranked=[{'doc_id':d,'rank':i+1,'score':scores[d]} for i,d in enumerate(sorted(scores,key=lambda d:(-scores[d],d)))]
                maps[qid,mode]={r['doc_id']:r for r in ranked}
                rankings.append({'query_id':qid,'mode':mode,'ranked_docs':ranked})
                original={r['doc_id']:r for r in baseline[qid,mode]}
                for k in cfg['k']:
                    ids=[r['doc_id'] for r in ranked]
                    metrics.append({'query_id':qid,'mode':mode,'raw':evaluate(ids,qrels[qid],k,allow_draft=True)})
                    if k>len(dt):
                        topchanges.append({'query_id':qid,'mode':mode,'k':k,'status':'N/A'});continue
                    a=[r['doc_id'] for r in baseline[qid,mode][:k]];b=ids[:k];common=set(a)&set(b)
                    topchanges.append({'query_id':qid,'mode':mode,'k':k,'entered':sorted(set(b)-set(a)),'exited':sorted(set(a)-set(b)), 'overlap':len(common)/k,'order_changed':a!=b, 'common_rank_changes':{d:ranked[ids.index(d)]['rank']-original[d]['rank'] for d in sorted(common) if maps[qid,mode][d]['rank']!=original[d]['rank']}})
        for c in contrasts:
            for variant,qid in c['query_ids'].items():
                for mode in ('O','D'):
                    m=maps[qid,mode];old={r['doc_id']:r for r in baseline[qid,mode]}
                    pos=[d for d,g in qrels[qid].grades.items() if g==2];neg=[d for d,g in qrels[qid].grades.items() if g==0]
                    assert len(pos)==1 and len(neg)==2
                    for d in neg:
                        good=pos[0];a=relation(old[good]['score'],old[d]['score']);b=relation(m[good]['score'],m[d]['score'])
                        pairs.append({'family':c['family_id'],'query_id':qid,'mode':mode,'positive':good,'negative':d,'hard_negative':d==c['roles']['hard_negative'], 'baseline_relation':a,'relation':b,'transition':transition(a,b),'margin':m[good]['score']-m[d]['score'],'baseline_margin':old[good]['score']-old[d]['score'], 'positive_result':m[good],'negative_result':m[d],'rank_order_reversed':(old[good]['rank']<old[d]['rank'])!=(m[good]['rank']<m[d]['rank'])})
                    diagnostic=contrast_diagnostic({d:v['score'] for d,v in m.items()},qrels[qid],c['roles'].values(),c['roles']['hard_negative'],allow_draft=True)
                    assert diagnostic['status']=='ok'
            for mode in ('O','D'):
                q0,qi,qd=(c['query_ids'][v] for v in ('q0','qINV','qDIR'))
                rankings_ids=lambda q: [d for d in sorted(maps[q,mode], key=lambda d:maps[q,mode][d]['rank'])]
                dr=evaluate_dir(rankings_ids(q0),rankings_ids(qd),qrels[q0],qrels[qd],allow_draft=True)
                dirs.append({'family':c['family_id'],'mode':mode,'raw':dr,'documents':{role:{'doc_id':d,'q0_rank':maps[q0,mode][d]['rank'],'qDIR_rank':maps[qd,mode][d]['rank'],'baseline_q0_rank':next(x['rank'] for x in baseline[q0,mode] if x['doc_id']==d),'baseline_qDIR_rank':next(x['rank'] for x in baseline[qd,mode] if x['doc_id']==d)} for role,d in c['roles'].items()}})
                good=c['roles']['original_relevant']; a=maps[q0,mode][good]['rank']<=10;b=maps[qi,mode][good]['rank']<=10
                outcomes=[]
                for d in (c['roles']['changed_relevant'],c['roles']['hard_negative']):
                    outcomes.append({'negative':d,'q0':relation(maps[q0,mode][good]['score'],maps[q0,mode][d]['score']),'qINV':relation(maps[qi,mode][good]['score'],maps[qi,mode][d]['score'])})
                inv.append({'family':c['family_id'],'mode':mode,'q0_hit':a,'qINV_hit':b,'conditional_retained':b if a else None,'pair_relations':outcomes})
        elapsed=time.perf_counter()-started
        for name,records in [('rankings',rankings),('pairs',pairs),('f04_group_scores',traces),('metrics',metrics),('topk_changes',topchanges),('dir',dirs),('inv',inv)]:lines(out/(name+'.jsonl'),records)
        lines(out/'normalization.jsonl',[{'doc_id':d['doc_id'],'before':tokens(d['title']+' '+d['text']),'after':dt[d['doc_id']]} for d in docs])
        manifest={'config':cfg,'source_hashes':before,'config_sha256':hashlib.sha256(cfgfile.read_bytes()).hexdigest(),'wall_seconds_including_evaluation':elapsed,'cpu_seconds_including_evaluation':time.process_time()-cpu,'ru_maxrss_MiB_cumulative_process':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'baseline_reused':not(f or n or p),'human_reviewed':False,'model_inference':False,'rankings':len(rankings)}
        save(out/'manifest.json',manifest)
        all_results.append({'condition':cfg['condition'],'wall':elapsed})
    assert hashes()==before, 'frozen inputs changed'
    print(json.dumps({'status':'passed','conditions':all_results,'unchanged_input_files':len(before),'rankings':192,'new_rankings':168,'baseline_reused':24}))


if __name__=='__main__':
    if sys.argv[1:] == ['--plan']: plan()
    elif sys.argv[1:] == ['--run']: run()
    else: raise SystemExit('Use --plan before --run')
