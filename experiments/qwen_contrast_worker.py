"""Phase 2: official Qwen yes/no scoring, exactly 36 synthetic pairs, no labels."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

MODEL='Qwen/Qwen3-Reranker-4B'
REVISION='22e683669bc0f0bd69640a1354a6d0aebcfeede5'
INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
CAP=512

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def probability(gap):
    if not math.isfinite(gap):raise ValueError('nonfinite score')
    return 1/(1+math.exp(-gap)) if gap>=0 else math.exp(gap)/(1+math.exp(gap))
def body(q,d):return f'<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}'
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    a=argparse.ArgumentParser();a.add_argument('--pairs',required=True);a.add_argument('--model-dir',required=True);a.add_argument('--out',required=True);a.add_argument('--preflight-only',action='store_true');args=a.parse_args()
    from transformers import AutoTokenizer
    pairs=json.loads(Path(args.pairs).read_text())['pairs']
    assert len(pairs)==36 and len({(r['query_id'],r['doc_id']) for r in pairs})==36
    assert all(set(r)=={'query_id','doc_id','query','document','query_hash','document_hash'} for r in pairs)
    assert all(digest(r['query'])==r['query_hash'] and digest(r['document'])==r['document_hash'] for r in pairs)
    tok=AutoTokenizer.from_pretrained(args.model_dir,padding_side='left',local_files_only=True,trust_remote_code=False)
    pre=tok.encode(PREFIX,add_special_tokens=False);suf=tok.encode(SUFFIX,add_special_tokens=False)
    false_id=tok.convert_tokens_to_ids('no');true_id=tok.convert_tokens_to_ids('yes')
    assert tok.encode('yes',add_special_tokens=False)==[true_id] and tok.encode('no',add_special_tokens=False)==[false_id]
    enc=[pre+tok(body(r['query'],r['document']),add_special_tokens=False,truncation=False)['input_ids']+suf for r in pairs]
    assert max(map(len,enc))<=CAP, 'Input exceeds cap; refuse truncation'
    manifest={'model':MODEL,'revision':REVISION,'dtype':'bfloat16','batch_size':1,'max_tokens':CAP,'token_lengths':list(map(len,enc)),'truncated_pairs':0,'padding_side':'left','instruction':INSTRUCTION,'prefix':PREFIX,'suffix':SUFFIX,'prompt_hash':digest(PREFIX+INSTRUCTION+SUFFIX),'yes_id':true_id,'no_id':false_id,'attention':'sdpa','ranking_score':'float32(logit_yes)-float32(logit_no)','probability':'two-class softmax yes','generation':False,'pair_file_hash':hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest()}
    out=Path(args.out);out.mkdir(exist_ok=True)
    write(out/'preflight.json',manifest)
    print(json.dumps({'stage':'token_preflight','pairs':36,'max_tokens_observed':max(map(len,enc))}),flush=True)
    if args.preflight_only:return
    if (out/'scores.jsonl').exists():raise RuntimeError('refuse duplicate run')
    import torch
    from transformers import AutoModelForCausalLM
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    torch.manual_seed(0);torch.set_num_threads(4)
    t=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).to('cuda').eval()
    torch.cuda.synchronize();manifest['load_seconds']=time.perf_counter()-t
    manifest['gpu_name']=torch.cuda.get_device_name(0);manifest['gpu_total_bytes']=torch.cuda.get_device_properties(0).total_memory
    manifest['versions']={k:importlib.metadata.version(k) for k in ['torch','transformers','tokenizers','huggingface-hub','safetensors']}
    def score(ids):
        feed=tok.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda')
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():logits=model(**feed,use_cache=False).logits[:,-1,:].float();selected=logits[:,[false_id,true_id]];probs=selected.softmax(dim=1)[:,1];gap=selected[:,1]-selected[:,0]
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        result=[{'logit_no':x[0],'logit_yes':x[1],'score':g,'p_yes':p} for x,g,p in zip(selected.cpu().tolist(),gap.cpu().tolist(),probs.cpu().tolist())]
        assert all(math.isfinite(v) for row in result for v in row.values())
        return result,elapsed
    _,warm=score([enc[0]]);manifest['warmup_seconds']=warm
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    records=[]
    for pair,ids in zip(pairs,enc):
        result,elapsed=score([ids]);r={**{k:pair[k] for k in ['query_id','doc_id','query_hash','document_hash']},**result[0],'tokens':len(ids),'truncated':False,'forward_seconds':elapsed};records.append(r)
        with (out/'scores.jsonl').open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
        print(json.dumps({'stage':'scored','count':len(records),'query_id':pair['query_id'],'doc_id':pair['doc_id']}),flush=True)
    manifest['scoring_wall_seconds']=time.perf_counter()-start
    manifest['forward_seconds_sum']=sum(r['forward_seconds'] for r in records)
    manifest['peak_allocated_bytes']=torch.cuda.max_memory_allocated();manifest['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    # Validation repeats only: first two approved pairs, order reversed, batch 1 and 2.
    repeated,_=score([enc[0]]);batched,_=score([enc[1],enc[0]])
    write(out/'numerical_checks.json',{'repeat_single':repeated,'reversed_batch2':batched,'original_first_two':records[:2],'extra_forward_pairs':3,'warmup_pairs':1})
    manifest['post_validation_peak_allocated_bytes']=torch.cuda.max_memory_allocated();manifest['post_validation_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    write(out/'manifest.json',manifest)
    del model;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    write(out/'cleanup.json',{'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True)})
    print('PHASE2_COMPLETE',flush=True)

if __name__=='__main__':main()
