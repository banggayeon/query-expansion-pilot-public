# Query Expansion Pilot 종합 실험 보고서

작성 기준: 2026-10-18, Phase 8.4 최종 의미 AI 검수 병합 완료본까지. 본 보고서는 저장된 실험 보고서·평가 JSON·검수 산출물을 종합한 사후 보고서다. 새 검색, GPU 추론, relevance 판정은 수행하지 않았다. 별도 [요약본](EXPERIMENT_EXECUTIVE_SUMMARY.md)을 함께 제공한다.

## 1. 종합 판단

이번 연구의 가장 중요한 결과는 특정 reranker의 승리가 아니라, **회수 실패·후보 내부 순위 오류·평가 라벨 오류를 구분해야 모델이나 알고리즘을 제대로 선택할 수 있다는 것**이다.

검증된 사전 expansion은 일부 정답의 순위를 올렸지만 같은 용어를 포함한 hard negative도 함께 올렸다. Reranker는 이 중 일부 의미 판별 오류를 교정했으나, 이미 BM25가 찾은 충분한 답변을 강등하기도 했다. BGE와 Qwen의 우열은 영어/한국어, 질의 표현, 문서 구성, relevance 기준에 따라 달라졌다. 쉬운 synthetic에서의 BGE 만점이나 어려운 synthetic에서의 Qwen 우세 어느 쪽도 자연 문서의 일반 성능으로 이어지지 않았다.

최신 Phase 8.4에서는 BM25+BGE RRF가 BM25보다 Strict Top-1에서 **41/80 → 48/80**, correction 9건·regression 2건을 보였다. 개선 방향은 유망하지만 사전등록 기준상 **우월성 미확증(inconclusive)**이다. Exact McNemar p=0.06543이 사전 기준 0.05를 통과하지 못했다. 관련성을 폭넓게 인정하는 Lenient 기준의 61/80 → 73/80 개선은 보조 결과이며 primary 성공을 대신하지 않는다.

따라서 현재 실용적 판단은 다음과 같다.

- BGE는 한국어 서비스의 비용 효율적 비교 기준으로 유지할 가치가 있다. 서비스에 이미 배포돼 있다는 뜻은 아니다.
- BM25+BGE RRF는 BM25의 성공을 보존하는 다음 검증 후보로 유망하다. 배포 우월성은 아직 확정하지 않는다.
- Qwen은 일부 복잡한 조건·의도 전환을 보완하지만, 어떤 질의에 추가 실행할지를 성공적으로 결정하는 정책은 확보하지 못했다.
- 다음 우선순위는 소수 핵심 사례의 전문가 판정과 source가 다른 평가다. 현재 라벨로 즉시 fine-tuning하거나 expansion을 확대할 근거는 부족하다.

## 2. 연구 질문과 실험의 확장 과정

출발점은 전문 도메인 검색에서 약어·동의어 expansion이 recall을 높이는지, 의미를 바꾸는지, BM25와 Dense에 다른 expansion을 적용하는 것이 유리한지, candidate budget에 따라 효과가 달라지는지였다. 초기 관찰에서 회수 개선과 의미 판별이 분리되면서 후속 실험의 초점이 reranker, 한국어 robustness, relevance 신뢰도, BM25 성공 보존으로 이동했다.

| 단계 | 실제 수행한 작업 | 핵심 규모 | 해석 범위 |
|---|---|---|---|
| 초기 설계·A단계 | protected span, 독립 scoring/evaluation, synthetic와 단위 테스트 | 60문서, 12 family, 36질의 | 기능·평가 코드 검증 |
| 최소 CPU | BM25 O/D, Dense O, Hybrid O/O·D/O | 동일 corpus·query, K=10/20/50 | expansion의 제한적 회수 효과 |
| BM25 ablation | F/N/P 8조건 × O/D | 4 family, 12질의 | 구현 민감도·점수 원인 분석 |
| Phase 1–2B | 공개 데이터 권리·ID·tokenizer·adapter 검증 | CaseHOLD/SciFact/ArguAna | 추론 전 데이터·입력 검증 |
| Phase 2C | BGE/Qwen 고정 후보 비교 | 7,020질의, 60,690 pair/model | 영어권 benchmark 재정렬 |
| Phase 3 | 기존 기술 contrast reranking | 12 family, 36질의, 108 pair/model | provisional 토큰 판별 |
| Phase 4 | GTE/Mixedbread 추가 및 실패 ablation | 기존 60,798 pair/model 재사용 | 4모델 품질·자원 비교 |
| Phase 5 | 한국어 benchmark와 표현 변형 | 919 원질의, robustness 100 family | 한국어·질의 표현 민감도 |
| Phase 6 | 상호 보완성 분석·쉬운 한국어 contrast | 40 family, 120질의 | oracle와 실제 fusion 구분 |
| Phase 7 | 오류 중심 relevance audit·hard contrast | audit 76질의/154pair, contrast 40 family | 라벨 오류·난도 영향 |
| Phase 8 | Kubernetes 자연 문서·routing·latency | 1,887 chunks, 120질의 | 자연 문서에서의 재검증 |
| Phase 8.1–8.3 | 독립 세션 AI 검수·전체 순위 복원·7전략 비교 | 목적 표본 40질의, 367 검수 pair | graded relevance와 성공 보존 |
| Phase 8.4 | 사전등록·잔여 80질의·의미 재검수 | primary 80질의, Top-1 고유 108pair | 고정 RRF의 retrospective holdout |

초기 계획의 LLM Expansion, Dense expansion과 대칭/비대칭 전체 비교는 완료하지 않았다. D/O와 O/O 비교만으로 모든 channel-asymmetric 전략의 우월성을 판단할 수 없다. HUPD는 권리 확인 문제로 본문·trap 평가를 진행하지 않았다. Qwen 4B는 최종 비교 모델이 아니며, 실제 주 비교는 0.6B였다. Fine-tuning·ES 연동도 이번 결과에 포함하지 않는다.

## 3. 평가 규약과 증거의 신뢰도

### 3.1 서로 다른 지표를 구분한 이유

**Retrieval Recall@K**는 알려진 관련 문서 중 후보 안에 들어온 비율이다. **Candidate-contained 비율**은 관련 문서가 적어도 하나 들어온 질의 비율이다. 여러 정답이 있는 데이터에서는 두 값이 다르다. **Reranking**은 후보의 순서만 바꾸므로 동일 Top-20을 전부 재정렬하면 Recall@20은 불변이다.

**Pairwise accuracy**는 target가 비교 문서보다 높은 점수를 얻으면 1, 동점이면 0.5다. **Deterministic Top-1**은 점수 동점을 candidate ID로 해소한 결과이고, **strict preference**는 실제 점수 우위를 요구한다. **Strict flip**은 q0와 qDIR 양쪽에서 바뀐 의도에 맞는 target를 엄격히 우선하는지 측정한다. 같은 Top-1이 유지되는 qINV stability는 두 질의가 모두 오답이어도 높을 수 있으므로 ‘둘 다 정답’과 구분했다.

Graded relevance에서는 2=충분한 답변, 1=부분 답변, 0=비관련, U=미판정/판단 불가다. Strict는 2만, Lenient는 1·2를 성공으로 센다. Graded Top-1은 등급 평균이다. U를 0으로 바꾸지 않고 공통 판정 가능 표본에서 전략을 비교해야 한다. nDCG@10은 상위10의 label뿐 아니라 IDCG 계산 대상의 relevance도 필요하므로 Top-10 일부 검수만으로 확정할 수 없다.

RRF는 `sum(1/(60 + rank))`로 계산한다. 모델 간 raw score를 더하거나 평균하지 않는다. Primary Phase 8.4에서는 동일 후보·고정 k·candidate_id tie-break를 사용했다.

### 3.2 라벨의 계보

| 라벨 출처 | 사용 단계 | 신뢰도 및 한계 |
|---|---|---|
| 공개 benchmark 기존 label/qrels | Phase 2C, 4, 5, 6 Part A | 기존 정의 존중. 미판정 문서·중복·orphan·변환 데이터 한계 존재 |
| 작성 의도에 따른 synthetic label | 초기·Phase 3·6·7 contrast | 인간 검수 없음. 템플릿과 생성 과정의 편향 |
| 작성 AI QC | 초기·synthetic 후속 | 작성자와 독립되지 않음. 구조 검사와 의미 정확성은 다름 |
| Qwen 자동 judge | Phase 7 relevance audit | 평가 모델 자기검수 및 Relevant 편향. 확정 성능에 사용 불가 |
| source-derived target | Phase 8 최초 평가 | 질의 생성 출처일 뿐 유일·충분한 정답을 보장하지 않음 |
| 별도 ChatGPT 세션 블라인드 AI 검수 | Phase 8.1 이후 | 모델 순위 노출을 줄였으나 인간 gold가 아님 |
| lexical 보조 draft 후 의미 재검수 | Phase 8.4 | 최종 primary는 의미 검수로 완성. draft와 최종 버전을 분리 |

이 프로젝트가 생성한 contrast와 Kubernetes relevance의 인간 검수는 완료되지 않았다. 공개 데이터셋의 기존 인간 annotation과 프로젝트 자체 검수 상태를 혼동하지 않는다. 블라인드화·인용 exact match·checksum은 편향과 연결 오류를 줄이지만 판정 내용의 진실성을 보증하지 않는다.

## 4. 초기 Query Expansion과 BM25 구현 분석

### 4.1 최소 CPU 결과

사전 expansion은 36질의 중 9개에만 적용됐다. f04 RPO/RTO, f05 ACL, f09 TLS의 q0/qINV/qDIR이며 나머지 27개는 미적용이다. 적용 9개에 대한 AI QC에서는 의미 변화가 관찰되지 않았으나 표본이 작고 인간 검수가 없어 일반 drift rate로 해석하지 않았다.

BM25는 원문 boost 2·alias 1, exact phrase 기반의 실험용 weighted synonym-max scorer였다. 일반 bag-of-words BM25나 Elasticsearch synonym/dis_max와 동일하지 않다. Dense는 all-MiniLM-L6-v2 ONNX CPU 모델이었다. Hybrid는 두 full-corpus 순위를 RRF로 합친 뒤 최종 K를 유지했다. 60문서이므로 K=100은 N/A, K=50의 포화는 주요 결과에서 제외했다.

| 방법 | 4-family contrast pairwise | Contrast target Top-1 | AI-QC known Recall@10 | AI-QC known nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 Original | .4167 | .1667 | .8333 | .4595 |
| BM25 Dictionary | .4167 | .1667 | .8333 | .4703 |
| Dense Original | .7500 | .6667 | 1.0000 | .7411 |
| Hybrid O/O | .5833 | .4167 | .9167 | .6616 |
| Hybrid D/O | .5833 | .4167 | 1.0000 | .6893 |

Known-qrels 지표는 36개 AI-QC label에 의존하며 @10 coverage가 약24–30%에 불과하다. 전체 draft180 label을 쓰면 BM25 nDCG 효과의 방향까지 달라졌다. 의미 판별 pairwise가 그대로인 점을 recall 개선과 함께 읽어야 한다.

대표적으로 f04 qDIR의 RTO 정답 d033은 BM25 41→15위, Hybrid 12→7위로 상승했다. 그러나 hard negative d050도 BM25 8→3위, Hybrid 4→2위로 상승했다. expansion 적용 질의의 draft relevant18pair 중 BM25는 개선3·악화2·불변13이었고 개선은 f04에 집중됐다. 문장 의미를 보존한 expansion도 검색 오류를 늘릴 수 있다. 이것을 expansion 자체의 semantic drift와 동일시하면 안 된다.

### 4.2 f04의 실제 점수 원인

Dictionary qDIR에서 d033=4.178233, d050=9.148708이었다. d050의 `how` 그룹 기여만 4.806304점이었다. 두 문서가 동일 alias를 포함하고 scorer가 부정·답변성을 판별하지 않으므로 alias 추가가 양쪽을 강화했다. 점수 분해를 통해 확인한 현상이며 단순히 ‘lexical anchoring 같다’고 추측한 것이 아니다.

### 4.3 F/N/P ablation

F는 보호 span 밖 독립 unigram 기능어 그룹 제거, N은 제한된 영어 정수·시간 단위 표기 정규화, P는 `token_sum_replacement`다. P는 phrase 점수를 token별 BM25 합으로 바꾸므로 TF·IDF·누적 방식이 함께 달라진다. F는 P=1에서도 phrase 내부 기능어를 유지했다.

| 조건 | Original pairwise | Dictionary pairwise |
|---|---:|---:|
| Baseline | .4167 | .4167 |
| F | .4792 | .5000 |
| N | .4583 | .4583 |
| P | .5417 | .5417 |
| F+N | .5208 | .5625 |
| F+P | .5833 | .6250 |
| N+P | .5417 | .5417 |
| F+N+P | .5833 | .6250 |

F만으로 f04 RPO는 동점, RTO는 여전히 hard negative 열세였다. F+N의 RTO는 양쪽 7.635452점으로 동점이다. ID 순서에 따른 정답 상위 배치는 교정으로 세지 않았다. F+P의 RTO margin은 +.038619로 역전됐지만 RPO 오류는 남았다. f02는 F/FN에서 family accuracy .8333→.1667로 악화됐다. f01 qINV는 F에서 60문서가 모두 0점이 됐으므로 겉보기 회수 개선에 의미를 부여하지 않았다.

결론은 전처리·scorer 민감도 확인이지 최적 scorer 선정이 아니다. F/N/P는 후속 reranker 주 비교 baseline을 대체하지 않았다. 근거: [CPU 결과](CPU_MINIMAL_20261011.md), [점수 재분석](CPU_REANALYSIS_20261011.md), [ablation](BM25_ABLATION_REPORT.md).

## 5. 공개 benchmark와 4개 reranker

### 5.1 데이터 검증과 후보 상한

CaseHOLD는 5,314 test문항, 각5 holding을 모두 평가했다. SciFact는 5,183문서·300질의이며 Top-20 macro recall .8027, 관련 문서 포함 질의247/300이다. ArguAna는 8,674문서·1,406질의, 포함1,084/1,406이다. SciFact는 증거 문서 relevance이며 support/refute 구분 자체의 검사가 아니다. ArguAna는 반론 검색이며 모든 non-gold가 찬성 논거라는 뜻은 아니다.

CaseHOLD 중복 distractor7건, ArguAna orphan5건·동일 query본문108쌍을 보존하고 sensitivity를 별도로 계산했다. ArguAna self-match는 사전 규칙으로 제외했다. 데이터 감사와 tokenizer/mock adapter 검증 후 GPU 실행을 진행했다. 초기 접속·CUDA 장애는 준비 단계의 이력이며 최종 추론 완료 여부와 구분했다.

### 5.2 네 모델의 주 결과

| 모델 | CaseHOLD Accuracy@1 | SciFact nDCG@10 | ArguAna nDCG@10 |
|---|---:|---:|---:|
| BGE v2-m3 | .5346 | .7170 | .5558 |
| Qwen3 0.6B | .6159 | .7253 | .6423 |
| GTE multilingual base | .5994 | .7165 | .5313 |
| Mixedbread base v2 | .6317 | .7394 | .5812 |
| 초기 BM25 | 해당 없음 | .6430 | .4152 |

Qwen−BGE의 SciFact 차이 +.00826은 CI [−.00882,.02514]로 불확실했다. CaseHOLD는 Qwen-only1,009/BGE-only577, exact McNemar p≈1.28e−27이었다. ArguAna 동일-text cluster 분석의 nDCG 차이 +.08848, CI [.07382,.10303]은 Qwen 쪽이었다. Phase 4의 Mixedbread−Qwen CaseHOLD 차이 +.0158, CI [.0024,.0294]는 탐색 결과이며 다중 모델·데이터 비교 전체를 보정한 확증 결론은 아니다.

| 모델 | SciFact correction/regression | ArguAna correction/regression |
|---|---:|---:|
| BGE | 46 / 9 | 349 / 104 |
| Qwen | 48 / 10 | 466 / 41 |
| GTE | 44 / 9 | 284 / 91 |
| Mixedbread | 53 / 5 | 371 / 86 |

CaseHOLD/SciFact에서는 Mixedbread가, ArguAna에서는 Qwen이 높은 값을 보였다. 영어 benchmark의 한 모델 순위를 한국어 실서비스 순위로 가져오지 않았다. 모든 값은 고정 후보 내부 결과이며 Recall@20은 불변이다.

### 5.3 기술 토큰 contrast와 실패 원인

12 family의 3-way 진단은 모델당108pair였다.

| 모델 | Pairwise | Target Top-1 | qDIR Top-1 | Strict flip |
|---|---:|---:|---:|---:|
| BGE | .8889 | .8056 | .5833 | .6667 |
| Qwen | .9236 | .8889 | .9167 | .6667 |
| GTE | .8889 | .7778 | .7500 | .5833 |
| Mixedbread | .9444 | .8889 | .8333 | .7500 |

Qwen은 RTO, disable, TLS1.3-only, IPv6 qDIR에서 BGE 오류를 교정했지만 client/server certificate에서는 Qwen만 실패했다. authentication/authorization 및 allow/deny에는 공통 실패가 남았다. Qwen f04 q0는 동점이므로 deterministic Top-1과 엄격한 성공을 구분했다. Qwen−BGE pairwise의 family CI는 [−.0972,.1528]로 작은 표본의 우월성은 미확증이다.

실패7family 대상 별도 ablation에서는 핵심 토큰 masking, core-only, 공통 문장 제거를 실제 추론했다. Masking/core-only는 원래 의미를 훼손하므로 accuracy 실험으로 해석하지 않았다. 공통 문장 제거는 BGE correction1/regression2, Qwen1/1로 일관된 개선이 아니었다. 기존 score에서 hard negative만 제거하면 BGE f06·f09 qDIR 두 오류가 교정됐지만 다른 opposite-intent 오류는 남았다. 이는 ‘토큰을 전혀 이해 못함’ 또는 ‘공통 문맥만 문제’라는 단일 원인 설명이 부족함을 보여준다.

근거: [Phase 2C](PHASE2C_GPU_RERANKER_RESULTS.md), [Phase 3](PHASE3_DOMAIN_TOKEN_RESULTS.md), [Phase 4 비교](PHASE4_RERANKER_COMPARISON.md), [실패 ablation](PHASE4_FAILURE_ANALYSIS.md).

## 6. 한국어 benchmark와 질의 표현 robustness

한국어에서는 영어 regex tokenizer를 그대로 쓰지 않고 Hangul 어절+문자 bigram, Latin/숫자 보존, title2/text1의 SQLite FTS5 adapter를 사용했다. 최초 어절-only feasibility 결과는 보존했고 gold 삽입 없이 v2 후보를 동결했다.

| 데이터 | 원질의 | Corpus | Top-20 포함 질의 | Known macro Recall@20 |
|---|---:|---:|---:|---:|
| MIRACL-ko | 213 | 1,486,752 | 143/213 | .5035 |
| AutoRAGRetrieval | 114 | 720 | 114/114 | 1.0000 |
| Ko-StrategyQA | 592 | 9,251 | 487/592 | .7358 |

당시 metadata 기준 MIRACL은 Apache-2.0, AutoRAG는 MIT였고 Ko-StrategyQA HF card에는 라이선스 선언이 없었다. 후자의 상업 사용·재배포는 별도 권리 확인 대상이다. MIRACL의 공식 judged pool 보조 실험은 label-selected이므로 BM25 Top-20 주 실험과 합치지 않았다.

| 모델 | MIRACL nDCG@10 / Top-1 | AutoRAG nDCG@10 / Top-1 | Ko-StrategyQA nDCG@10 / Top-1 |
|---|---|---|---|
| BGE | .4966 / .5446 | .9647 / .9123 | .7155 / .7314 |
| Qwen | .5079 / .5775 | .9283 / .8509 | .7112 / .7230 |
| GTE | .4981 / .5493 | .9088 / .7807 | .7005 / .7010 |
| Mixedbread | .4895 / .5258 | .9373 / .8596 | .7035 / .7044 |

MIRACL은 Qwen, AutoRAG와 Ko-StrategyQA는 BGE의 수치가 높았다. Qwen−BGE nDCG CI는 MIRACL [+.00087,+.02210], AutoRAG [−.06703,−.00844], Ko-StrategyQA [−.01037,+.00174]다. Top-1의 세 데이터셋 McNemar Holm 보정 p는 모두 .05 이상이었다. 지표별 검정과 의미를 구분해야 한다.

Candidate-contained 100 family에서 Natural은 원질의, Keyword는 제한적 조사·종결형 제거, Verbose는 새로운 사실 없는 고정 wrapper였다. 자동 의미 점검만 거쳤다.

| 모델 | Keyword nDCG@10 | Natural | Verbose | Verbose−Natural |
|---|---:|---:|---:|---:|
| BGE | .8360 | .8427 | .8226 | −.0200 |
| Qwen | .8255 | .8431 | .8308 | −.0123 |
| GTE | .8063 | .8157 | .7897 | −.0260 |
| Mixedbread | .8237 | .8288 | .7899 | −.0389 |

Qwen의 Keyword 손실 CI [−.0341,−.0029], Mixedbread의 Verbose 손실 CI [−.0611,−.0184]가 0을 제외했다. BGE의 두 변형 CI는 0을 포함했다. Specific은 조건을 추가하므로 기존 qrels로 성능을 확정하지 않고 순위 변화만 기록했다.

원질의 어절 수 ≤4/5–6/≥7 집단에서 긴 질의의 평균 성능이 높았지만 데이터셋·정보량·후보 유입률이 섞여 있어 길이의 인과 효과가 아니다. Verbose도 길이와 표현 변경이 동시에 발생하므로 ‘길기 때문에 성능 하락’으로 단정하지 않았다. 근거: [Phase 5 결과](PHASE5_KOREAN_RERANKER_RESULTS.md), [데이터 검토](PHASE5_DATA_AND_READINESS.md). 후자의 GPU blocked 표기는 준비 당시 이력이며 최종 실행은 완료됐다.

## 7. 한국어 상호 보완성과 synthetic 난도 변화

### 7.1 기존 benchmark의 보완 가능성

| 데이터 | Both correct | BGE only | Qwen only | Both wrong | Oracle Top-1 |
|---|---:|---:|---:|---:|---:|
| MIRACL 213 | 112 | 4 | 11 | 86 | .5962 |
| AutoRAG 114 | 94 | 10 | 3 | 7 | .9386 |
| Ko-StrategyQA 592 | 410 | 23 | 18 | 141 | .7618 |
| 합계 919 | 616 | 37 | 32 | 234 | 685/919 |

BGE 단독653정답, Qwen648정답, oracle685정답이다. BGE 대비 oracle 추가 이득은32/919=3.48%p지만 실제 router가 label 없이 이를 회수할 수 있다는 뜻은 아니다. 고정 BGE+Qwen RRF는653정답으로 BGE와 같았다. Both wrong234개 중175개는 후보에 알려진 정답 자체가 없어 reranker 추가만으로 해결할 수 없다. Qwen-only가 짧은 MIRACL 질의에 집중되는 경향은 다른 데이터셋에서 일관되지 않았다.

### 7.2 쉬운 contrast와 어려운 contrast

| 단계·모델 | Target Top-1 | Pairwise | qDIR Top-1 | Strict flip | qINV Top-1 안정성 |
|---|---:|---:|---:|---:|---:|
| Phase 6 BGE | 100% | 1.0000 | 100% | 100% | 100% |
| Phase 6 Qwen | 95.00% | .9708 | 92.50% | 85.00% | 92.50% |
| Phase 7 BGE | 70.83% | .8458 | 75.00% | 47.50% | 87.50% |
| Phase 7 Qwen | 86.67% | .9313 | 82.50% | 70.00% | 92.50% |

각 단계는40family·120질의·360pair/model이다. Phase6는 명시적 대상 표식과 비응답임이 드러나는 HN을 포함해 쉬웠다. Phase7은 반대 조건 동시 언급, 조건·조치의 문장 분산, 점검/정상 모드 구분, 82–128어절 문서와 뒤쪽 근거를 추가했다. 문서 간 lexical overlap 평균 .5973이었다.

Phase7은 Both correct75, BGE-only10, Qwen-only29, Both wrong6이다. Qwen−BGE Top-1 차이 +15.83%p, family bootstrap CI [+5.0,+26.67]%p다. 수치 조건은 두 모델 모두66.67%, action은 모두77.78%로 Qwen 우세가 모든 유형에 걸친 것은 아니다. hk17의 10MB 경계, hk24의 요청자/승인자, hk28의 연결 상태가 공통 약점이었다. hk12 삭제 금지에서는 BGE만 성공하는 사례도 있었다.

이 비교는 난도·길이·조건 배치·템플릿이 함께 바뀐 것이므로 Qwen의 ‘긴 문맥 능력’ 하나를 분리한 인과 실험이 아니다. 사후 qINV 조사25건을 교정한 v4 sensitivity에서 Qwen 전체는86.67→85.83%, BGE는불변이었다. 이를 primary v3와 섞지 않았다.

### 7.3 Phase 7 relevance audit의 실패와 교훈

오류 중심76질의154pair를 자동 판정했지만 Relevant134, Partial1, U19, Non-relevant0이었다. 평가 대상과 동일한 Qwen checkpoint를 판정기로 썼고 근거 문장은 표면 중복 기반으로 뽑았다. 이 결과는 독립 검수가 아니며 corrected benchmark로 채택할 수 없다.

다만 MIRACL105의 미판정 문서가 “1939년부터”를 직접 포함하고 AutoRAG103_commerce의 미판정 문서가 아바톡 기능을 설명하는 사례는 qrels 누락 점검의 필요성을 보여줬다. 이후 Phase8에서 평가 모델 자기검수를 피하고 별도 블라인드 검수로 전환한 이유다.

근거: [Phase6 상호 보완성](PHASE6_KOREAN_COMPLEMENTARITY.md), [Phase6 contrast](PHASE6_KOREAN_DOMAIN_CONTRAST.md), [Phase7 contrast](PHASE7_HARD_KOREAN_CONTRAST.md), [Phase7 audit 한계](PHASE7_RELEVANCE_AUDIT.md).

## 8. Kubernetes 자연 문서와 selective routing

### 8.1 데이터 현실성과 남은 생성 편향

Kubernetes 한국어 공식 문서567파일에서 H2–H4 섹션을 추출해1,887chunk를 만들었다. 원본 commit은 `2f85f2802e59eb2128543dbfbf114ad641db4be0`, 수집일은2026-10-17, 기록된 라이선스는 CC BY4.0이다. Chunk는180–5,000자, overlap없음이며 코드 블록·Hugo shortcode를 제거했다.

120질의는 정의/조건89, 절차28, 장애대응3으로 실제 사용자 로그가 아닌 자동 생성 표본이다. 원문 heading에서 파생했으므로 title boost BM25와 Keyword에 유리할 수 있다. Source target가 BM25 Top20에 포함된 질의는110/120이지만 이것은 완전 relevance coverage가 아니다. Natural/Keyword 동일후보 총4,800pair/model을 추론했고 truncation은0이었다.

### 8.2 최초 target-only 결과의 의미

| Natural 120질의 | BM25 | BGE | Qwen | BGE+Qwen RRF |
|---|---:|---:|---:|---:|
| source-target Top-1 | .592 | .517 | .558 | .583 |
| target-only nDCG@10 | .714 | .709 | .747 | .760 |

이는 source target 하나만 정답으로 놓은 sensitivity다. Qwen−BGE Natural nDCG +.0374의 source-group CI는[−.0125,+.0890]이었다. Keyword Top-1은 BGE .858, Qwen .658로 반대였다. target 길이618자 이상 구간도 BGE66.7%, Qwen38.5%로 Phase7의 긴 문맥 우세를 단순 재현하지 않았다. 독립 evidence annotation은 없었다.

### 8.3 Router는 실제 이득을 확보했는가

Source 단위 train64/validation35/test21로 나누고 label-free query 특징·후보 길이·BGE margin을 썼다. Validation에서 선택된 규칙은 평균 후보 길이≤715.4자일 때 Qwen을 실행하고 RRF를 반환하는 것이었다.

| Test21 정책 | target-only Top-1 | Qwen 호출률 | BGE 대비 correction/regression |
|---|---:|---:|---:|
| BGE | .714 | 0% | — |
| Qwen | .667 | 100% | — |
| Always BGE+Qwen RRF | .667 | 100% | — |
| Selective RRF | .667 | 95.2% | 0 / 1 |
| Oracle | .810 | 정답 이용 | 실현 가능한 정책 아님 |

호출률은 저장 Qwen 점수로 계산한 사후 정책 시뮬레이션이다. 물리적 GPU 호출을95.2%만 실행한 것이 아니다. 호출을 거의 줄이지 못했고 품질도 낮아 이 정책은 성공하지 못했다. 이후 AI relevance가 보완됐지만 완전 검수된 동일 routing test의 확증 결과를 새로 확보한 것은 아니다.

근거: [Phase8 기본 평가](PHASE8_REALISTIC_KOREAN_EVALUATION.md), [routing·요청 latency](PHASE8_SELECTIVE_ROUTING.md).

## 9. Relevance 재검수가 바꾼 결론: Phase 8.1–8.3

### 9.1 40질의·367pair의 블라인드 AI 검수

기존1,359pair review pool에서40질의를 목적 표집하고367pair를 별도 ChatGPT 세션이 검수했다. Label은2=67,1=134,0=155,U=11이다. 기존 target3건은2에서1로 낮아졌다. Source section이 유일하거나 충분한 정답이라는 가정이 깨진 사례다.

39개 비-U Top-1 표본에서 BGE는 Strict22/39, Qwen19/39지만 Lenient는29/39 대34/39였다. Strict 교차표는 Both11/BGE-only11/Qwen-only8/Both wrong9, Lenient는26/3/8/2였다. 충분한 답변과 부분 답변을 구분하면 모델 선택 방향도 달라진다.

**Phase8.1의 초기 bootstrap은 최종 증거로 재사용하지 않는다.** Phase8.3에서 U를 binary0으로 바꾼 통계 처리와 query source가 아닌 첫 candidate source로 묶은 오류를 발견했다. Top-1 원시 count와 별개로 기존 CI는 폐기 대상으로 구분하고, 올바른 query-source·U제외 방식의 Phase8.3/8.4를 사용했다. U를 전부0 또는2로 둔 nDCG도 엄밀한 lower/upper bound가 아니다.

### 9.2 전체 Top20 복원 후 7전략 비교

Phase8.2에는 완전한367label/Top20이 연결되지 않아 공통31질의만 비교했다. Phase8.3은40×20순위를 복원하고7전략의 Top-1이 모두 비-U인 공통38질의를 사용했다.

| 전략 | Strict | Lenient | Graded | BM25 대비 Strict C/R |
|---|---:|---:|---:|---:|
| BM25 | 27/38 | 30/38 | 1.500 | — |
| BGE | 22/38 | 29/38 | 1.342 | 3 / 8 |
| Qwen | 19/38 | 33/38 | 1.368 | 3 / 11 |
| BGE+Qwen RRF | 25/38 | 33/38 | 1.526 | 3 / 5 |
| BM25+BGE RRF | 28/38 | 35/38 | 1.658 | 3 / 2 |
| BM25+Qwen RRF | 27/38 | 35/38 | 1.632 | 5 / 5 |
| 3-way RRF | 28/38 | 35/38 | 1.658 | 4 / 3 |

BM25+BGE의 Strict 차이는 +2.6%p, CI는 [−7.9,+13.2]%p였다. Lenient 차이는 +13.2%p, CI는 [+2.6,+26.3]%p였다. 목적 표본의 사후 비교이므로 배포 우월성으로 해석하지 않는다.

BM25가2점 문서를1위에 둔27질의에서 같은 문서의1위 유지는 BGE17, Qwen14, BM25+BGE25건이었다. BGE Strict regression8건은2→0이6건,2→1이2건, Qwen11건은각3건·8건이었다. BM25+BGE의 regression2건은모두2→1이었다.

대표적으로 q008은 파드 안 명령 실행 대신 kubectl의 실행 환경·인증 문서를, q061은 기존 연결에 대한 NetworkPolicy 조건 대신 일반 네트워크 문서를, q090은 세션 고정 timeout 대신 deployment deadline을 선택했다. Qwen q071은 Secret 타입 전체 질문에 Basic authentication 한 타입을 선택했다. ‘주제는 관련되나 요구 속성·범위가 불충분’한 오류가 반복됐다.

BM25+BGE도 q028 백오프 정책과 q117 GatewayClass에서 부분 답변으로 강등했다. BM25 Top1이 BGE Top3 안이면 보존하는 탐색 규칙은 소규모 test7에서 BM25와6/7로 같고 correction0이었다. 복잡한 정책의 실익은 입증되지 않았다.

이 단계의 nDCG는 전체 후보 미검수로 확정할 수 없었다. Candidate20 내 보수적 평균 범위는 BM25 [.351,.978], BM25+BGE [.399,.965]로 크게 겹쳤다. 근거: [Phase8.3 최종 복원 분석](PHASE8_3_SUCCESS_PRESERVATION.md). Phase8.1 보고서는 역사적 결과로만 함께 보존한다.

## 10. Phase 8.4: 사전등록과 최종 의미 검수 결과

### 10.1 독립성의 실제 수준

기존40질의를 제외한80질의를 holdout으로 고정했다. Primary는 BM25 대 BM25+BGE RRF(k60), Strict Top-1이다. 같은 source의 유사 질의는 cluster로 처리했다.21질의는 기존40과 source가 겹치고59질의는 query source가 분리됐다. 모든 후보는 동일 corpus이며120질의의 점수·일부 집계가 이미 존재했으므로 외부 신규 test와 동등한 독립성은 없다.

Holdout434pair와 reliability106pair를 별도 블라인드 패키지로 만들었다. 최초 제출540pair는 lexical 보조 draft 성격이 확인돼 그대로 확정하지 않았다. 이전40의 reliability 재검수는 diagnostic이며 holdout 우월성의 증거로 합산하지 않았다.

### 10.2 검수 버전과 결론 변화

| 버전 | Primary BM25 Strict | RRF Strict | Correction/regression | Exact p | 해석 |
|---|---:|---:|---:|---:|---|
| lexical 보조 draft | 44/80 | 53/80 | 10 / 1 | .01172 | 수치상 통과하나 라벨 신뢰도 부적합 |
| 95pair 의미 검수+잔여 draft | 46/80 | 53/80 | 9 / 2 | .06543 | 혼합 라벨 sensitivity |
| 최종 의미 검수 | 41/80 | 48/80 | 9 / 2 | .06543 | primary 완료·우월성 미확증 |

최종 95+52=147pair를 의미 검수 결과로 연결했고 기존 lexical draft와 63pair가 달라졌다. 새 52pair는 두 전략이 같은 Top-1을 선택한 질의였으므로 양쪽 절대 성공률이 동시에 5/80 낮아졌지만 paired 차이는 불변이다. Primary 80질의의 고유 108개 Top-1 pair는 모두 명시적 비-U 의미 검수다.

### 10.3 최종 수치와 사전등록 판정

| 기준 | BM25 | BM25+BGE | 차이 | Source-cluster 95% CI | C/R |
|---|---:|---:|---:|---|---:|
| Strict | 51.25% | 60.00% | +8.75%p | [+1.23,+16.67]%p | 9 / 2 |
| Lenient | 76.25% | 91.25% | +15.00%p | [+6.49,+23.75]%p | 13 / 1 |
| Graded | 1.2750 | 1.5125 | +.2375 | [+.1013,+.3797] | 16 / 3 |

80질의·73source cluster,10,000bootstrap이다. Strict exact McNemar p=.0654296875. 사전등록은 양의 차이·CI 하한>0·p<.05를 함께 요구했다. Bootstrap CI가0을 제외해도 별도의 exact 검정 기준을 통과하지 못했으므로 **inconclusive**를 유지한다. 두 방법의 동등성이 입증됐다는 의미도 아니다. Lenient p=.001831은 보조 탐색 결과다.

전이는2→2=39,2→1=2,2→0=0,1→2=3,1→1=16,1→0=1,0→2=6,0→1=7,0→0=6이다. Grade2를 완전히 무관한0으로 교체한 사례는 없었지만 부분 답변 강등은 남았다.

기존 40개와 query source가 다른 59질의·52cluster에서도 Strict 31/59→37/59, correction 7/regression 1, 차이 +10.17%p, CI [+1.69,+19.30]%p, p=.0703125로 미확증이다.

최종 Strict regression은 q096(확장 리소스 quota의 구체 설정 대신 일반 확장 리소스 설명), q115(감사 log backend 관련 특정 설정 대신 log/webhook batching 설명)이며 모두2→1이다. 질문 범위와 부분/충분 relevance 경계에 전문가 합의가 필요한 사례다.

### 10.4 완료와 미완료의 경계

Primary80 검수·병합·평가는 완료됐다. 전체 mapping540행 중147행이 최종 의미 검수이며393행은U로 남는다.393행은 primary 누락이 아니라 secondary/ranking/reliability의 잔여 범위다. BGE49, Qwen48 등 secondary의 eligible n이 다르므로 그 평균으로 모델 순위를 다시 정하지 않는다. 전체20후보가 검수된 질의는0개이므로 exact nDCG@10은 없다.

Raw Markdown 연결은540개 review pair 모두 가능했다. 대응 source section에서 코드 블록173건, 표16건, shortcode271건을 발견했다. 이는 review pair별 중복 포함 집계이며 고유 문서 수나 모든 사례의 답변 손실 건수가 아니다. 실제 모델이 본 chunk relevance를 primary로, 원문 보충 문맥을 diagnostic으로 분리했다. ‘원문에 답이 있음’이 ‘모델 입력에 답이 있음’을 보장하지 않는다.

근거: [최종 의미 검수 보고서](PHASE8_4_FINAL_SEMANTIC_EVALUATION.md), [최종 평가 JSON](../runs/phase8_realistic_korean_v1/phase8_4_review_results_v3/evaluation.json), [80질의 비교 원문](../runs/phase8_realistic_korean_v1/phase8_4_review_results_v3/primary_query_comparison.csv).

## 11. 추론 효율과 운영 가능성

실제 장치는 A10080GB 전체가 아니라 MIG1g.10gb 할당이었다. Phase2C에서 PyTorch가 보고한 총VRAM은10.20GB였다. 모델은 순차 로드했고 BF16·operational cap2048을 사용했다. BGE는 scalar classification logit, Qwen은 공식 template의 final yes/no logit 차이를 사용했다. 모델별 pinned revision과 GTE remote-code 감사는 각 manifest에 보존돼 있다.

| 모델 | Phase4 영어 pair/s | Phase5 한국어 pair/s | Phase5 peak allocated | Phase5 batch |
|---|---:|---:|---:|---:|
| BGE | 40.77 | 59.25 | 2.02GB | 16 |
| Qwen0.6B | 11.85 | 13.03 | 5.63GB | 4 |
| GTE | 67.95 | 101.99 | 1.71GB | 16 |
| Mixedbread | 24.39 | 27.59 | 5.46GB | 4 |

입력 길이·batch·architecture가 달라 hardware-normalized 성능이 아니다. Phase4의 estimated Top10/20은 throughput 역산값이며 온라인 latency로 사용하지 않는다. Phase8 기본 보고서의 ‘약5배 latency’ 표현도 전체 pair/s에 근거한 부분은 처리량 차이로 읽어야 한다.

Phase8에서는 모델별 단독 상주, warm-up5회, 후보 수별40요청, micro-batch4로 실제 scoring latency를 별도 측정했다.

| 후보 수 | BGE P50/P95/P99 | Qwen P50/P95/P99 |
|---|---|---|
| 5 | 99 /185 /218ms | 520 /831 /882ms |
| 10 | 228 /375 /421ms | 1,112 /1,674 /1,790ms |
| 20 | 428 /674 /719ms | 2,220 /3,113 /3,460ms |

이는 해당 GPU·문서·warm조건의 scoring 요청 시간이다. 서비스 네트워크·queue·동시 사용자 SLA와 같지 않다.40요청의 P99는 불안정하며 두 모델 동시 상주·실제 sequential route는 측정하지 않았다. Percentile을 더해 실측 sequential latency라고 보고할 수 없다. Phase8 Qwen reserved9.95GB는 할당 한도에 가까웠다. 과금 단가가 없고 정확 금액 계산은 범위에서 제외돼 총비용을 임의 산정하지 않았다.

## 12. 연구 질문별 최종 답

| 질문 | 지금까지 확보한 답 | 남은 검증 |
|---|---|---|
| 약어·동의어 expansion이 recall을 높이는가? | f04 등 일부 회수 개선, HN 동반 상승. 효과가 한 family에 집중 | 다양한 실제 도메인·사전 provenance·독립 label |
| 의미 drift가 발생하는가? | 적용9질의 AI QC drift0이나 검색 오류와 별개 | 인간 의미 검수, LLM expansion은 미실행 |
| 채널별 다른 expansion이 좋은가? | D/O가 O/O보다 일부 known recall 개선 | Dense expansion·대칭 조건 미비로 일반 결론 불가 |
| K에 따라 달라지는가? | 정답41→15는@20에만, Hybrid12→7은@10에 반영 | 대규모 corpus·실제 ANN budget. K100 미실험 |
| Reranker가 핵심 토큰을 구분하는가? | 다수 성공하나 부정·역할·경계·의도 전환 오류 존재 | 실제 독립 검수 도메인·템플릿 의존 제거 |
| 한국어 최선 모델은 무엇인가? | 데이터별 우열 다름. BGE 효율, Qwen 일부 보완 | 동일 신뢰도 label·실제 서비스 분포 |
| Qwen 선택 실행이 가치 있는가? | oracle 여지는 있으나 시험 router는 개선0/악화1 | 별도 holdout·비용 포함 검증 |
| BM25+BGE RRF가 더 나은가? | 최종80 Strict+8.75%p, C9/R2, primary미확증 | 사람 경계 검수·다른 source 확증 |


## 13. 재현성과 산출물 안내

| 범위 | 대표 원시 결과 | 해석 문서 |
|---|---|---|
| 초기 CPU | `runs/cpu_minimal_20261011/` | `CPU_MINIMAL_20261011.md` |
| BM25 ablation | `runs/bm25_ablation_v1_F*/` | `BM25_ABLATION_REPORT.md` |
| 공개 benchmark | `runs/reranker_phase2c_v1/evaluation/evaluation.json` | `PHASE2C_GPU_RERANKER_RESULTS.md` |
| 4모델 | `runs/phase4_rerankers_v1/evaluation/evaluation.json` | `PHASE4_RERANKER_COMPARISON.md` |
| 한국어 benchmark | `runs/phase5_korean_v1/evaluation_v1/evaluation.json` | `PHASE5_KOREAN_RERANKER_RESULTS.md` |
| 한국어 contrast | `runs/phase6_korean_v1/`, `runs/phase7_korean_v1/` | Phase6/7 contrast 보고서 |
| Kubernetes frozen | `runs/phase8_realistic_korean_v1/prepared_v2/`, `gpu_results/` | Phase8 기본·routing 보고서 |
| 367pair·7전략 | `runs/phase8_realistic_korean_v1/success_preservation_v1/` | `PHASE8_3_SUCCESS_PRESERVATION.md` |
| 최신 primary | `runs/phase8_realistic_korean_v1/phase8_4_review_results_v3/` | `PHASE8_4_FINAL_SEMANTIC_EVALUATION.md` |

초기62테스트, CPU66, ablation71, Phase2A97, Phase3 112, Phase4 준비118 등의 통과 기록은 각 시점의 코드 검증이다. 모두 현재 통합 suite를 새로 실행한 결과라고 합산하지 않는다. Phase8.4의 checksum·ID·매핑·중복·U처리·paired transition 검증도 label의 객관적 정답성을 검증하는 것은 아니다.

보고서의 우선순위는 ‘더 최근 문서’만이 아니라 같은 입력·동일 표본·label 신뢰도·검증된 evaluator다. 특히 Phase8.1 통계는8.3 교정을, Phase8.4 lexical draft는최종 v3를 우선한다. 초기 blocked 문서는 실패 이력으로 남기고 완료된 run을 미실행으로 오인하지 않는다. Synthetic와 benchmark, 목적표본40과holdout80을 합쳐 하나의 종합 accuracy를 만들지 않는다.

현재까지의 성과는 재사용 가능한 평가 pipeline, 원본을 보존하는 비교·검수 절차, 오류 유형별 사례, 실제 GPU 자원 관찰, 고정 RRF의 개선 가능성과 불확실성이다. 회사 검색 성능의 확정적 개선이나 최종 배포 모델 선정은 아직 이 결과가 입증하는 범위 밖이다.
