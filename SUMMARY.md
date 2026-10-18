# Query Expansion Pilot 실험 요약

2026-10-18 기준, 초기 CPU pilot부터 Phase8.4 최종 AI 의미 검수까지. 구체 조건·사례·통계·원시 결과 위치는 [종합 보고서](COMPREHENSIVE_EXPERIMENT_REPORT.md)에 있다.

## 핵심 결론

**BGE는 비용 효율적인 한국어 비교 기준으로 유지할 가치가 있고, BM25+BGE RRF는 다음 검증 후보로 유망하다. Qwen을 항상 또는 선택적으로 추가하는 배포 정책의 이득은 아직 입증하지 못했다.** 최신 결과의 라벨도 사람 검수 없는 provisional AI relevance다.

이번 실험은 ‘어떤 모델이 항상 최고인가’보다 세 가지를 구분하는 데 의미가 있었다. 정답이 후보에 없는 회수 실패, 정답이 있는데 강등하는 순위 오류, 답변 문서를 오답으로 계산하는 라벨 누락은 각각 다른 조치가 필요하다.

## 무엇을 실행했고 무엇을 배웠나

| 단계 | 핵심 관찰 | 실용적 의미 |
|---|---|---|
| Query Expansion CPU pilot | 36질의 중9개 expansion. RTO 정답41→15위, hard negative8→3위 | 회수 개선이 의미 판별 개선을 보장하지 않음 |
| BM25 F/N/P ablation | contrast pairwise .4167→최대 .6250, 다른 family regression 존재 | 전처리·점수 구현 영향이 크며 일반 BM25/ES 우월성은 아님 |
| 영어 4모델 benchmark | Mixedbread가 CaseHOLD/SciFact, Qwen이 ArguAna에서 높은 값 | 태스크별 모델 우열이 다름 |
| 한국어3 benchmark | MIRACL은 Qwen, AutoRAG/Ko-StrategyQA는 BGE가 높은 값 | 영어 순위를 한국어 서비스에 그대로 적용할 수 없음 |
| 쉬운 한국어 contrast | BGE100%, Qwen95% | 쉬운 템플릿의 만점은 일반 능력을 보장하지 않음 |
| 어려운 한국어 contrast | BGE70.83%, Qwen86.67%; Qwen-only29/BGE-only10 | 복잡한 조건에서 보완 가능성, 다만 synthetic·AI label |
| Kubernetes 자연 문서 | Natural Qwen 이득 불확실, Keyword는 BGE 우세 | Qwen의 긴 문맥 일반 우세는 재현되지 않음 |
| Selective Qwen | test21에서 correction0/regression1, 호출95.2% | 시험 정책은 품질·비용 목표를 달성하지 못함 |
| 의미 검수·BM25 성공 보존 | Reranker가 BM25의 충분한 답변을 부분/비관련 문서로 교체 | relevance 검수와 BM25 순위 보존이 중요 |

LLM Expansion, HUPD 본문 평가, Qwen4B 비교, Fine-tuning, ES 연동은 완료한 실험에 포함되지 않는다.

## 네 모델의 주요 비교

숫자는 각 데이터의 동일 frozen 후보에 대한 결과다. 서로 다른 데이터셋을 한 점수로 평균하지 않는다.

| 모델 | CaseHOLD Acc@1 | SciFact nDCG@10 | ArguAna nDCG@10 | MIRACL-ko nDCG@10 | AutoRAG nDCG@10 | Ko-StrategyQA nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BGE | .5346 | .7170 | .5558 | .4966 | .9647 | .7155 |
| Qwen0.6B | .6159 | .7253 | .6423 | .5079 | .9283 | .7112 |
| GTE | .5994 | .7165 | .5313 | .4981 | .9088 | .7005 |
| Mixedbread | .6317 | .7394 | .5812 | .4895 | .9373 | .7035 |

한국어919질의에서는 BGE-only37, Qwen-only32였다. Oracle의 BGE 대비 추가 이득은32/919=3.48%p지만 고정 BGE+Qwen RRF의 정답 수는 BGE와 같았다. Oracle은 정답을 미리 아는 상한이며 구현된 router 성능이 아니다. 공통 실패234개 중175개는 후보 안에 알려진 정답이 없었다.

## 가장 최근의 검증 결과

Phase8.3의 목적표본40개에서 BM25+BGE RRF가 유망해 보여, 나머지80개 질의에 고정된 규칙을 검증했다. 두 전략 Top-1 고유108pair를 모두 의미 AI 검수했고 U로 인한 primary 제외는 없다.

| 기준 | BM25 | BM25+BGE RRF | 개선/악화 |
|---|---:|---:|---:|
| Strict: 충분한 답변2점 | 41/80, 51.25% | 48/80, 60.00% | 9 / 2 |
| Lenient: 부분 답변도 관련 | 61/80, 76.25% | 73/80, 91.25% | 13 / 1 |
| Graded 평균 | 1.2750 | 1.5125 | 16 / 3 |

Strict 차이는 +8.75%p, source-cluster 95% CI는 [+1.23,+16.67]%p다. 그러나 exact McNemar p=.06543으로 사전등록 p<.05 조건을 만족하지 않아 **primary 결론은 우월성 미확증**이다. 개선이 없거나 두 방법이 동등하다는 증명도 아니다.

Strict regression2건은 모두2→1이고2→0은 없었다. 기존40개와 source가 분리된59개에서도31→37정답으로 방향은 같지만 p=.07031로 역시 미확증이다. 이80개는 동일 corpus의 기존 계산 점수를 재사용한 retrospective holdout이며 외부 신규 테스트는 아니다.

초기 lexical 보조 draft는44→53정답으로 유의해 보였지만 의미 재검수 후41→48로 바뀌었다. **라벨 신뢰도가 모델·전략 선정 결론을 바꿀 수 있음**을 보여준다. 전체 후보20개가 완전 검수된 질의는 없어 exact nDCG@10 개선을 추가로 주장하지 않는다.

## 비용과 응답시간

A100 전체가 아니라 MIG1g.10gb 환경에서 실행했다. 한국어 batch 처리량은 GTE101.99, BGE59.25, Mixedbread27.59, Qwen13.03pair/s였다. Batch가 달라 동일 조건의 latency 비교는 아니다.

별도 Kubernetes 요청 측정에서 후보20개 scoring P50/P95는 BGE428/674ms, Qwen2,220/3,113ms였다. 조건당40회라 P99·실서비스 SLA에는 부족하며 두 모델 동시 상주나 sequential 요청은 실측하지 않았다. Qwen의 추가 호출은 명확한 correction 이득이 있을 때 검토할 비용이다. 실제 과금 금액은 산출하지 않았다.

## 다음 행동에 대한 권고

1. 충분한 답변과 부분 답변의 업무상 기준을 합의하고, 회귀2건 및 코드·표 손실 사례부터 전문가가 판정한다.
2. BM25 단독과 고정 BM25+BGE RRF(k60)를 다른 source 또는 내부 보안 환경의 별도 평가셋에서 비교한다.
3. 후보에 정답이 없을 때 retrieval·검증된 expansion을, 후보 내부 강등이면 reranking·fusion·chunking을 점검한다.
4. Qwen router는 지금 정책으로 도입하지 않는다. 신뢰할 수 있는 label과 독립 평가에서 추가 비용 대비 보완 효과를 확인한다.
5. Fine-tuning은 인간 검수된 조건·역할·수치 hard negative가 충분해진 뒤 검토한다.

공개 benchmark qrels도 불완전하며, 자체 synthetic·Kubernetes label은 인간 gold가 아니다. 현 결과는 재사용 가능한 방법론과 유망 후보를 제공하지만 회사 트래픽에서의 우월성이나 배포 승인을 대신하지 않는다.

최신 근거: [Phase8.4 최종 보고서](PHASE8_4_FINAL_SEMANTIC_EVALUATION.md), [평가 JSON](../runs/phase8_realistic_korean_v1/phase8_4_review_results_v3/evaluation.json), [80질의 원문 비교](../runs/phase8_realistic_korean_v1/phase8_4_review_results_v3/primary_query_comparison.csv).
