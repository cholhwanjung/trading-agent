# trading-agent

에이전트 메모리 중심의 self-improve 트레이딩 에이전트. 검증된 메모리 메커니즘(통계적 admission 게이트, residual/confidence 영향력 제어)을 라이브 페이퍼 트레이딩 결정에 이식한다.

> 페이퍼·모의투자 전용. 투자 자문이 아니다.

[English README](README.md)

## 왜 만드는가

- **"Time travel is cheating."** LLM 가중치에는 과거 시장이 이미 내재되어 있어 백테스트 성과는 승격 근거가 되지 못한다. 성공 판정은 모델 cutoff 이후 라이브(페이퍼) 데이터로만 한다. 백테스트는 신호 스크리닝 전용.
- **Verifier가 에이전트보다 먼저.** 측정 하네스(어댑터·페이퍼 루프·구조화 로깅)가 없으면 self-improve는 자기기만이다.
- **프로세스로 이기고 자원으로 진다.** 무료·일간 해상도 데이터(OHLCV·RSS·공시)만 쓴다. 고빈도·시장미시구조·유료 대체데이터·대규모 팩터 군비경쟁에는 들어가지 않는다. 목표는 기관급 알파가 아니라 B&H 대비 α>0.

## 핵심 설계 원칙

- **메모리는 calibrated residual.** 교훈은 base 결정에 대한 보정항이다. 반복 n≥5, 부호검정 p≤0.05, 라이브 OOS probation을 통과해야 개입한다. 영향력은 confidence로 스케일되고 편차 상한이 걸린다. 인용되지 않은 메모리의 영향력은 0.
- **실패는 하드 veto, 성공은 소프트 프라이어.** 결정론적으로 차단하는 것은 Forbidden(실패) 패턴뿐이며, 그것도 고신뢰일 때만.
- **Risk Engine은 LLM을 참조하지 않는다.** MDD 서킷브레이커 → Forbidden veto → 종목당 상한 → 최소 현금 → turnover 제한.
- **행동 공간은 배분비율 벡터.** 이산 Buy/Sell/Hold가 아니라 현금 포함 자산별 비율(∑=1, long-only). 현금도 포지션이다.
- **시장별 격리.** 전략과 메모리는 시장별 네임스페이스로 분리하고, 시장 간 일반화를 가정하지 않는다.
- **누출 통제.** 관측 상한 `t−1`은 불가침이며 당일 데이터를 차단한다. 창의 *길이*는 실험 변수다 — 봉은 최근 N거래일(기본 3, 월요일에도 N봉이 보이도록), 뉴스는 최근 N캘린더일(기본 7, 사건은 거래 세션을 따르지 않으므로 봉 창과 디커플). 모든 관측에 수집 타임스탬프를 기록한다.
- **자동 프롬프트 변이 금지.** self-improve는 월간 제안서만 생성하며, 승인 없이는 적용되지 않는다.

## 일일 루프

```
관측 (봉 ≤ t−1, 뉴스 ≤ t−1) ─ 누출 가드
  → feature 7종 (RSI · MACD hist · SMA 괴리 · 수익률 · ATR · 거래량비 · drawdown)
  → alpha 신호 top-3 (Alpha Lab 승인 팩터)
  → 예정 거시 이벤트 (FOMC / 금통위, 공개 캘린더)
  → LLM 2-pass 결정 (base → 인용된 메모리만 residual blend)
  → Risk Engine (결정론 가드레일)
  → 주문 (페이퍼)
  → 가상 4-arm 병행 (llm / llm_base / B&H / random)
  → 메모리 기록 → 결과 채점 → admission / probation / retention
  → shadow 채널 (국면 · 변동성 · 수급 · 집중도 · 시장 간 예산)
  → 일일 브리핑
```

모든 결정 로그는 구조화 JSON이며, 인용한 memory·신호 ID를 포함한다(credit assignment의 원천).

## 장중 트리거

무상태 워커가 15분마다 점검하고, 롤링 참조가 대비 급변일 때만 발동한다. 발동 시 동일한 결정 경로를 다시 타되 장중 변동은 **분리된 라벨 채널**로 주입해 당일 정보가 관측 창을 오염시키지 않게 한다.

| 시장 | 임계 | 장 시간 게이팅 |
|---|---|---|
| 크립토 | 8% | 없음 (24/7) |
| KR | 5% | 평일 09:00–15:30 KST |

트리거 결정은 학습 파이프라인에서 제외한다 — 당일 정보를 admission에 넣으면 승격 통계가 오염된다.

## 구조

```
adapters/     시장 어댑터 (Binance testnet · Alpaca · KIS 국내/해외 · Upbit)
              + 관측 누출 가드, 뉴스/RSS, DART 공시, FRED 매크로
harness/      일일 페이퍼 루프 + baseline + 구조화 JSON 로깅 + 런 데드라인
trader/       LLM 결정(2-pass residual) + 결정 스키마 + feature + 이벤트 캘린더
memory/       episodic/semantic/procedural + admission/retention 게이트 + 영향력 제어
risk/         결정론 가드레일 + Forbidden veto + 배분 집중도
regime/       시장 국면 상태기계 + 매크로 + 실현변동성 + 시장 간 배분 제안
watcher/      장중 트리거 판정 (순수 함수)
treasury/     시장 간 자본 이체 가드, 자동/수동 레그
alpha_lab/    팩터 DSL → IC 백테스터 → 4단계 admission (writer/judge LLM 루프)
reflection/   주간 성과·기여 재평가
interaction/  Chat Gateway, 답변은 근거 ID 인용 강제 (FastAPI + MCP)
eval/         4-arm ablation + rolling 유의성 + 시장 간 결합 지수
gui/          읽기 전용 대시보드
llm/          멀티 프로바이더 백본 + 토큰·비용 계측
scripts/      운영: 일일 스텝 · alpha 사이클 · 워처 · 자본 이체 · 브리핑 · 스케줄러
```

## 메모리 파이프라인

```
결정 ──▶ episodic (pattern_key + 결과 채점: 행동수익 − 보유수익)
           │  admission: n≥5 + 부호검정 p≤0.05 + 임베딩 중복체크
           ▼
        probation (7일 라이브 OOS, 승격 근거 샘플 제외)
           │  통과
           ▼
  semantic (성공 → 소프트 프라이어) / procedural (실패 → 하드 veto)
           │  retention: 부호 반전 무효화 + 중복 통합, diversity 우선
           ▼
        retired
```

단일 매매로는 아무것도 승격되지 않는다. 개별 P&L은 noise다.

## 관측 채널

*shadow*로 표시된 채널은 계산·로깅만 하며 결정과 Risk Engine에 개입하지 않는다. 라이브 비교로 유용성이 확인된 뒤에만 승격한다.

| 채널 | 원천 | 상태 |
|---|---|---|
| OHLCV 봉, feature 7종 | 브로커 API | 주입 |
| 뉴스 헤드라인 | Google News RSS (종목별 + 시장 레벨), CoinDesk/Cointelegraph | 주입 |
| 기업 공시 (KR) | DART OpenAPI, 뉴스 채널에 합류 | 주입 |
| Alpha 신호 | Alpha Lab 승인 팩터 | 주입 |
| 예정 거시 이벤트 | 공표된 FOMC / 금통위 일정 | 주입 |
| 장중 급변 | 브로커 시세, 분리된 라벨 채널 | 발동 시 주입 |
| 시장 국면 | 지수 프록시 일봉, 분산일 상태기계 | shadow |
| 매크로 국면 | FRED 일간 시계열 | shadow |
| 실현변동성 | 지수 프록시, 20일 연율화 | shadow |
| 투자자 수급 (KR) | KIS 일별 외인/기관/개인 순매수 | shadow |
| 배분 집중도 | 자체 배분 + 60일 상관 | shadow |
| 시장 간 예산 | 고정 균등 대비 국면 틸트 | shadow |

뉴스는 결정론적으로 필터링한다 — 발행처 블록리스트로 블로그·개인 콘텐츠를 제거하고, 재발행 기사는 정규화된 제목으로 중복 제거한다. 종목명 질의와 함께 시장 레벨 질의를 돌려 지수 급변·서킷브레이커·규제 뉴스가 관측에 들어오게 한다.

집중도 지표를 두는 이유는 종목당 상한이 **명목** 분산만 강제하기 때문이다. 상관 0.9로 동조하는 두 종목은 각각 상한까지 채워도 아무 규칙을 위반하지 않지만 포트폴리오는 사실상 단일 베팅이 된다. 이 지표는 상관 조정 유효 종목 수 `1/(wᵀΡw)`를 보고하며, 그런 경우 값이 1로 붕괴한다.

## Alpha Lab (주간)

LLM writer가 제한된 DSL(AST 화이트리스트, 미래참조 불가)로 팩터 후보를 제안한다. 경량 rank IC/ICIR 백테스터가 스크리닝한 뒤 4단계 admission을 적용한다 — |IC| ≥ 0.02 → 라이브러리 상관 < 0.7 → 배치 중복 제거 → OOS 부호 유지. 승인 팩터는 다음 일일 결정에 top-3 신호로 주입된다. 연구 유니버스와 매매 유니버스는 분리한다.

각 사이클은 라이브 실현 IC 우위가 소멸한 팩터를 퇴출하며 시작한다 — 알파는 발견되면 감쇠하므로, 영속을 가정하지 않고 시스템이 능동 추적한다.

시장: 크립토(연구 유니버스 10종), 미국 주식(12종).

## 시장 간 자본 배분 (shadow)

세 시장은 하나의 예산을 나눠 쓴다. 각 시장의 국면을 무료 일간 지수 봉으로 분류하고, 고정 균등 배분을 기준선으로 국면차만큼 예산을 결정론적으로 틸트한다. 상대 국면차가 없으면 예산을 옮기지 않는다 — 하방 방어는 각 시장 내부의 현금 비중이 담당하며 격리를 유지한다.

실제 자본 이동은 LLM 비개입 가드를 통과한 것만 집행한다 — 하드코딩된 목적지 allowlist(런타임·관측·LLM에서 유입 불가), 건당·일일 상한, 드리프트 게이트, 쿨다운, 잔고 대조. 거래소 KRW 출금은 API로 자동이며 목적지는 본인 명의 등록 계좌로 고정된다. 증권↔은행 레그는 자금이동 API가 없어 사람이 옮기고, 시스템은 사람의 확인을 신뢰하는 대신 잔고 조회로 완료를 검증한다.

집행 없는 미리보기: `uv run python scripts/run_treasury_step.py`

## 평가

같은 관측에 대해 4개 arm을 가상 운용해 매일 ablation을 축적한다.

| arm | 내용 |
|---|---|
| `llm` | 메모리 포함 전체 파이프라인 |
| `llm_base` | 메모리 제거 — memory delta = llm − llm_base |
| `bh` | Buy & Hold |
| `random` | 랜덤 배분 |

유의성은 **겹치지 않는** 20일 청크의 부호검정으로만 판단한다. 롤링 창은 기술 통계로만 보고한다 — 중첩 창은 자기상관으로 표본을 부풀린다.

리포트: `uv run python scripts/report_ablation.py`

## 시작

```bash
uv sync
cp .env.example .env   # 브로커·LLM 키 기입
uv run python scripts/check_credentials.py
uv run python scripts/run_paper_step.py --dry-run
uv run python scripts/run_paper_step.py
```

### 환경 변수

| 변수 | 용도 |
|---|---|
| `BINANCE_TESTNET_API_KEY/SECRET` | 크립토 페이퍼 — 주문은 testnet, 시세는 mainnet 공개 API |
| `ALPACA_PAPER_API_KEY/SECRET` | 미국 주식 페이퍼 |
| `KIS_PAPER_APP_KEY/SECRET/ACCOUNT` | 한국 주식 모의투자 |
| `KIS_REAL_APP_KEY/SECRET/ACCOUNT` | 선택 — KIS 해외주식 실계좌 소액 |
| `LIVE_MAX_ORDER_USD` / `LIVE_MAX_DAILY_USD` | 실계좌 경로의 절대 명목금액 상한 |
| `LLM_SMART` / `LLM_FAST` / `LLM_EMBED` | `provider:model` 형식 (예: `openai:gpt-5.5`) |
| `DART_API_KEY` / `FRED_API_KEY` | KR 공시, US 매크로 시계열 |
| `OBSERVATION_TRADING_DAYS` / `OBSERVATION_NEWS_DAYS` | 관측 창 길이 (기본 3 / 7) |
| `INTERACTION_API_TOKEN` | Chat Gateway Bearer 토큰, 외부 노출 시 필수 |

## 운영

| 명령 | 주기 | 용도 |
|---|---|---|
| `scripts/run_paper_step.py --markets CRYPTO,US` | 매일 23:00 KST | 관측 → 결정 → Risk → 주문 → 메모리 |
| `scripts/run_paper_step.py --markets KR` | 매일 10:00 KST | 동일, KR 장중 |
| `scripts/run_watcher.py --market CRYPTO\|KR` | 15분마다 | 장중 트리거 점검 |
| `scripts/run_alpha_lab.py` | 일요일 22:00 | 팩터 생성 → 백테스트 → admission |
| `scripts/request_capabilities.py` | 매월 1일 20:30 | 측정된 갭 기반 능력 요구 생성 |
| `scripts/propose_improvements.py` | 매월 1일 21:00 | self-improve 제안서, 자동 적용 없음 |
| `scripts/run_treasury_step.py` | 주/월, dry-run | 이체 계획·가드 판정, 집행 없음 |
| `scripts/report_ablation.py` | 수동 | memory delta, B&H 대비 α, 시장 간 델타 |
| `streamlit run gui/dashboard.py` | 수동 | 읽기 전용 대시보드 |
| `uvicorn interaction.api:app --port 8721` | 상시 | Chat Gateway |
| `scripts/run_mcp_server.py` | 온디맨드 | Claude용 MCP 서버 |

launchd(`scripts/launchd/`)와 Docker Compose 스케줄러 중 하나만 사용한다.

## Docker

```bash
docker compose up -d   # gateway(:8721) + scheduler
```

호스트 launchd 잡과 동시 가동 금지. 상태·로그는 `./data` 볼륨에 영속한다.

## 진행 상태

| 영역 | 상태 |
|---|---|
| 측정 하네스 (어댑터·페이퍼 루프·로깅) | 완료 — 크립토·US·KR 전부 가동 |
| LLM Trader + Risk Engine | 완료 |
| 메모리 스택 | 완료, 데이터 축적 중 |
| Alpha Lab + 주간 reflection | 완료, 사이클 가동 중 |
| 장중 트리거 | 배치 완료 — 크립토·KR |
| 시장 간 자본 배분 | shadow·dry-run, 검증 후 집행 |
| Interaction Gateway | 완료 |
| 실계좌 소액 전환 | 페이퍼 검증 이후 |

## 결과

2026년 7월 페이퍼 트레이딩 시작. 현재 표본은 시장당 정렬 거래일 14~21일로, 첫 부호검정 p-value에 필요한 ~100일에 크게 못 미친다. 따라서 **어떤 arm도 베이스라인 대비 통계적으로 우월하거나 열등하다고 말할 수 없다.**

방향성만 보면 일관된 우위는 낙폭 하나다 — 전 시장에서 에이전트의 MDD가 B&H보다 낮고, 초과수익은 대체로 무승부다. 이는 알파가 아니라 리스크 엔진이 작동한다는 뜻으로 읽어야 한다.

백테스트 수치는 의도적으로 게재하지 않는다.

## 계보

검증된 메커니즘을 개인 규모 운용에 이식한 것 — FactorMiner(팩터 admission 게이트), AlphaMemo(residual 메모리, 비대칭 평가), Alpha Arena(feature 정예화 근거), LiveTradeBench(라이브 평가 프로토콜), FinMem(계층적 에이전트 메모리).

## 면책

실계좌 전환은 충분한 페이퍼 검증과 명시적 승격 조건 충족 후 소액부터 시작한다. 본 소프트웨어는 투자 자문이 아니며, 사용에 따른 손실은 사용자 책임이다.
