# trading-agent

자기 결정 이력에서 배우는 LLM 트레이딩 에이전트입니다. 검증된 메모리 메커니즘(통계적 admission 게이트, residual·confidence 영향력 제어)을 라이브 매매 결정에 이식하고, 그 아래에 결정론적 리스크 엔진을 둡니다.

> 주문은 실계좌로 나가지만 규모는 의도적으로 작게 유지합니다. 크립토는 키를 넣기 전까지 거래소 testnet에 머뭅니다. 모든 결정 옆에서 가상 포트폴리오 4종이 함께 도는데, 측정의 근거는 계좌가 아니라 그쪽입니다. 투자 자문이 아닙니다.

[English README](README.md)

## 전제 규칙

- **"Time travel is cheating."** LLM 가중치에는 과거 시장이 이미 들어 있어서 백테스트 성과로는 아무것도 승격시킬 수 없습니다. 승격은 모델 cutoff 이후 라이브 데이터로만 판정하고, 백테스트는 신호 스크리닝에만 씁니다.
- **Verifier가 에이전트보다 먼저.** 어댑터·일일 루프·구조화 로깅이 먼저 돌았습니다. 측정이 없으면 self-improve는 자기기만입니다.
- **프로세스로 이기고 자원으로 집니다.** 무료·일간 해상도 데이터만 씁니다 — OHLCV·RSS·공시·FRED. 고빈도, 유료 대체데이터, 대규모 팩터 군비경쟁에는 들어가지 않습니다. 목표는 기관급 알파가 아니라 B&H 대비 α > 0입니다.
- **메모리는 calibrated residual입니다.** 교훈은 base 결정에 대한 보정항입니다. 반복 n ≥ 5, 부호검정 p ≤ 0.05, 라이브 OOS probation을 전부 통과해야 개입합니다. 영향력은 confidence로 스케일되고 상한이 걸리며, 인용되지 않은 메모리의 영향력은 0입니다.
- **리스크 엔진은 LLM을 참조하지 않습니다.** MDD 서킷 → 실패 패턴 veto → 배분 정의역 → 종목당 상한 → 최소 현금 → turnover 제한 순으로 적용합니다.
- **행동 공간은 배분비율 벡터입니다.** 이산 Buy/Sell/Hold가 아니라 현금을 포함해 합이 1이 되는 비율(long-only)입니다. 현금도 포지션입니다.

## 일일 루프

```
관측 (봉 · 뉴스 · 공시 ≤ t−1) ─ 누출 가드
  → 가격 feature 7종 · 재무 3종 · 시장 밸류에이션 · ETF 괴리율
  → alpha 신호 top-3 · 예정 거시 이벤트 · 계좌 예산
  → LLM 결정 (교훈이 조회된 경우에만 2-pass, residual blend)
  → 리스크 엔진 → 거래가능 게이트 → 실주문 상한
  → 주문
  → 가상 4-arm, t−1 종가 마킹 (llm / llm_base / B&H / random)
  → 메모리 기록 → 결과 채점 → admission / probation / retention
  → shadow 채널 → 일일 브리핑
```

## 유니버스와 집행

| 시장 | 관측 | 배분 | 체결 |
|---|---|---|---|
| KR | 대형 4 + KODEX 200TR | `278530` | KIS 실계좌 (실전 키 없으면 모의) |
| US | 메가캡 5 + SCHX | `SCHX` | KIS 해외 실계좌 (없으면 Alpaca 페이퍼) |
| 크립토 | BTC · ETH · SOL | 동일 | Upbit 실계좌 (없으면 Binance testnet) |

배분 집합은 관측 집합보다 좁습니다. 소액 계좌에서는 개별 종목 1주 값이 이미 종목당 상한을 넘어서 어떤 비중도 표현되지 않고, 그런 종목에 낸 배분은 매번 통째로 현금으로 되돌아갑니다. 그래서 배분 벡터는 광범위 지수 ETF 위에 정의하고, 개별 종목은 판단 근거로만 관측에 남깁니다. 계좌가 커지면 개별 종목이 배분 집합으로 올라옵니다.

### 실주문 가드

- **1회 매수 상한** — `min(절대 천장, 평가액 × 비율)`입니다. 평가액은 매 스텝 브로커에서 직접 읽으므로 입출금·손익이 자동으로 반영되고 재설정할 일이 없습니다.
- **일일 누적 매수 상한** — 날짜가 바뀌면 리셋됩니다.
- **매도에는 상한이 없습니다.** long-only에서 매도 수량의 상한은 보유량 자체이고 결과는 현금입니다. 상한이 매도를 막으면 방어가 급한 바로 그 순간에 방어가 느려집니다.
- **거래가능 게이트** — 거래정지·거래량 0·상하한가 종목은 제출 전에 걸러냅니다. 정리매매 종목은 매수만 막고 매도는 열어 둡니다.

## 장중 트리거

무상태 워커가 15분마다 점검하고, 롤링 참조가 대비 급변일 때만 발동합니다. 발동하면 같은 결정 경로를 다시 타되 장중 변동은 **분리된 라벨 채널**로 주입해서 당일 정보가 관측 창을 오염시키지 않게 합니다.

| 시장 | 임계 | 장 시간 게이팅 |
|---|---|---|
| 크립토 | 8% | 없음 (24/7) |
| KR | 5% | 평일 09:00–15:30 KST |

US 워처는 아직 없습니다. 트리거 결정은 학습 파이프라인에서 제외합니다 — 당일 정보를 admission에 넣으면 승격 통계가 오염되기 때문입니다.

## 메모리

```
결정 ──▶ episodic (pattern key + 결과: 행동수익 − 보유수익)
           │  admission: n≥5 + 부호검정 p≤0.05 + 임베딩 중복 기각
           ▼
        probation (7일 라이브 OOS, 승격 근거 샘플 제외)
           │  통과
           ▼
  semantic (성공 → 소프트 프라이어) / procedural (실패 → 하드 veto)
           │  retention: 부호 반전 무효화 + 중복 통합, diversity 우선
           ▼
        retired
```

단일 매매로는 아무것도 승격되지 않습니다. 개별 P&L은 noise입니다. veto된 결정은 반사실로 남겨 둡니다 — 그래야 차단된 패턴도 재검증에 필요한 표본을 계속 쌓습니다.

## 관측 채널

*shadow* 채널은 계산·로깅만 하고 결정과 리스크 엔진에는 개입하지 않습니다. 라이브 비교로 유용성이 확인된 뒤에만 승격합니다.

| 채널 | 원천 | 상태 |
|---|---|---|
| OHLCV 봉, feature 7종 | 브로커 API | 주입 |
| 뉴스 | Google News RSS(KR), Alpaca(US), CoinDesk / Cointelegraph | 주입 |
| 공시 | DART(KR), SEC EDGAR 8-K/10-Q(US) | 주입 |
| 재무 (PER · ROE · 부채비율, TTM) | DART 재무제표, EDGAR XBRL | 주입 |
| 시장 밸류에이션 | 대리 펀드 공개값 기반 시장 단위 배수 | 주입 |
| ETF 괴리율 | KIS — KR 전용, US는 무료 원천 없음 | 주입 |
| Alpha 신호 | Alpha Lab 승인 팩터 | 주입 |
| 예정 거시 이벤트 | 공표된 FOMC / 금통위 일정 | 주입 |
| 계좌 예산 | 브로커 현금 대비 1주 값·목표 금액 | 주입 |
| 장중 급변 | 브로커 시세, 분리된 라벨 채널 | 발동 시 주입 |
| 시장 국면 | 지수 프록시 일봉, 분산일 상태기계 | shadow |
| 시장 국면 (jump model) | 같은 프록시, 통계적 상태 분류 | shadow |
| 매크로 국면 | FRED 일간 시계열 | shadow |
| 실현변동성 | 지수 프록시, 20일 연율화 | shadow |
| 투자자 수급 (KR) | KIS 외인 / 기관 / 개인 순매수 | shadow |
| 배분 집중도 | 자체 배분 + 60일 상관 | shadow |
| 시장 간 예산 | 고정 균등 대비 국면 틸트 | shadow |
| 시세 상태 · 집행 갭 · 매수여력 | 브로커 응답 | shadow |

뉴스는 결정론적으로 거릅니다. 발행처 블록리스트로 블로그·개인 콘텐츠를 빼고, 재발행 기사는 정규화된 제목으로 중복 제거합니다. 종목명 질의와 함께 시장 레벨 질의를 돌려서 지수 급변·서킷브레이커·규제 뉴스가 아예 안 보이는 일을 막습니다. 채널은 이어붙이지 않고 라운드로빈으로 병합하는데, 뒤에 붙인 것이 프롬프트 절삭에 가장 먼저 잘리기 때문입니다.

재무는 가격 feature 벡터와 **별도 채널**이지 feature를 늘린 것이 아닙니다. 분기마다 바뀌는 값이 pattern key에 들어가면 admission이 기대는 반복 관측이 매 분기 갈라집니다.

집중도 지표를 두는 이유는 종목당 상한이 **명목** 분산만 강제하기 때문입니다. 상관 0.9로 동조하는 두 종목은 각각 상한을 채워도 아무 규칙을 어기지 않지만 포트폴리오는 사실상 단일 베팅입니다. 이 지표는 상관 조정 유효 종목 수 `1/(wᵀΡw)`를 보고하고, 그런 경우 값이 1로 붕괴합니다.

## Alpha Lab (주간)

LLM writer가 제한된 DSL(AST 화이트리스트, 미래 참조 불가)로 팩터를 제안합니다. 경량 rank IC/ICIR 백테스터로 스크리닝한 뒤 4단계 admission을 적용합니다 — |IC| ≥ 0.02 → 라이브러리 상관 < 0.70 → 배치 중복 제거 → OOS 부호 유지. 승인된 팩터는 다음 일일 결정에 top-3 신호로 들어갑니다. 연구 유니버스와 매매 유니버스는 분리합니다 — 크립토 10종, US 12종입니다.

각 사이클은 라이브 실현 IC 우위가 사라진 팩터를 퇴출하며 시작합니다. 알파는 발견되면 감쇠하므로 영속을 가정하지 않고 능동적으로 추적합니다. 현재 active는 크립토 1·US 1이고, 퇴출은 2건입니다.

## 시장 간 자본 배분 (shadow)

각 시장의 국면을 무료 일간 지수 봉으로 분류하고, 균등 배분을 기준선으로 국면차만큼 예산을 결정론적으로 틸트합니다. 상대 국면차가 없으면 옮기지 않습니다 — 하방 방어는 각 시장 내부의 현금 비중이 맡고, 그래야 시장 격리가 유지됩니다.

실제 자본 이동은 LLM 비개입 가드를 통과한 것만 집행합니다. 하드코딩된 목적지 allowlist(런타임·관측·LLM에서 유입 불가), 건당·일일 상한, 드리프트 게이트, 쿨다운, 잔고 대조입니다. 거래소 KRW 출금은 본인 명의 등록 계좌로 자동이지만, 증권↔은행 레그는 자금이동 API가 없어서 사람이 옮기고 시스템은 사람의 확인을 신뢰하는 대신 잔고 조회로 완료를 검증합니다.

집행 없이 미리 보려면 이렇게 실행하세요: `uv run python scripts/run_treasury_step.py`

## 평가

같은 관측에 대해 4개 arm을 가상 운용하고 t−1 종가로 마킹합니다(거래비용 포함).

| arm | 내용 |
|---|---|
| `llm` | 실제로 제출된 배분 — 리스크 엔진 적용 후 |
| `llm_base` | 원본 LLM 배분 — 메모리도 리스크 엔진도 없음 |
| `bh` | Buy & Hold |
| `random` | 랜덤 배분 |

유의성은 **겹치지 않는** 20일 청크의 부호검정으로만 판단합니다. 롤링 창은 기술 통계로만 보고합니다 — 중첩 창은 자기상관으로 표본을 부풀리기 때문입니다.

리포트: `uv run python scripts/report_ablation.py`

## 결과 (2026-08-19)

2026년 7월 시작, 지수 100,000 기준 가상 arm입니다.

| 시장 | 스텝 | 에이전트 | B&H | 에이전트 MDD | B&H MDD |
|---|---|---|---|---|---|
| 크립토 | 32 | +1.8% | +4.9% | 3.8% | 5.2% |
| US | 26 | +3.4% | +4.8% | 3.6% | 5.3% |
| KR | 24 | +5.2% | +4.6% | 7.2% | 21.4% |

첫 부호검정 p-value에는 정렬 거래일 100일 안팎이 필요합니다. 따라서 **어떤 arm도 베이스라인 대비 통계적으로 우월하거나 열등하다고 말할 수 없습니다.** 방향성만 보면 일관된 우위는 낙폭 하나입니다 — 세 시장 모두 MDD가 B&H보다 낮고 초과수익은 엇갈립니다. 알파가 아니라 리스크 엔진이 작동한다는 뜻으로 읽어야 합니다.

메모리 승격은 아직 0건입니다(episodic 102건, 승격 0). 얇은 표본에서 admission 게이트가 설계대로 동작하고 있는 상태입니다. 첫 승격 전까지 `llm − llm_base` 격차는 메모리가 아니라 리스크 엔진과 turnover 제한을 재고 있습니다.

백테스트 수치는 의도적으로 게재하지 않습니다.

## 구조

```
adapters/     시장 어댑터 (KIS 국내/해외 · Alpaca · Upbit · Binance testnet)
              + 누출 가드, 뉴스/RSS, DART·EDGAR 공시, FRED, 시장별 현금 장부
harness/      일일 루프 + baseline + 구조화 JSON 로깅 + 런 데드라인 + 사용량 계측
trader/       LLM 결정(2-pass residual) · 결정 스키마 · feature · 재무 · 이벤트
memory/       episodic/semantic/procedural + admission/retention 게이트 + 영향력 제어
risk/         결정론 가드레일 · 실패 패턴 veto · 실주문 상한 · 집중도 지표
regime/       상태기계 + jump model + 매크로 + 실현변동성 + 시장 간 배분 제안
watcher/      장중 트리거 판정 (순수 함수)
treasury/     시장 간 자본 이체 가드, 자동/수동 레그
alpha_lab/    팩터 DSL → IC 백테스터 → 4단계 admission (writer/judge LLM 루프)
reflection/   주간 성과·기여 재평가
interaction/  Chat Gateway, 답변은 근거 ID 인용 강제 (FastAPI + MCP)
eval/         4-arm ablation + rolling 유의성 + 시장 간 결합 지수
gui/          읽기 전용 대시보드
llm/          멀티 프로바이더 백본 + 토큰·비용 계측
scripts/      운영: 일일 스텝 · alpha 사이클 · 워처 · 자본 이체 · 리포트 · 스케줄러
```

## 시작

```bash
uv sync
cp .env.example .env   # 브로커·LLM 키 기입
uv run python scripts/check_credentials.py
uv run python scripts/run_paper_step.py --dry-run
uv run python scripts/run_paper_step.py --markets CRYPTO
```

어댑터는 존재하는 키에 따라 고릅니다. 실전 키가 하나도 없으면 전 시장이 모의·testnet 계좌로만 돕니다.

| 변수 | 용도 |
|---|---|
| `KIS_REAL_APP_KEY/SECRET/ACCOUNT` | KIS 통합증거금 계좌 하나로 KR + US 실주문 |
| `KIS_PAPER_APP_KEY/SECRET/ACCOUNT` | KR 모의투자 (실전 키가 없을 때 사용) |
| `UPBIT_API_KEY/SECRET` | 크립토 실계좌(KRW 마켓) — 자산조회·주문만 허용하고 출금 권한은 끕니다 |
| `BINANCE_TESTNET_API_KEY/SECRET` | 크립토 testnet 체결. 시세·연구 데이터는 공개 API |
| `ALPACA_PAPER_API_KEY/SECRET` | US 뉴스 · Alpha Lab US 패널 봉 · 실전 키 없을 때의 US 페이퍼 체결 |
| `LIVE_MAX_ORDER_*` / `LIVE_MAX_DAILY_*` | 체결 통화(USD / KRW)별 매수 절대 천장 |
| `LIVE_MAX_ORDER_PCT_{KR,US,CRYPTO}` | 평가액 대비 1회 매수 상한. 천장과 비교해 작은 쪽이 적용됩니다 |
| `LLM_SMART` / `LLM_FAST` | `provider:model` + 해당 프로바이더 API 키 |
| `DART_API_KEY` / `FRED_API_KEY` / `SEC_USER_AGENT` | KR 공시, US 매크로, EDGAR 연락처 식별자 |
| `OBSERVATION_TRADING_DAYS` / `OBSERVATION_NEWS_DAYS` | 관측 창 길이 (기본 3 / 7) |
| `ALERT_WEBHOOK_URL` | 주문 거부·서킷 발동 통지 (Slack 또는 Discord 웹훅) |
| `INTERACTION_API_TOKEN` | Chat Gateway Bearer 토큰, 외부 노출 시 필수 |

## 운영

주문은 각 시장의 장중에만 체결되므로 잡을 시장별로 나눕니다.

| 명령 | 주기 | 용도 |
|---|---|---|
| `scripts/run_paper_step.py --markets KR` | 매일 10:00 KST | 관측 → 결정 → 리스크 → 주문 → 메모리 |
| `scripts/run_paper_step.py --markets US` | 매일 00:30 KST | 동일. 서머타임 적용·해제기 모두 미국 정규장 안 |
| `scripts/run_paper_step.py --markets CRYPTO` | 매일 23:00 KST | 동일. 24/7 거래소 |
| `scripts/run_watcher.py --market CRYPTO\|KR` | 15분마다 | 장중 트리거 점검 |
| `scripts/run_alpha_lab.py` | 일요일 22:00 | 팩터 생성 → 백테스트 → admission |
| `scripts/request_capabilities.py` | 매월 1일 20:30 | 측정된 갭에서 에이전트가 쓰는 능력 요구 |
| `scripts/propose_improvements.py` | 매월 1일 21:00 | self-improve 제안서, 자동 적용 없음 |
| `scripts/run_treasury_step.py` | 수동, dry-run | 이체 계획·가드 판정, 집행 없음 |
| `scripts/report_ablation.py` | 수동 | memory delta, B&H 대비 α, 국면 신호 채점 |
| `streamlit run gui/dashboard.py` | 수동 | 읽기 전용 대시보드 |
| `uvicorn interaction.api:app --port 8721` | 상시 | Chat Gateway |
| `scripts/run_mcp_server.py` | 온디맨드 | Claude용 MCP 서버 |

스케줄은 launchd(`scripts/launchd/`)와 Docker Compose 스케줄러 중 하나만 쓰세요.

```bash
docker compose up -d   # gateway(:8721) + scheduler
```

상태·로그는 `./data` 볼륨에 저장됩니다.

## 계보

검증된 메커니즘을 개인 규모 운용에 이식한 것입니다.

| 출처 | 이식한 것 | 논문 |
|---|---|---|
| FactorMiner | 팩터 admission 게이트 | [arXiv:2602.14670](https://arxiv.org/abs/2602.14670) |
| AlphaMemo | residual 메모리, 비대칭 veto | [arXiv:2606.20625](https://arxiv.org/abs/2606.20625) |
| LiveTradeBench | 라이브 평가 프로토콜 | [arXiv:2511.03628](https://arxiv.org/abs/2511.03628) |
| FinMem | 계층적 에이전트 메모리 | [arXiv:2311.13743](https://arxiv.org/abs/2311.13743) |
| Alpha Arena | feature 정예화 근거 | [nof1.ai](https://nof1.ai/) |

## 면책

실계좌 매매는 소액으로 제한하며, 규모 확대는 라이브 데이터로 측정한 명시적 승격 조건을 충족할 때만 합니다. 본 소프트웨어는 투자 자문이 아니며, 사용에 따른 손실은 사용자 책임입니다.
