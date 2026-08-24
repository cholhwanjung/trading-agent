"""inner loop — writer(smart) 팩터 생성 → judge(fast) 리뷰 → 프로그래매틱 검증.

QuantAgent inner/outer 결합: LLM 은 가설과 수식만 내고, 진짜 판정은 outer
(백테스트 admission, library.admit)가 한다. judge 는 경제적 타당성·단순성의
저비용 1차 필터일 뿐 — judge 통과가 승격 근거가 아니다 (비대칭).
"""

from __future__ import annotations

import json

from adapters.retry import with_retry
from alpha_lab.dsl import DSL_SPEC, DSLError, validate
from alpha_lab.library import FactorCandidate, FactorLibrary
from llm import LLMError, LLMRouter, extract_json

# writer 재시도 — 주간 잡이라 한 번의 일시 실패가 곧 한 주의 발견 사이클 손실이다.
# 백엔드에도 재시도가 있지만 그쪽은 전송 오류(httpx)만 잡는다. 여기서 잡는 것은
# **정상 200 으로 오는 쓸모없는 응답**(빈 본문·JSON 아님·팩터 0개) — 예외가 아니라서
# 전송 계층 재시도를 그대로 통과해 사이클을 끝내 버린다.
WRITER_ATTEMPTS = 3
WRITER_RETRY_DELAY = 2.0

WRITER_PROMPT = """\
너는 퀀트 팩터 연구자다. 아래 DSL 로 {asset} 일간 횡단면 팩터 후보 {n}개를 제안하라.

{dsl_spec}

목표: 익일 수익률 예측 rank IC 0.03~0.05 수준의 저상관 보조 신호 (기관급 알파가 아니다).
각 팩터는 서로 다른 정보 채널(모멘텀/반전/변동성/거래량/상관)을 노려 다양성을 확보하라.

기존 라이브러리 (중복 금지):
{existing}

성공 경험 (참고):
{successful}

실패 경험 (같은 실수 반복 금지):
{forbidden}

JSON 만 출력:
{{"factors": [{{"name": "<snake_case>", "expression": "<DSL 수식>", "hypothesis": "<경제적 근거 1문장>"}}, ...]}}"""

JUDGE_PROMPT = """\
너는 퀀트 리뷰어다. 각 팩터 후보를 검토하라 — 기준: ① 경제적 가설이 수식과 일치하는가
② lookahead/자기참조 위험 ③ 과도한 복잡성(과적합 신호).

후보:
{candidates}

JSON 만 출력 (기각은 이유 필수):
{{"reviews": [{{"name": "<name>", "verdict": "ok|reject", "reason": "<기각 사유 또는 빈 문자열>"}}, ...]}}"""


async def _writer_factors(router: LLMRouter, prompt: str) -> list:
    """writer 1회 호출 → 팩터 목록. 쓸 수 없는 응답이면 ValueError(재시도 신호).

    빈 본문·JSON 아님·팩터 0개를 모두 실패로 본다. 셋 다 HTTP 200 으로 오므로
    호출부가 재시도하지 않으면 그대로 사이클이 끝난다.
    """
    resp = await router.complete(
        "smart", purpose="alpha_writer",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096, json_mode=True,
    )
    data = extract_json(resp.text)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 없음: {resp.text[:100]!r}")
    factors = data.get("factors") or []
    if not factors:
        raise ValueError(f"팩터 0개: {resp.text[:100]!r}")
    return factors


async def generate_candidates(
    router: LLMRouter, library: FactorLibrary, n: int = 5, asset_label: str = "크립토"
) -> list[FactorCandidate]:
    """writer → judge → DSL 검증. 반환 후보의 rejected 필드에 1차 필터 결과 반영."""
    existing = [f.expression for f in library.active()] or ["(없음)"]
    successful = library.experience["successful"][-5:] or ["(없음)"]
    forbidden = library.experience["forbidden"][-10:] or ["(없음)"]
    prompt = WRITER_PROMPT.format(
        n=n,
        asset=asset_label,
        dsl_spec=DSL_SPEC,
        existing=json.dumps(existing, ensure_ascii=False),
        successful=json.dumps(successful, ensure_ascii=False),
        forbidden=json.dumps(forbidden, ensure_ascii=False),
    )

    raw = await with_retry(
        lambda: _writer_factors(router, prompt),
        attempts=WRITER_ATTEMPTS,
        base_delay=WRITER_RETRY_DELAY,
        exceptions=(ValueError, LLMError),
    )
    candidates = [
        FactorCandidate(
            name=str(f.get("name", f"factor_{i}"))[:60],
            expression=str(f.get("expression", "")),
            hypothesis=str(f.get("hypothesis", ""))[:300],
        )
        for i, f in enumerate(raw)
    ]

    # 프로그래매틱 DSL 검증 (judge 이전 — 문법 불량은 LLM 리뷰 낭비)
    for c in candidates:
        try:
            validate(c.expression)
        except DSLError as e:
            c.rejected = f"dsl: {e}"

    # judge 리뷰 (경량 tier). 재시도하지 않는다 — 비대칭 필터라 통과가 승격 근거가
    # 아니고, 건너뛰면 후보가 outer 게이트로 그냥 넘어갈 뿐 사이클은 성립한다.
    # 예외를 넓게 잡는 것은 그 계약을 그대로 옮긴 것이다: judge 는 어떤 이유로 실패해도
    # 사이클을 끝내지 못한다(호출 실패·파싱 실패 구분 없이). 실행 여부는 LLM 사용량
    # 로그의 purpose=alpha_judge 유무로 사후 확인된다.
    reviewable = [c for c in candidates if not c.rejected]
    if reviewable:
        try:
            judge_resp = await router.complete(
                "fast",
                purpose="alpha_judge",
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            candidates=json.dumps(
                                [
                                    {"name": c.name, "expression": c.expression,
                                     "hypothesis": c.hypothesis}
                                    for c in reviewable
                                ],
                                ensure_ascii=False,
                            )
                        ),
                    }
                ],
                max_tokens=2048,
                json_mode=True,
            )
            judge_data = extract_json(judge_resp.text)
            reviews = (
                {r["name"]: r for r in judge_data.get("reviews", [])}
                if isinstance(judge_data, dict)
                else {}
            )
            for c in reviewable:
                review = reviews.get(c.name)
                if review and review.get("verdict") == "reject":
                    c.rejected = f"judge: {review.get('reason', '')[:150]}"
        except Exception:
            pass  # judge 실패는 비치명 — outer 게이트가 최종 판정
    return candidates
