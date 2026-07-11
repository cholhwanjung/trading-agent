"""inner loop — writer(smart) 팩터 생성 → judge(fast) 리뷰 → 프로그래매틱 검증 (R11).

QuantAgent inner/outer 결합: LLM 은 가설과 수식만 내고, 진짜 판정은 outer
(백테스트 admission, library.admit)가 한다. judge 는 경제적 타당성·단순성의
저비용 1차 필터일 뿐 — judge 통과가 승격 근거가 아니다 ([ADR-001] 비대칭).
"""

from __future__ import annotations

import json
import re

from alpha_lab.dsl import DSL_SPEC, DSLError, validate
from alpha_lab.library import FactorCandidate, FactorLibrary
from llm import LLMRouter

WRITER_PROMPT = """\
너는 퀀트 팩터 연구자다. 아래 DSL 로 크립토 일간 횡단면 팩터 후보 {n}개를 제안하라.

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


def _extract_json(text: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError(f"JSON 없음: {text[:100]!r}")
    return json.loads(match.group())


async def generate_candidates(
    router: LLMRouter, library: FactorLibrary, n: int = 5
) -> list[FactorCandidate]:
    """writer → judge → DSL 검증. 반환 후보의 rejected 필드에 1차 필터 결과 반영."""
    existing = [f.expression for f in library.active()] or ["(없음)"]
    successful = library.experience["successful"][-5:] or ["(없음)"]
    forbidden = library.experience["forbidden"][-10:] or ["(없음)"]

    resp = await router.complete(
        "smart",
        messages=[
            {
                "role": "user",
                "content": WRITER_PROMPT.format(
                    n=n,
                    dsl_spec=DSL_SPEC,
                    existing=json.dumps(existing, ensure_ascii=False),
                    successful=json.dumps(successful, ensure_ascii=False),
                    forbidden=json.dumps(forbidden, ensure_ascii=False),
                ),
            }
        ],
        max_tokens=4096,
        json_mode=True,
    )
    raw = _extract_json(resp.text).get("factors", [])
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

    # judge 리뷰 (경량 tier)
    reviewable = [c for c in candidates if not c.rejected]
    if reviewable:
        judge_resp = await router.complete(
            "fast",
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
        try:
            reviews = {
                r["name"]: r for r in _extract_json(judge_resp.text).get("reviews", [])
            }
            for c in reviewable:
                review = reviews.get(c.name)
                if review and review.get("verdict") == "reject":
                    c.rejected = f"judge: {review.get('reason', '')[:150]}"
        except (ValueError, KeyError, json.JSONDecodeError):
            pass  # judge 실패는 비치명 — outer 게이트가 최종 판정
    return candidates
