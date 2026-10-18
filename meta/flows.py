"""입출금 분배 — 전략 anchor 를 향해 흐름으로 먼저 맞춘다(매도 없는 리밸런싱).

KIS 통합증거금 계좌 하나를 KR·US 가 나눠 쓰므로 외부 입출금은 장부 지분으로 갈린다.
지분을 상수(0.5)로 두면 입금이 현재 비율을 그대로 복제한다. 대신 anchor 대비 밴드 밖으로
벗어난 쪽이 있으면 흐름을 **부족한 시장에 전부** 보내고, 밴드 안이면 anchor 비율로 나눈다.
입금액을 미리 알 수 없어 정확히 채울 수는 없다 — 넘치면 다음 흐름이 되돌린다.
이 규칙은 LLM 제안이 아니라 anchor 만 본다. 메타 제안의 집행은 승격 뒤의 일이다.
"""

from __future__ import annotations

import json
from pathlib import Path

ANCHOR_KR_SHARE = 0.5  # KR : US 균등 — 전략 anchor(1/3 : 1/3)의 KIS 안 비율
FLOW_BAND = 0.05


def _equity(state_dir: Path, market: str) -> float | None:
    path = state_dir / f"risk_{market}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("prev_equity")
    except json.JSONDecodeError:
        return None
    return float(value) if value and value > 0 else None


def kis_flow_share(
    state_dir: Path | str,
    anchor_kr: float = ANCHOR_KR_SHARE,
    band: float = FLOW_BAND,
) -> float:
    """다음 외부 입출금에서 KR 이 받을 몫. 평가액을 모르면 anchor 비율."""
    state_dir = Path(state_dir)
    kr, us = _equity(state_dir, "KR"), _equity(state_dir, "US")
    if kr is None or us is None:
        return anchor_kr
    current = kr / (kr + us)
    if current < anchor_kr - band:
        return 1.0  # KR 부족 → 흐름 전부 KR
    if current > anchor_kr + band:
        return 0.0  # KR 초과 → 흐름 전부 US
    return anchor_kr
