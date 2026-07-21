"""팩터 라이브러리 + 4단계 admission (R11 · FactorMiner 이식).

admission: ① IC 스크리닝 → ② 기존 라이브러리 상관 체크 → ③ 배치 중복 제거
→ ④ OOS 견고성. 통과분만 active. 경험 메모리(Successful/Forbidden)를 함께 영속.

저장: JSON 단일 파일. 팩터 스코어는 저장하지 않는다 — 수식이 결정론이므로
상관 체크 시 재계산(evaluate)한다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from alpha_lab.backtest import (
    BacktestResult,
    daily_rank_ic,
    forward_returns,
    run_backtest,
    score_correlation,
)
from alpha_lab.dsl import DSLError, evaluate

# admission 임계 — ADR-012: 목표는 IC 0.03~0.05 보조 신호
MIN_TRAIN_IC = 0.02
MIN_TRAIN_DAYS = 100
MIN_OOS_IC = 0.01
MAX_LIBRARY_CORR = 0.70
MAX_BATCH_CORR = 0.85

# 라이브 감쇠 퇴출 ([ADR-022]) — 알파는 crowding 으로 감쇠한다. admission 이후 실현
# IC 가 우위 방향을 잃으면(방향성 IC ≤ DECAY_FLOOR) retire. 메모리 retention 과 대칭.
MIN_LIVE_DAYS = 20  # post-admission 최소 표본일 — 미만이면 판단 보류(유지, diversity 보존)
DECAY_FLOOR = 0.0  # 방향성 라이브 IC ≤ 이 값이면 우위 소멸 → retire


@dataclass
class FactorCandidate:
    name: str
    expression: str
    hypothesis: str
    result: BacktestResult | None = None
    rejected: str | None = None  # 기각 사유 (Forbidden 경험으로 축적)


@dataclass
class FactorRecord:
    name: str
    expression: str
    hypothesis: str
    status: str  # active | retired
    train_ic: float
    train_icir: float
    oos_ic: float
    oos_icir: float
    admitted_day: str
    sign: int = field(default=1)  # IC 부호 — 신호 방향
    # 라이브 감쇠 추적 ([ADR-022]) — 주간 review_decay 가 갱신. 기존 JSON 은 기본값으로 로드.
    live_ic: float | None = field(default=None)  # 최근 post-admission 실현 rank-IC
    live_ic_n: int = field(default=0)  # 라이브 IC 표본일 수
    live_ic_day: str | None = field(default=None)  # 마지막 갱신일


class FactorLibrary:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            self.factors = [FactorRecord(**f) for f in state["factors"]]
            self.experience = state["experience"]
        else:
            self.factors = []
            self.experience = {"successful": [], "forbidden": []}

    def active(self) -> list[FactorRecord]:
        return [f for f in self.factors if f.status == "active"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"factors": [asdict(f) for f in self.factors], "experience": self.experience},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    def admit(
        self,
        candidates: list[FactorCandidate],
        panel: dict[str, np.ndarray],
        asof_day: date,
    ) -> list[dict]:
        """4단계 admission. 이벤트 리스트 반환(로그용). 통과분은 라이브러리에 추가."""
        events: list[dict] = []

        # 백테스트 (스크리닝 전용 — ADR-002)
        for c in candidates:
            if c.rejected:
                continue
            try:
                c.result = run_backtest(c.expression, panel)
            except DSLError as e:
                c.rejected = f"dsl: {e}"

        # ① IC 스크리닝
        for c in candidates:
            if c.rejected or not c.result:
                continue
            t = c.result.train
            if abs(t.mean_ic) < MIN_TRAIN_IC or t.n_days < MIN_TRAIN_DAYS:
                c.rejected = f"screening: ic={t.mean_ic:.4f} n={t.n_days}"

        # ② 기존 라이브러리 상관 (수식 재계산)
        active_scores = {}
        for f in self.active():
            try:
                active_scores[f.name] = run_backtest(f.expression, panel).scores
            except DSLError:
                continue
        for c in candidates:
            if c.rejected or not c.result:
                continue
            for name, scores in active_scores.items():
                corr = score_correlation(c.result.scores, scores)
                if abs(corr) >= MAX_LIBRARY_CORR:
                    c.rejected = f"library_corr: {name} corr={corr:.2f}"
                    break

        # ③ 배치 중복 제거 — |ICIR| 높은 쪽 생존
        survivors = [c for c in candidates if not c.rejected and c.result]
        survivors.sort(key=lambda c: abs(c.result.train.icir), reverse=True)
        kept: list[FactorCandidate] = []
        for c in survivors:
            dup = next(
                (
                    k
                    for k in kept
                    if abs(score_correlation(c.result.scores, k.result.scores)) >= MAX_BATCH_CORR
                ),
                None,
            )
            if dup:
                c.rejected = f"batch_dup: {dup.name}"
            else:
                kept.append(c)

        # ④ OOS 견고성 — 부호 유지 + 최소 크기
        for c in kept:
            train, oos = c.result.train, c.result.oos
            if np.sign(oos.mean_ic) != np.sign(train.mean_ic) or abs(oos.mean_ic) < MIN_OOS_IC:
                c.rejected = f"oos: train={train.mean_ic:.4f} oos={oos.mean_ic:.4f}"

        # 결과 반영 + 경험 메모리 축적
        for c in candidates:
            if c.rejected:
                self.experience["forbidden"].append(
                    {"expression": c.expression, "reason": c.rejected}
                )
                events.append({"event": "factor_rejected", "name": c.name, "reason": c.rejected})
            elif c.result:
                record = FactorRecord(
                    name=c.name,
                    expression=c.expression,
                    hypothesis=c.hypothesis,
                    status="active",
                    train_ic=round(c.result.train.mean_ic, 5),
                    train_icir=round(c.result.train.icir, 4),
                    oos_ic=round(c.result.oos.mean_ic, 5),
                    oos_icir=round(c.result.oos.icir, 4),
                    admitted_day=asof_day.isoformat(),
                    sign=int(np.sign(c.result.train.mean_ic)),
                )
                self.factors.append(record)
                self.experience["successful"].append(
                    {"expression": c.expression, "oos_ic": record.oos_ic,
                     "hypothesis": c.hypothesis}
                )
                events.append(
                    {"event": "factor_admitted", "name": c.name,
                     "train_ic": record.train_ic, "oos_ic": record.oos_ic,
                     "oos_icir": record.oos_icir}
                )
        # Forbidden 경험은 최근 30건만 유지 (프롬프트 크기 통제)
        self.experience["forbidden"] = self.experience["forbidden"][-30:]
        self.save()
        return events

    def review_decay(
        self, panel: dict[str, np.ndarray], dates: list[date], asof_day: date
    ) -> list[dict]:
        """active 팩터의 라이브(post-admission) 실현 IC 를 갱신하고, 우위가 감쇠한
        팩터를 retire ([ADR-022] — 알파 crowding 감쇠. 메모리 retention 과 대칭).

        라이브 IC 는 admission 이후 날짜의 rank-IC 만 집계 — 승격 표본과 분리(ADR-002).
        표본일 < MIN_LIVE_DAYS 면 증거 부족으로 유지(diversity 보존, 하드룰). 방향성
        라이브 IC(live_ic × sign)가 DECAY_FLOOR 이하로 떨어지면 우위 소멸 → retire.
        """
        fwd = forward_returns(panel["close"])
        events: list[dict] = []
        for f in self.active():
            try:
                scores = evaluate(f.expression, panel)
            except DSLError:
                continue
            ics = daily_rank_ic(scores, fwd)
            admitted = date.fromisoformat(f.admitted_day)
            live = np.array(
                [ics[i] for i, d in enumerate(dates) if d > admitted and not np.isnan(ics[i])]
            )
            if len(live) < MIN_LIVE_DAYS:
                continue  # 표본 부족 — 판단 보류(유지)
            f.live_ic = round(float(live.mean()), 5)
            f.live_ic_n = int(len(live))
            f.live_ic_day = asof_day.isoformat()
            if f.live_ic * f.sign <= DECAY_FLOOR:
                f.status = "retired"
                events.append(
                    {
                        "event": "factor_decayed",
                        "name": f.name,
                        "live_ic": f.live_ic,
                        "live_n": f.live_ic_n,
                        "admission_oos_ic": f.oos_ic,
                    }
                )
        self.save()
        return events
