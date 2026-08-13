"""통계적 jump model — 2상태 국면 분류 (결정론 · 순수 함수 + 명시적 상태 파일).

k-means 목적함수에 **전환당 고정비용(점프 페널티 λ)** 을 더한 것이다:

    min_{Θ,S}  Σ_t ½‖x_t − θ_{s_t}‖²  +  λ · Σ_t 1{s_t ≠ s_{t−1}}

λ=0 이면 시간 정보를 무시하는 순수 k-means 가 되고, λ 가 커질수록 전환이 드물어진다.
지속성이 룰 곳곳에 흩어져 있는 분산일·FTD FSM(`regime/pulse.py`)과 달리 **손잡이 하나로**
조절된다는 것이 이 모델의 요점이다. 최적화는 좌표하강 — 중심점 Θ 갱신과 상태열 S 의
동적계획 갱신을 번갈아 한다. k=2 라 DP 는 O(n·4) 로 사실상 공짜다.

**λ 는 튜닝하지 않는다**(50.0 고정). 원 논문은 검증창 Sharpe 를 최적화해 λ 를 고르지만,
비교 상대인 FSM 이 외부 출처 상수를 우리 표본에 맞춘 적 없는 무튜닝 모델이다. 한쪽만
우리 데이터로 조율하면 판정이 그쪽에 유리하게 기운다. 성과는 논문보다 낮게 나올 수
있고, 그게 정직한 비교의 대가다.

**라벨 고정이 이 모델의 최대 함정이다.** 군집 번호(0/1)는 임의라 재적합마다 뒤집힐 수
있다. 훈련 구간 누적수익이 높은 쪽을 bull 로 정해 상태 파일에 함께 저장한다 — 저장하지
않으면 시계열이 중간에 부호 반전되고, 값은 계속 나오므로 로그만 봐서는 드러나지 않는다.

표준화 파라미터(평균·표준편차)도 훈련창의 것을 저장해 추론에 그대로 쓴다. 매일 그날
창으로 다시 표준화하면 같은 시장 상태가 날마다 다른 좌표에 찍힌다.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

K_STATES = 2
JUMP_PENALTY = 50.0  # 표준화 좌표 기준 — 원 논문의 통상값, 우리 표본으로 튜닝 금지
MAX_ITER = 30  # 좌표하강 상한 (보통 10회 안에 상태열이 고정된다)
N_RESTARTS = 10  # 초기값 무작위 재시작 — 목적값 최소인 것을 채택
MIN_TRAIN_BARS = 750  # 학습 최소 표본(≈3년) — 창에 두 국면이 안 들어오면 잡음을 쪼갠다
REFIT_DAYS = 180  # 재적합 주기 — Θ 안정성을 위해 자주 갱신하지 않는다
BULL, BEAR = "bull", "bear"


@dataclass(frozen=True)
class JumpModel:
    centroids: list[list[float]]  # k × d, 표준화 좌표
    mean: list[float]  # 훈련창 피처 평균 (추론에 재사용)
    std: list[float]  # 훈련창 피처 표준편차
    bull_index: int  # 어느 군집이 bull 인가 (누적수익 기준, 재적합 간 부호 고정)
    penalty: float
    n_train: int
    trained_through: str  # ISO 날짜 — 재적합 시점 판단


def _sq_dist(row: list[float], centroid: list[float]) -> float:
    return 0.5 * sum((a - b) ** 2 for a, b in zip(row, centroid))


def standardize(rows: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """열별 표준화 + 사용한 평균·표준편차 반환. 분산 0 인 열은 1.0 으로 나눈다."""
    d = len(rows[0])
    mean = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
    std = []
    for j in range(d):
        var = sum((r[j] - mean[j]) ** 2 for r in rows) / len(rows)
        std.append(var**0.5 or 1.0)
    return apply_standardize(rows, mean, std), mean, std


def apply_standardize(
    rows: list[list[float]], mean: list[float], std: list[float]
) -> list[list[float]]:
    return [[(v - m) / s for v, m, s in zip(row, mean, std)] for row in rows]


def assign_states(
    rows: list[list[float]], centroids: list[list[float]], penalty: float
) -> tuple[list[int], float]:
    """목적함수를 최소화하는 상태열 — 동적계획. (상태열, 목적값) 반환.

    전환마다 penalty 를 물리므로 상태가 끈적해진다. 매 시점 최근접 중심점을 그냥
    고르는 방식(penalty 무시)보다 훨씬 덜 흔들린다.
    """
    k = len(centroids)
    loss = [[_sq_dist(row, c) for c in centroids] for row in rows]
    cost = list(loss[0])
    back: list[list[int]] = []

    for t in range(1, len(rows)):
        nxt, choice = [0.0] * k, [0] * k
        for s in range(k):
            best_j, best_v = 0, float("inf")
            for j in range(k):
                v = cost[j] + (0.0 if j == s else penalty)
                if v < best_v:
                    best_j, best_v = j, v
            nxt[s], choice[s] = loss[t][s] + best_v, best_j
        back.append(choice)
        cost = nxt

    s = min(range(k), key=lambda i: cost[i])
    objective, states = cost[s], [s]
    for choice in reversed(back):
        s = choice[s]
        states.append(s)
    states.reverse()
    return states, objective


def _update_centroids(
    rows: list[list[float]], states: list[int], previous: list[list[float]]
) -> list[list[float]]:
    """상태별 평균으로 중심점 갱신. 비어 있는 상태는 직전 값을 유지(군집 붕괴 방지)."""
    d = len(rows[0])
    out = []
    for s in range(len(previous)):
        members = [row for row, st in zip(rows, states) if st == s]
        if not members:
            out.append(list(previous[s]))
            continue
        out.append([sum(m[j] for m in members) / len(members) for j in range(d)])
    return out


def fit(
    rows: list[list[float]],
    returns: list[float],
    trained_through: date,
    penalty: float = JUMP_PENALTY,
    restarts: int = N_RESTARTS,
    seed: int = 0,
) -> JumpModel | None:
    """피처 행렬 + 같은 날 수익률 → 적합된 모델. 표본 부족이면 None.

    seed 고정이라 같은 입력이면 같은 모델이 나온다 — 재현 불가능한 적합은 감사할 수 없다.
    """
    if len(rows) < MIN_TRAIN_BARS or len(rows) != len(returns):
        return None

    std_rows, mean, std = standardize(rows)
    rng = random.Random(seed)
    best: tuple[float, list[int], list[list[float]]] | None = None

    for _ in range(restarts):
        centroids = [list(std_rows[i]) for i in rng.sample(range(len(std_rows)), K_STATES)]
        states: list[int] = []
        for _ in range(MAX_ITER):
            new_states, _ = assign_states(std_rows, centroids, penalty)
            if new_states == states:
                break
            states = new_states
            centroids = _update_centroids(std_rows, states, centroids)
        states, objective = assign_states(std_rows, centroids, penalty)
        if best is None or objective < best[0]:
            best = (objective, states, centroids)

    if best is None:  # restarts=0 로 부른 경우
        return None
    _, states, centroids = best
    # 누적수익이 높은 군집이 bull — 군집 번호는 임의라 이 배정을 저장해야 부호가 고정된다
    totals = [
        sum(r for r, s in zip(returns, states) if s == k) for k in range(K_STATES)
    ]
    return JumpModel(
        centroids=[[round(v, 6) for v in c] for c in centroids],
        mean=[round(v, 6) for v in mean],
        std=[round(v, 6) for v in std],
        bull_index=max(range(K_STATES), key=lambda k: totals[k]),
        penalty=penalty,
        n_train=len(rows),
        trained_through=trained_through.isoformat(),
    )


def infer(rows: list[list[float]], model: JumpModel) -> str | None:
    """최근 창의 마지막 상태 라벨(bull/bear). 행이 없으면 None.

    최근접 중심점을 그냥 고르지 않고 창 전체에 DP 를 돌린 뒤 마지막 상태를 취한다 —
    페널티가 반영돼야 추론 상태열도 훈련만큼 끈적해진다.
    """
    if not rows:
        return None
    std_rows = apply_standardize(rows, model.mean, model.std)
    states, _ = assign_states(std_rows, model.centroids, model.penalty)
    return BULL if states[-1] == model.bull_index else BEAR


def needs_refit(model: JumpModel | None, asof_day: date, every: int = REFIT_DAYS) -> bool:
    """모델이 없거나 마지막 적합이 every 일보다 오래됐으면 True."""
    if model is None:
        return True
    return (asof_day - date.fromisoformat(model.trained_through)).days >= every


def load_model(path: Path | str) -> JumpModel | None:
    """상태 파일 → 모델. 없거나 스키마가 안 맞으면 None(재적합 유도)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return JumpModel(**json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def save_model(path: Path | str, model: JumpModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(model), ensure_ascii=False, indent=1), encoding="utf-8")
