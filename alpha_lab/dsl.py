"""공식형 팩터 DSL — AST 화이트리스트 안전 평가기 (R11 · [ADR-016]).

writer LLM 이 생성하는 표현식을 파이썬 eval 없이 AST 로 파싱·검증·평가한다.
팩터는 *수식*이라 lookahead 민감도가 낮다(FactorMiner 논리) — 단 DSL 자체가
미래 참조를 표현할 수 없도록 연산자를 과거 참조(ts_*/delay)로만 제한한다.

데이터 모델: 패널 = dict[field, np.ndarray(T, N)] — T 일 × N 자산, 날짜 오름차순.
모든 연산은 (T, N) → (T, N). 워밍업 구간은 NaN.
"""

from __future__ import annotations

import ast

import numpy as np

FIELDS = ("open", "high", "low", "close", "volume", "returns")
MAX_NODES = 60  # 복잡도 상한 — 과적합 수식 차단
MAX_WINDOW = 120


class DSLError(ValueError):
    """표현식이 DSL 계약 위반 (금지 연산·미지 심볼·복잡도 초과)."""


# ── 시계열 연산 (axis=0 = 시간) — 전부 과거 참조만 ──


def _delay(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if n > 0:
        out[n:] = x[:-n]
    else:
        raise DSLError(f"delay n={n} — 미래 참조 금지 (n ≥ 1)")
    return out


def _delta(x: np.ndarray, n: int) -> np.ndarray:
    return x - _delay(x, n)


def _rolling(x: np.ndarray, n: int, fn) -> np.ndarray:
    if not 1 < n <= MAX_WINDOW:
        raise DSLError(f"window n={n} 범위 위반 (2..{MAX_WINDOW})")
    out = np.full_like(x, np.nan)
    for t in range(n - 1, x.shape[0]):
        out[t] = fn(x[t - n + 1 : t + 1])
    return out


def _ts_mean(x, n):
    return _rolling(x, n, lambda w: np.nanmean(w, axis=0))


def _ts_std(x, n):
    return _rolling(x, n, lambda w: np.nanstd(w, axis=0))


def _ts_min(x, n):
    return _rolling(x, n, lambda w: np.nanmin(w, axis=0))


def _ts_max(x, n):
    return _rolling(x, n, lambda w: np.nanmax(w, axis=0))


def _ts_rank(x, n):
    """윈도우 내 현재값의 백분위 (0~1)."""

    def pct(w):
        cur = w[-1]
        with np.errstate(invalid="ignore"):
            return np.nansum(w <= cur, axis=0) / w.shape[0]

    return _rolling(x, n, pct)


def _ts_corr(x, y, n):
    def corr(wx, wy):
        mx, my = np.nanmean(wx, axis=0), np.nanmean(wy, axis=0)
        cov = np.nanmean((wx - mx) * (wy - my), axis=0)
        sx, sy = np.nanstd(wx, axis=0), np.nanstd(wy, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sx * sy > 0, cov / (sx * sy), np.nan)

    if not 1 < n <= MAX_WINDOW:
        raise DSLError(f"window n={n} 범위 위반")
    out = np.full_like(x, np.nan)
    for t in range(n - 1, x.shape[0]):
        out[t] = corr(x[t - n + 1 : t + 1], y[t - n + 1 : t + 1])
    return out


# ── 횡단면 연산 (axis=1 = 자산) — 당일 값만 사용, 누출 없음 ──


def _rank(x: np.ndarray) -> np.ndarray:
    """당일 횡단면 백분위 (0~1). NaN 은 NaN 유지."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        valid = ~np.isnan(row)
        if valid.sum() >= 2:
            order = row[valid].argsort().argsort().astype(float)
            out[t, valid] = order / (valid.sum() - 1)
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, (x - mean) / std, np.nan)


# ── 원소 연산 ──


def _sign(x):
    return np.sign(x)


def _abs(x):
    return np.abs(x)


def _log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log(x), np.nan)


FUNCS = {
    "delay": (_delay, 2),
    "delta": (_delta, 2),
    "ts_mean": (_ts_mean, 2),
    "ts_std": (_ts_std, 2),
    "ts_min": (_ts_min, 2),
    "ts_max": (_ts_max, 2),
    "ts_rank": (_ts_rank, 2),
    "ts_corr": (_ts_corr, 3),
    "rank": (_rank, 1),
    "zscore": (_zscore, 1),
    "sign": (_sign, 1),
    "abs": (_abs, 1),
    "log": (_log, 1),
}

DSL_SPEC = (
    "필드: open, high, low, close, volume, returns (일간 패널)\n"
    "함수: delay(x,n) delta(x,n) ts_mean(x,n) ts_std(x,n) ts_min(x,n) ts_max(x,n) "
    "ts_rank(x,n) ts_corr(x,y,n) rank(x) zscore(x) sign(x) abs(x) log(x)\n"
    "연산: + - * / 및 단항 - · 숫자 상수. n 은 2..120 정수(delay 는 1..120).\n"
    "미래 참조 불가 — delay/ts_* 는 과거만 본다. 예: rank(-delta(close, 5) / ts_std(returns, 20))"
)


def validate(expression: str) -> ast.Expression:
    """파싱 + 화이트리스트 검증. 위반 시 DSLError."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise DSLError(f"파싱 실패: {e}") from e
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > MAX_NODES:
        raise DSLError(f"복잡도 초과: nodes={n_nodes} > {MAX_NODES}")
    func_names = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }  # 함수명 Name 노드는 필드 검사에서 제외
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if id(node) in func_names:
                continue
            if node.id not in FIELDS:
                raise DSLError(f"미지 필드: {node.id}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                raise DSLError(f"미지 함수: {ast.dump(node.func)[:50]}")
            if node.keywords:
                raise DSLError("키워드 인자 금지")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                raise DSLError(f"금지 연산: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.USub):
                raise DSLError(f"금지 단항: {type(node.op).__name__}")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise DSLError(f"금지 상수: {node.value!r}")
        elif not isinstance(
            node,
            (ast.Expression, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub),
        ):  # 연산자 타입 노드는 BinOp/UnaryOp 분기에서 이미 검증됨
            raise DSLError(f"금지 구문: {type(node).__name__}")
    return tree


def evaluate(expression: str, panel: dict[str, np.ndarray]) -> np.ndarray:
    """표현식 → (T, N) 팩터 행렬. panel 은 FIELDS 의 부분집합."""
    tree = validate(expression)

    def ev(node) -> np.ndarray | float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Name):
            if node.id not in panel:
                raise DSLError(f"패널에 없는 필드: {node.id}")
            return panel[node.id]
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.asarray(left) / np.asarray(right)
                return np.where(np.isfinite(out), out, np.nan)
        if isinstance(node, ast.Call):
            fn, arity = FUNCS[node.func.id]
            if len(node.args) != arity:
                raise DSLError(f"{node.func.id} 인자 수 {len(node.args)} ≠ {arity}")
            args = []
            for i, arg in enumerate(node.args):
                # 윈도우/지연 인자(마지막 자리, ts_corr 은 3번째)는 정수 상수만
                is_window = (arity >= 2 and i == arity - 1) and node.func.id != "rank"
                if is_window:
                    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, int)):
                        raise DSLError(f"{node.func.id} 윈도우는 정수 상수만")
                    args.append(arg.value)
                else:
                    val = ev(arg)
                    if isinstance(val, float):
                        raise DSLError(f"{node.func.id} 인자는 시리즈여야 한다")
                    args.append(val)
            return fn(*args)
        raise DSLError(f"평가 불가 노드: {type(node).__name__}")

    result = ev(tree)
    if isinstance(result, float) or result.ndim != 2:
        raise DSLError("결과가 (T, N) 패널이 아님 — 상수/스칼라 표현식 금지")
    return result.astype(float)
