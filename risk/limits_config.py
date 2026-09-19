"""시장별 리스크 한도의 조립 — 기준값 파일 + 운용 중 덮어쓰기 → RiskLimits.

한도는 세 겹이다.
- 바깥 한계(BOUNDS) — 이 모듈의 상수. 덮어쓰기가 움직일 수 있는 항목과 범위이며 코드 변경으로만
  바뀐다. 여기 없는 항목(일일 회전율·낙폭 서킷)은 덮어쓸 수 없다: 어느 쪽으로 움직여도 위험이
  늘 수 있는 값이라(회전율을 낮추면 현금화도 느려지고, 서킷은 청산이 아니라 동결이다) 기준값
  변경과 같은 경로를 거친다.
- 기준값(limits.toml) — 코드와 같은 경로(리뷰·커밋)로만 바뀌는 값.
- 덮어쓰기(data/state/limits_override.json) — 로컬 CLI 가 남기는 운용 중 조정. 지우면 기준값.

검증은 범위 비교로 끝내지 않고 **실제 조립**을 해 본다. 지수 ETF 의 상한은 최소 현금과 개별주
상한에서 도출되므로 범위 안의 값도 조립에서 실패할 수 있다 — 개별주 상한을 낮추면 ETF 의 최대
구성종목 노출이 그 상한을 넘는다. 읽을 때와 검증할 때 같은 조립 함수를 쓴다.

덮어쓰기가 조립되지 않는 시장은 기준값으로 돌아가고 사유를 errors 에 남긴다. 잘못된 파일 하나로
그날의 결정이 통째로 사라지는 것보다 리뷰를 거친 값으로 도는 편이 낫다 — 호출부는 errors 를
로그와 통지로 드러내야 한다(조용히 기준값으로 도는 것은 조인 한도가 풀린 채 도는 것일 수 있다).
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from adapters.universe import resolve_asset_caps
from risk.engine import RiskLimits, limits_payload, limits_rev

BASELINE_PATH = Path(__file__).with_name("limits.toml")
OVERRIDE_REL = Path("data") / "state" / "limits_override.json"
REV_DIR_REL = Path("data") / "state" / "config_revs"

#: 덮어쓸 수 있는 항목과 그 범위(양끝 포함). 범위 밖은 기준값 변경 — 코드 리뷰 — 으로만.
BOUNDS: dict[str, tuple[float, float]] = {
    "min_cash": (0.05, 1.0),
    "max_weight_per_asset": (0.01, 1.0),
}
#: 값이 커질수록 조이는 항목. 나머지는 작아질수록 조인다.
_TIGHTER_WHEN_HIGHER = frozenset({"min_cash"})
_FIELDS = ("max_weight_per_asset", "min_cash", "max_daily_turnover", "mdd_circuit")


@dataclass(frozen=True)
class LimitsConfig:
    limits: dict[str, RiskLimits]
    baseline: dict[str, dict[str, float]]
    overrides: dict[str, dict[str, float]]  # 실제로 적용된 덮어쓰기만
    errors: list[str] = field(default_factory=list)  # key=value — 버려진 덮어쓰기와 그 사유


def load_baseline(path: Path | str = BASELINE_PATH) -> dict[str, dict[str, float]]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return {market: {k: float(values[k]) for k in _FIELDS} for market, values in data.items()}


def read_overrides(root: Path | str) -> dict[str, dict[str, float]]:
    """덮어쓰기 파일 원문. 없으면 {}. 형식이 깨졌으면 ValueError."""
    path = Path(root) / OVERRIDE_REL
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"override_unreadable path={path} error={e}") from e
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise ValueError(f"override_malformed path={path}")
    return data


def assemble(values: dict[str, float], symbols: list[str]) -> RiskLimits:
    """한 시장의 값 → RiskLimits. 읽기와 검증이 함께 쓰는 유일한 조립 경로."""
    return RiskLimits(
        max_weight_per_asset=values["max_weight_per_asset"],
        min_cash=values["min_cash"],
        max_daily_turnover=values["max_daily_turnover"],
        mdd_circuit=values["mdd_circuit"],
        # ETF 상한은 최대 구성종목 노출이 개별주 상한을 넘지 않는 선에서 따로 도출한다 —
        # 부등식이 깨지면 여기서 ValueError
        asset_caps=resolve_asset_caps(symbols, values["max_weight_per_asset"], values["min_cash"]),
        # 정의역을 엔진에도 알린다 — 결정 단계 검증만으로는 직전 배분에 남은 옛 종목이
        # turnover blend 를 타고 되살아난다.
        tradable=frozenset(symbols),
    )


def check_value(key: str, value) -> str | None:
    """덮어쓰기 값 1건의 항목·형식·범위 검사. 문제없으면 None."""
    if key not in BOUNDS:
        return f"not_overridable key={key} allowed={','.join(sorted(BOUNDS))}"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return f"not_a_number key={key} value={value!r}"
    low, high = BOUNDS[key]
    if not low <= value <= high:
        return f"out_of_bounds key={key} value={value} low={low} high={high}"
    return None


def check_market(
    baseline: dict[str, float], override: dict, symbols: list[str]
) -> tuple[RiskLimits | None, list[str]]:
    """한 시장의 덮어쓰기 전체를 검사하고 조립해 본다. (조립 결과, 오류)."""
    errors = [e for e in (check_value(k, v) for k, v in override.items()) if e]
    if errors:
        return None, errors
    try:
        return assemble({**baseline, **override}, symbols), []
    except ValueError as e:
        return None, [f"not_loadable {e}"]


def load_limits(
    root: Path | str, tradable: dict[str, list[str]], baseline_path: Path | str = BASELINE_PATH
) -> LimitsConfig:
    """기준값 + 덮어쓰기 → 시장별 RiskLimits. 기준값이 조립되지 않으면 예외(코드 결함)."""
    baseline = load_baseline(baseline_path)
    errors: list[str] = []
    try:
        raw = read_overrides(root)
    except ValueError as e:
        raw, errors = {}, [str(e)]
    limits, applied = {}, {}
    for market, symbols in tradable.items():
        override = raw.get(market) or {}
        built, problems = check_market(baseline[market], override, symbols)
        if built is None:
            errors += [f"override_dropped market={market} {p}" for p in problems]
            built, override = assemble(baseline[market], symbols), {}
        limits[market] = built
        if override:
            applied[market] = dict(override)
    errors += [f"override_unknown_market market={m}" for m in raw if m not in tradable]
    return LimitsConfig(limits=limits, baseline=baseline, overrides=applied, errors=errors)


def direction(key: str, old: float, new: float) -> str:
    """값 변화의 방향 — tighten · loosen · same."""
    if new == old:
        return "same"
    return "tighten" if (new > old) == (key in _TIGHTER_WHEN_HIGHER) else "loosen"


def write_overrides(root: Path | str, overrides: dict[str, dict[str, float]]) -> Path:
    """덮어쓰기 저장(빈 시장은 뺀다). 임시 파일에 쓰고 바꿔치기한다 — 쓰는 도중 시작한 잡이
    반쯤 쓰인 파일을 읽지 않게."""
    path = Path(root) / OVERRIDE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = {m: v for m, v in sorted(overrides.items()) if v}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(kept, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def record_limits_rev(
    root: Path | str,
    market: str,
    limits: RiskLimits,
    override: dict[str, float] | None = None,
    asof_day: date | None = None,
) -> bool:
    """한도 판본을 처음 볼 때만 내용을 저장한다 → True 는 신규. 결정 기록에는 지문만 실리므로
    이 스냅샷이 없으면 나중에 그 지문이 어떤 한도였는지 답할 수 없다."""
    rev = limits_rev(limits)
    path = Path(root) / REV_DIR_REL / f"{rev}.json"
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rev": rev,
                "market": market,
                "first_seen": (asof_day or date.today()).isoformat(),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "limits": limits_payload(limits),
                "override": dict(override or {}),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return True
