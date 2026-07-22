"""자격증명 검증 스크립트 — .env의 키가 실제로 동작하는지 API 핑으로 판정.

사용법:
    uv run python scripts/check_credentials.py

- 대상: 3종 (Binance testnet / Alpaca paper / KIS 모의투자).
- 키가 비어 있으면 status=skip (실패 아님). 확장 시장 키는 설정 여부만 보고.
- 출력: key=value 구조화 로그.
- 종료코드: 설정된 키 중 하나라도 인증 실패 시 1, 아니면 0.
- 주의: KIS 토큰 발급은 분당 1회 제한 — 연속 실행 시 EGW00133류 오류는 재시도로 해석.
- 의존성: ccxt(기존 의존성) + 표준 라이브러리만. 외부 추가 설치 불필요.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import load_env  # noqa: E402

ENV_PATH = ROOT / ".env"

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
KIS_PAPER_BASE = "https://openapivts.koreainvestment.com:29443"
TIMEOUT = 15


def report(check: str, status: str, detail: str = "") -> None:
    print(f"check={check} status={status}" + (f" detail={detail}" if detail else ""))


def http_json(req: urllib.request.Request) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {}
        return e.code, body


def check_binance_testnet(env: dict[str, str]) -> str:
    key, secret = env.get("BINANCE_TESTNET_API_KEY"), env.get("BINANCE_TESTNET_SECRET")
    if not key or not secret:
        report("binance_testnet", "skip", "키 미설정")
        return "skip"
    try:
        import ccxt  # 기존 의존성

        ex = ccxt.binance({"apiKey": key, "secret": secret})
        ex.set_sandbox_mode(True)  # testnet.binance.vision으로 라우팅
        balance = ex.fetch_balance()
        assets = sum(1 for v in balance.get("total", {}).values() if v)
        report("binance_testnet", "ok", f"인증 성공, 보유자산 {assets}종")
        return "ok"
    except Exception as e:
        report("binance_testnet", "fail", str(e)[:200])
        return "fail"


def check_alpaca_paper(env: dict[str, str]) -> str:
    key, secret = env.get("ALPACA_PAPER_API_KEY"), env.get("ALPACA_PAPER_SECRET")
    if not key or not secret:
        report("alpaca_paper", "skip", "키 미설정")
        return "skip"
    req = urllib.request.Request(
        f"{ALPACA_PAPER_BASE}/v2/account",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
    )
    try:
        status, body = http_json(req)
    except Exception as e:
        report("alpaca_paper", "fail", str(e)[:200])
        return "fail"
    if status == 200 and body.get("status") == "ACTIVE":
        report("alpaca_paper", "ok", f"계좌 ACTIVE, equity={body.get('equity')}")
        return "ok"
    report("alpaca_paper", "fail", f"http={status} body={str(body)[:150]}")
    return "fail"


def check_kis_paper(env: dict[str, str]) -> str:
    app_key = env.get("KIS_PAPER_APP_KEY")
    app_secret = env.get("KIS_PAPER_APP_SECRET")
    account = env.get("KIS_PAPER_ACCOUNT")
    if not app_key or not app_secret:
        report("kis_paper", "skip", "키 미설정")
        return "skip"
    if not account or "-" not in account:
        report("kis_paper", "fail", "KIS_PAPER_ACCOUNT 형식 오류 (예: 12345678-01)")
        return "fail"
    payload = json.dumps(
        {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
    ).encode()
    req = urllib.request.Request(
        f"{KIS_PAPER_BASE}/oauth2/tokenP",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        status, body = http_json(req)
    except Exception as e:
        report("kis_paper", "fail", str(e)[:200])
        return "fail"
    if status == 200 and body.get("access_token"):
        report("kis_paper", "ok", f"모의투자 토큰 발급 성공 (expires_in={body.get('expires_in')})")
        return "ok"
    # 분당 1회 제한(EGW00133)은 키 자체는 유효할 수 있음 — 안내만
    msg = body.get("error_description") or body.get("msg1") or str(body)[:150]
    report("kis_paper", "fail", f"http={status} msg={msg}")
    return "fail"


def check_optional(env: dict[str, str]) -> None:
    """확장 시장 키 — 설정 여부만 보고 (인증 핑 안 함)."""
    for name, phase in [
        ("ANTHROPIC_API_KEY", "phase1"),
        ("FRED_API_KEY", "phase3"),
        ("DART_API_KEY", "phase3"),
    ]:
        status = "set" if env.get(name) else "unset"
        report(name.lower(), status, phase)


def main() -> int:
    if not ENV_PATH.exists():
        report("env_file", "fail", ".env 없음 — .env.example을 복사해 키 기입")
        return 1
    env = load_env(ENV_PATH)
    report("env_file", "ok", str(ENV_PATH))

    results = [
        check_binance_testnet(env),
        check_alpaca_paper(env),
        check_kis_paper(env),
    ]
    check_optional(env)

    ok = results.count("ok")
    fail = results.count("fail")
    skip = results.count("skip")
    print(f"summary ok={ok} fail={fail} skip={skip} phase0_ready={fail == 0 and ok == 3}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
