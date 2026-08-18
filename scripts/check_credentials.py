"""자격증명 검증 스크립트 — .env의 키가 실제로 동작하는지 API 핑으로 판정.

사용법:
    uv run python scripts/check_credentials.py

- 대상: 페이퍼 3종 (Binance testnet / Alpaca paper / KIS 모의투자) + Upbit 실계좌.
- 키가 비어 있으면 status=skip (실패 아님). 확장 시장 키는 설정 여부만 보고.
- **조회만 한다** — 어떤 경로에서도 주문·출금을 내지 않는다. Upbit 실계좌도 잔고만 읽는다.
- 출력: key=value 구조화 로그.
- 종료코드: 설정된 키 중 하나라도 인증 실패 시 1, 아니면 0.
- 주의: KIS 토큰 발급은 분당 1회 제한 — 연속 실행 시 EGW00133류 오류는 재시도로 해석.
- 의존성: ccxt·httpx (모두 기존 필수 의존성). 외부 추가 설치 불필요.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.env import load_env  # noqa: E402

ENV_PATH = ROOT / ".env"

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
KIS_PAPER_BASE = "https://openapivts.koreainvestment.com:29443"
TIMEOUT = 15


def report(check: str, status: str, detail: str = "") -> None:
    print(f"check={check} status={status}" + (f" detail={detail}" if detail else ""))


def _json_or_empty(resp: httpx.Response) -> dict:
    try:
        return resp.json()
    except ValueError:
        return {}


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
    try:
        resp = httpx.get(
            f"{ALPACA_PAPER_BASE}/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=TIMEOUT,
        )
    except Exception as e:
        report("alpaca_paper", "fail", str(e)[:200])
        return "fail"
    body = _json_or_empty(resp)
    if resp.status_code == 200 and body.get("status") == "ACTIVE":
        report("alpaca_paper", "ok", f"계좌 ACTIVE, equity={body.get('equity')}")
        return "ok"
    report("alpaca_paper", "fail", f"http={resp.status_code} body={str(body)[:150]}")
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
    try:
        resp = httpx.post(
            f"{KIS_PAPER_BASE}/oauth2/tokenP",
            json={"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret},
            timeout=TIMEOUT,
        )
    except Exception as e:
        report("kis_paper", "fail", str(e)[:200])
        return "fail"
    body = _json_or_empty(resp)
    if resp.status_code == 200 and body.get("access_token"):
        report("kis_paper", "ok", f"모의투자 토큰 발급 성공 (expires_in={body.get('expires_in')})")
        return "ok"
    # 분당 1회 제한(EGW00133)은 키 자체는 유효할 수 있음 — 안내만
    msg = body.get("error_description") or body.get("msg1") or str(body)[:150]
    report("kis_paper", "fail", f"http={resp.status_code} msg={msg}")
    return "fail"


def check_upbit_live(env: dict[str, str]) -> str:
    """Upbit **실계좌** — 잔고 조회만. 주문·출금은 내지 않는다.

    판정하는 것은 '자산조회가 되는가'까지다. 주문 권한과 출금 권한은 조회 API 가
    돌려주지 않는다 — 주문 권한은 주문을 내야, 출금 권한은 출금을 시도해야 드러나므로
    확인 자체가 위험한 행위다. 허용 IP 도 마찬가지로, 호출이 성공했다는 사실만으로는
    '이 IP 가 등록돼 있다'와 'IP 제한이 아예 없다'를 구분할 수 없다.

    그래서 이 검사가 ok 라는 것은 **실주문을 낼 준비가 됐다는 뜻이 아니다**. 셋은
    거래소 콘솔에서 사람이 봐야 하고, 특히 출금 권한은 켜져 있으면 체결과 출금을
    갈라 둔 이 시스템의 경계가 무의미해진다.
    """
    key, secret = env.get("UPBIT_API_KEY"), env.get("UPBIT_SECRET")
    if not key or not secret:
        report("upbit_live", "skip", "키 미설정 — 크립토는 Binance testnet 폴백")
        return "skip"
    try:
        import ccxt  # 기존 의존성

        ex = ccxt.upbit({"apiKey": key, "secret": secret, "timeout": 15000})
        balance = ex.fetch_balance()
        krw = float((balance.get("free") or {}).get("KRW") or 0)
        coins = sum(1 for s, v in (balance.get("total") or {}).items() if s != "KRW" and v)
    except Exception as e:
        report("upbit_live", "fail", str(e)[:200])
        return "fail"
    report(
        "upbit_live", "ok",
        f"자산조회 성공, 주문가능 {krw:,.0f} KRW, 보유 {coins}종 "
        "(주문·출금 권한과 허용 IP 는 조회로 판정 불가 — 콘솔 확인 필요)",
    )
    return "ok"


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

    # 페이퍼 준비도와 실계좌 가용성은 다른 질문이다 — 실계좌 키가 없다고 페이퍼가
    # 덜 준비된 것이 아니고, 실계좌 키가 깨졌다고 페이퍼 판정을 뒤집을 이유도 없다.
    paper = [
        check_binance_testnet(env),
        check_alpaca_paper(env),
        check_kis_paper(env),
    ]
    live = [check_upbit_live(env)]
    check_optional(env)

    results = paper + live
    ok = results.count("ok")
    fail = results.count("fail")
    skip = results.count("skip")
    ready = paper.count("fail") == 0 and paper.count("ok") == 3
    print(f"summary ok={ok} fail={fail} skip={skip} phase0_ready={ready}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
