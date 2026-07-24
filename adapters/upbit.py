"""Upbit 자금 이체 어댑터 — 버킷 간 자본 이동의 자동 레그(크립토버킷 → 은행).

ccxt upbit 래핑. **거래(주문·시세)는 미포함** — 크립토 라이브 체결 어댑터는 별도.
이 클래스는 이체 계층의 'Upbit KRW 출금' 자동 레그 전용이다(TreasuryCapable 구현).

- withdrawable_krw: 가용 KRW 조회(멱등 GET, 재시도 O).
- withdraw_krw: KRW 출금 집행. **비멱등·실자금 이동** — 재시도 금지(timeout 재시도 시
  이중 출금 위험). 이체 가드(allowlist·상한·쿨다운) 통과분만 호출. 목적지는 KYC
  등록 계좌 고정(자유 주소 아님 → 목적지 조작 불가).
"""

from __future__ import annotations

from adapters.retry import with_retry


class UpbitTreasury:
    """Upbit KRW 자금 이체(자동 레그). ccxt upbit 로 잔고 조회 + KRW 출금만 담당."""

    venue = "UPBIT"

    def __init__(self, api_key: str, secret: str) -> None:
        import ccxt.async_support as ccxt_async

        # timeout(ms) 명시 — 다른 브로커 어댑터 REST(15s)와 통일
        self.ex = ccxt_async.upbit({"apiKey": api_key, "secret": secret, "timeout": 15000})

    async def close(self) -> None:
        """aiohttp 세션 정리. 사용 후 반드시 호출."""
        await self.ex.close()

    async def withdrawable_krw(self) -> float:
        """출금 가능한 KRW 가용 잔고(free). 잠금·미체결분 제외."""
        balance = await with_retry(self.ex.fetch_balance)
        return float(balance.get("free", {}).get("KRW") or 0)

    async def withdraw_krw(self, amount: float) -> dict:
        """등록 계좌로 KRW 출금 — **실자금 이동, 비멱등**. 이체 가드 통과 후에만 호출.

        ccxt withdraw(code="KRW") 는 /withdraws/krw 로 라우팅되며 address 인자는 무시된다
        (KRW 목적지 = 거래소 KYC 등록 계좌 고정). 재시도 없음 — 이중 출금 방지.
        일부 계정은 Upbit 가 two_factor_type 를 요구할 수 있다(활성화 시 확인해 params 전달).
        """
        tx = await self.ex.withdraw("KRW", amount, "")
        info = tx.get("info") or {}
        return {
            "uuid": tx.get("id") or info.get("uuid"),
            "state": info.get("state"),
            "amount": float(tx.get("amount") or amount),
            "raw": info,
        }
