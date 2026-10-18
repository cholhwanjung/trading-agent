"""시장 간 메타 배분(LLM shadow) — 주간 제안·감사·원장·결과 기입.

시장 내부 트레이더와 분리된 상위 층이다. 입력은 시장별 **집계** 신호(국면·추세·변동성·
밸류에이션·매크로·뉴스 서사)뿐이고, 개별 포지션·메모리·시장 내부 결정은 넘어오지 않는다.
제안은 anchor ± max_tilt 안으로 결정론 clamp 되며, 집행은 없다(원장·채점만).
"""

from meta.audit import AuditResult, deterministic_audit, llm_audit
from meta.digest import assign_ids, digest_news, load_recent_news, select_news
from meta.ledger import last_record, load_llm_ledger, record_llm_proposal, week_key
from meta.outcomes import fill_outcomes, reliability_lines, signal_reliability
from meta.panel import MARKETS, build_panel, market_panel, signal_type
from meta.propose import MetaLLMProposal, MetaProposalError, clamp_weights, parse_proposal, propose

__all__ = [
    "MARKETS",
    "AuditResult",
    "MetaLLMProposal",
    "MetaProposalError",
    "assign_ids",
    "build_panel",
    "clamp_weights",
    "deterministic_audit",
    "digest_news",
    "fill_outcomes",
    "last_record",
    "llm_audit",
    "load_llm_ledger",
    "load_recent_news",
    "market_panel",
    "parse_proposal",
    "propose",
    "record_llm_proposal",
    "reliability_lines",
    "select_news",
    "signal_reliability",
    "signal_type",
    "week_key",
]
