"""프롬프트 블록 조립 + 판본 스냅샷 — 에이전트 행동을 코드가 아니라 데이터로 둔다.

두 가지를 한다.

**① 조립** — 결정 시스템 프롬프트를 파이썬 문자열 상수가 아니라 `prompts/manifest.toml`
이 선언한 블록 목록에서 만든다. 매니페스트에 없는 파일은 프롬프트에 들어갈 수 없다.
블록 하나가 곧 변경 단위라, 신호 의미론을 고치는 일이 출력 스키마를 건드리지 않는다.

**② 판본 스냅샷** — 조립된 정책 텍스트의 12자리 지문(`rev`)을 결정마다 기록하고,
그 판본을 처음 쓸 때 **원문 그대로** 스냅샷으로 남긴다. 지문만으로는 "무언가 바뀌었다"
까지만 알 수 있고 "무엇이었나"에 답하지 못한다 — 개정 전 텍스트를 복원할 수 없으면
판본별 성과 비교가 숫자 두 개로 끝난다.

스냅샷은 내용 주소화다: 파일명이 곧 그 내용의 해시라 사후에 손대면 이름이 스스로
틀려진다. 감사 추적에 필요한 성질이 저장 구조에서 나온다.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
BLOCK_DIR = PROMPT_DIR / "blocks"
MANIFEST_PATH = PROMPT_DIR / "manifest.toml"

#: 블록 파일에서 제거하는 것 — 사람이 읽는 주석은 모델 입력이 아니다.
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

_JOIN = {"section": "\n\n", "line": "\n"}


class PromptAssemblyError(ValueError):
    """매니페스트가 없는 블록을 가리키거나 조립 규칙이 깨졌다."""


@dataclass(frozen=True)
class PromptSpec:
    """한 에이전트 경로의 조립 결과. text 가 곧 rev 의 원본이다."""

    text: str  # 기본 시스템 프롬프트 (치환 전)
    trigger_text: str  # 트리거 절까지 붙인 판 (치환 전)
    blocks: list[str]
    trigger_blocks: list[str]
    tier: str

    @property
    def rev(self) -> str:
        return revision(self.trigger_text)


def load_block(name: str) -> str:
    """블록 1개의 본문. 주석 제거 후 양끝 개행 정리."""
    path = BLOCK_DIR / f"{name}.md"
    if not path.exists():
        raise PromptAssemblyError(f"블록 없음: {path}")
    return _COMMENT.sub("", path.read_text(encoding="utf-8")).strip("\n")


def _join(entries: list[dict], start: str = "") -> tuple[str, list[str]]:
    """블록 항목 목록 → (조립 텍스트, 블록 이름 목록)."""
    text, names = start, []
    for entry in entries:
        name = entry.get("name")
        if not name:
            raise PromptAssemblyError(f"블록 항목에 name 이 없다: {entry}")
        join = entry.get("join", "section")
        if join not in _JOIN:
            raise PromptAssemblyError(f"알 수 없는 join='{join}' (block={name})")
        body = load_block(name)
        text = body if not text else text + _JOIN[join] + body
        names.append(name)
    return text, names


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def assemble(manifest: dict | None = None) -> PromptSpec:
    """매니페스트 → 결정 프롬프트. 포맷 치환은 소비 시점에 `render` 가 한다."""
    manifest = manifest if manifest is not None else load_manifest()
    decision = manifest.get("decision")
    if not decision:
        raise PromptAssemblyError("manifest 에 [decision] 절이 없다")

    entries = decision.get("blocks") or []
    text, names = _join(entries)
    trigger_entries = (decision.get("trigger") or {}).get("blocks") or []
    trigger_text, trigger_names = _join(trigger_entries, start=text)
    return PromptSpec(
        text=text,
        trigger_text=trigger_text,
        blocks=names,
        trigger_blocks=trigger_names,
        tier=decision.get("tier", "smart"),
    )


def _formatted(entries: list[dict]) -> set[str]:
    return {e["name"] for e in entries if e.get("format")}


def render(spec: PromptSpec, manifest: dict, *, trigger: bool = False, **values) -> str:
    """조립 텍스트에 {market}·{universe}·{tradable} 를 치환해 실제 프롬프트를 만든다.

    치환은 `format = true` 인 블록에만 적용한다 — 중괄호가 없는 블록(플레이북 등)에
    format 을 걸면 본문의 중괄호가 치환 문법으로 오인돼 조용히 깨지거나 터진다.
    """
    decision = manifest["decision"]
    fmt = _formatted(decision.get("blocks") or [])
    fmt |= _formatted((decision.get("trigger") or {}).get("blocks") or [])

    entries = list(decision.get("blocks") or [])
    if trigger:
        entries += list((decision.get("trigger") or {}).get("blocks") or [])

    out = ""
    for entry in entries:
        body = load_block(entry["name"])
        if entry["name"] in fmt:
            body = body.format(**values)
        join = _JOIN[entry.get("join", "section")]
        out = body if not out else out + join + body
    return out


# ── 판본 ─────────────────────────────────────────────────────────────────────


def revision(policy_text: str) -> str:
    """정책 텍스트의 12자리 지문. 치환 **전** 텍스트를 해싱한다.

    시장별 포맷을 적용한 뒤 해싱하면 세 시장이 다른 값을 갖게 되어, 시장 차이와
    개정 차이가 한 필드에 섞인다. 판본이 같으면 전 시장이 같은 값이어야 판본별
    성과 비교가 성립한다.
    """
    return hashlib.sha256(policy_text.encode("utf-8")).hexdigest()[:12]


def snapshot_path(store_dir: Path | str, rev: str) -> Path:
    return Path(store_dir) / f"{rev}.json"


def record_revision(
    store_dir: Path | str, spec: PromptSpec, asof_day: date | None = None
) -> bool:
    """판본을 처음 볼 때만 원문을 저장. 이미 있으면 손대지 않는다 → True 는 신규 기록.

    `policy_text` 가 `rev` 의 해시 원본이라, 읽는 쪽이 다시 해싱해 파일이 그 판본의
    것임을 스스로 검증할 수 있다.
    """
    path = snapshot_path(store_dir, spec.rev)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rev": spec.rev,
                "first_seen": (asof_day or date.today()).isoformat(),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "tier": spec.tier,
                "blocks": spec.blocks,
                "trigger_blocks": spec.trigger_blocks,
                "policy_text": spec.trigger_text,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return True


def load_revision(store_dir: Path | str, rev: str) -> dict | None:
    """저장된 판본. 없으면 None. `policy_text` 를 다시 해싱해 무결성을 확인한다."""
    path = snapshot_path(store_dir, rev)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    text = data.get("policy_text", "")
    data["intact"] = revision(text) == rev
    return data


def list_revisions(store_dir: Path | str) -> list[dict]:
    """저장된 판본을 first_seen 오름차순으로. 없으면 []."""
    store = Path(store_dir)
    if not store.exists():
        return []
    out = []
    for path in sorted(store.glob("*.json")):
        rev = load_revision(store, path.stem)
        if rev:
            out.append(rev)
    return sorted(out, key=lambda r: (r.get("first_seen") or "", r["rev"]))
