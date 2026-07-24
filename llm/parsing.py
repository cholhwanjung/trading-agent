"""LLM 텍스트 응답에서 JSON 페이로드 추출 — 코드펜스·전후 산문 관용 파싱.

json_mode 를 강제할 수 없는 프로바이더가 있어 모든 호출부가 같은 관용 파싱을
공유한다. 실패 시맨틱(예외/폴백/빈 값)은 호출부가 결정한다 — 여기서는 None 반환.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | list | None:
    """첫 JSON 객체/배열을 파싱해 반환. 실패 시 None.

    코드펜스를 걷어낸 전체 텍스트 파싱을 먼저 시도하고, 실패하면 처음 나오는
    {...} 또는 [...] 구간을 파싱한다. dict/list 가 아닌 값(숫자 등)도 None.
    """
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", stripped, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, (dict, list)) else None
