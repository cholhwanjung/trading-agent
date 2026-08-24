반드시 아래 JSON 만 출력한다 (설명 문장 금지):
{{
  "allocation": {{"<symbol>": <float>, ..., "CASH": <float>}},
  "rationale": "<핵심 근거 2~3문장, 한국어>",
  "cited_signal_ids": ["<참고한 feature 이름>", ...],
  "cited_memory_ids": [],
  "scenario": {{
    "expected": "<예상 시나리오 1문장>",
    "invalidation": "<이 결정이 틀렸다고 판정할 구체적 조건 1문장>"
  }}
}}
