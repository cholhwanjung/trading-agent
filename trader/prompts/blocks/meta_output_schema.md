반드시 아래 JSON 만 출력한다 (설명 문장 금지):
{
  "weights": {"CRYPTO": <float>, "KR": <float>, "US": <float>},
  "confidence": <0~1 float>,
  "cited": ["<입력에 있는 signal id 또는 news id 또는 lesson id>", ...],
  "rationale": "<한국어 3문장 이내>",
  "horizon": "4w"
}
