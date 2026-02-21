# Server API Spec (v1)

## POST /case/generate
- 1주차에서는 더미로 고정 사건(case_001)을 반환해도 됨.
- 2주차부터 실제 생성 로직/LLM 연결 가능.

### Response (200)
{
  "case": { ...case_interrogation_01.json 내용... }
}

---

## POST /interrogation/qna

### Request
{
  "case_id": "case_001",
  "user_text": "어제 밤 어디 있었지?"
}

### Response (200)
{
  "user_text": "어제 밤 어디 있었지?",
  "suspect_text": "집에 있었습니다.",
  "pressure_delta": 0.0,
  "confession_probability": 0.15,
  "confession_triggered": false,
  "audio_wav_b64": ""
}

### Notes
- 1주차: audio_wav_b64는 빈 문자열("")이어도 됨.
- 필드명/구조는 1주차 이후 변경 금지.
