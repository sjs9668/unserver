# Server API Spec

## POST /case/generate
- Returns 3 prebuilt cases from `cases/prebuilt`.
- Includes both full case payloads and public briefing payloads for case selection UI.

### Response (200)
```json
{
  "source": "prebuilt",
  "cases": [{ "...full case json..." }],
  "case": { "...first full case..." },
  "briefing_cases": [{ "...public briefing..." }],
  "briefing_case": { "...first public briefing..." }
}
```

## POST /interrogation/setup
- Stores the player-selected Big Five values before interrogation starts.
- `personality_json` must include all 5 traits:
  - `openness`
  - `conscientiousness`
  - `extraversion`
  - `agreeableness`
  - `neuroticism`

### Request
```json
{
  "case_id": "case_2026_0418_parking",
  "personality_json": {
    "openness": 0.3,
    "conscientiousness": 0.8,
    "extraversion": 0.2,
    "agreeableness": 0.4,
    "neuroticism": 0.7
  }
}
```

## POST /interrogation/qna

### Request
```json
{
  "case_id": "case_2026_0418_parking",
  "user_text": "그 시간에 어디 있었습니까?",
  "personality_json": {
    "openness": 0.3,
    "conscientiousness": 0.8,
    "extraversion": 0.2,
    "agreeableness": 0.4,
    "neuroticism": 0.7
  }
}
```

### Response (200)
```json
{
  "user_text": "그 시간에 어디 있었습니까?",
  "question_category": "basic_fact",
  "question_category_label": "기본 사실 질문",
  "suspect_text": "그 시간에는 건물 안에 있었습니다.",
  "pressure_delta": 0.08,
  "breakdown_probability": 0.34,
  "core_fact_exposed": false,
  "statement_collapse_stage": 2,
  "judgment_ready": false,
  "audio_wav_b64": ""
}
```

## POST /interrogation/judgment/submit
- Saves the player's final judgment.

## POST /interrogation/report
- Returns the structured statement log and final psychological report.

## POST /interrogation/result/reveal
- Reveals the actual case result after judgment.
