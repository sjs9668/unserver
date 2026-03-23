# 사건 데이터 스키마 가이드

본 문서는 `cases/prebuilt/*.json` 형태의 사전 제작 사건 파일이
서버에서 어떻게 사용되는지 설명한다.

## 핵심 구조

사건 파일은 크게 두 층으로 나뉜다.

1. 브리핑 / UI용 데이터
- `selection_card`
- `documents`
- `suspect_profile.default_personality`
- `suspect_profile.mental_state`

2. 심문 엔진용 데이터
- `overview`
- `suspect`
- `false_statement`
- `truth_slots`
- `evidences`
- `contradictions`

## suspect_profile

### default_personality
- 사건 파일에 저장된 기본 Big Five 값
- UI 초기값이나 디자이너 기준값으로 사용
- 실제 심문 계산은 이 값이 아니라 플레이어가 고른 `selected_personality` 기준으로 진행

### mental_state
- 사건 시작 시점의 PAD 기본값
- `pleasure`
- `arousal`
- `dominance`

## evidences

각 증거는 다음 필드를 가진다.
- `id`
- `name`
- `description`
- `aliases`

`aliases`는 질문 분석과 증거 언급 탐지에 사용된다.

## contradictions

각 모순 데이터는 다음 필드를 가진다.
- `id`
- `description`
- `related_evidence`
- `slot`
- `truth_value`
- `contradiction_type`

이 데이터는 하드 모순 판정의 기준이 된다.

## 런타임 성향 적용

사건 파일의 `default_personality`는 고정 데이터다.
실제 심문 시작 전에는 언리얼이 Big Five 5개 값을 서버로 보내고,
서버는 그 값을 `selected_personality`로 저장해 심문 엔진 계산에 사용한다.
