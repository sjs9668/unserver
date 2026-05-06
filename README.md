# 음성 기반 AI 심문 게임 서버

Unreal Engine 클라이언트와 연동되는 FastAPI 기반 심문 추론 게임 서버입니다.  
플레이어가 형사가 되어 사전 제작된 사건을 선택하고, NPC 성격을 설정한 뒤 음성 또는 텍스트 질문으로 용의자를 심문하는 구조입니다.

서버는 질문 분석, 모순 판단, 압박 계산, 심리 상태 갱신, 용의자 답변 생성, 최종 판단 리포트 생성을 담당합니다.

## 주요 구현

| 영역 | 구현 내용 |
| --- | --- |
| 사건 선택 | `cases/prebuilt`의 사전 제작 사건 JSON을 로드하고, 클라이언트에 사건 브리핑을 제공합니다. |
| 음성 입력 | OpenAI STT를 사용해 음성 질문을 텍스트로 변환합니다. |
| 질문 분석 | 질문의 의도, 대상 슬롯, 언급 증거, 압박 수준을 구조화합니다. |
| 모순 판단 | 사건 데이터와 충돌하는 하드 모순, 대화 중 진술 변화인 소프트 모순을 분리해 감지합니다. |
| 압박 계산 | 증거 제시, 모순 발생, 질문 강도, 반복 질문을 반영해 턴별 압박과 누적 압박을 계산합니다. |
| 붕괴 확률 | 누적 압박을 Sigmoid 함수에 통과시켜 `breakdown_probability`를 계산합니다. |
| 심리 모델 | HEXACO 성격과 PAD 감정 상태를 사용해 말투, 협조도, 스트레스, 붕괴 저항을 조절합니다. |
| 답변 생성 | NPC 성격, 현재 감정, 붕괴 단계, 질문 맥락을 프롬프트에 반영해 용의자 답변을 생성합니다. |
| 음성 출력 | 생성된 용의자 답변을 TTS로 변환해 `audio_wav_b64`로 반환합니다. |
| 최종 판단 | 10턴 종료 후 플레이어 판단, 실제 정체 공개, 심리 리포트를 제공합니다. |

## 심문 흐름

```text
Unreal Client
  -> /case/generate
  -> /interrogation/setup
  -> /interrogation/qna

/interrogation/qna 내부 흐름
  1. 음성 파일이 있으면 STT 수행
  2. 사건 데이터와 이전 진행 상태 로드
  3. 질문 분석
  4. 질문 카테고리 분류
  5. 용의자 답변 생성
  6. 하드 모순 판단
  7. 소프트 모순 판단
  8. 압박, 스트레스, 협조도, PAD, 붕괴 확률 계산
  9. 진술 기록 저장
  10. TTS 음성 포함 JSON 응답 반환
```

## 프로젝트 구조

```text
server/
  app/
    main.py                    # FastAPI 앱 생성 및 router 등록
    config.py                  # 환경변수, 모델명, 저장 경로 설정
    routers/
      health.py                # /health
      cases.py                 # /case/generate
      interrogation.py         # /interrogation/* 라우트
    services/
      case_service.py          # 사건 로딩, 정규화, 브리핑 생성
      interrogation_service.py # 심문 API orchestration
      openai_service.py        # STT, LLM, TTS 호출
      scoring_service.py       # 압박, 붕괴 확률, PAD, HEXACO 계산
      contradiction_service.py # 하드/소프트 모순 판단
    storage/
      case_store.py            # 사건 캐시와 런타임 저장
      progress_store.py        # 심문 진행 상태 저장
    schemas/
      case.py                  # 사건/성격 관련 정규화
      interrogation.py         # 진행 상태, 판단, 리포트 정규화
    utils/
      json.py                  # JSON 파일/문자열 유틸
      text.py                  # 텍스트 정규화 유틸
  cases/
    prebuilt/                  # 사전 제작 사건 JSON
  docs/
    api_spec.md
    case_schema.md
  requirements.txt
```

## API

| Method | Endpoint | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/case/generate` | 사전 제작 사건 후보 반환 |
| `POST` | `/interrogation/setup` | 선택한 NPC HEXACO 성격값 저장 |
| `POST` | `/interrogation/qna` | 질문 처리, 답변 생성, 심문 상태 갱신 |
| `POST` | `/interrogation/judgment/submit` | 플레이어 최종 판단 제출 |
| `POST` | `/interrogation/report` | 진술 기록과 심리 리포트 조회 |
| `POST` | `/interrogation/result/reveal` | 실제 사건 결과 공개 |
| `POST` | `/interrogation/debug_confess` | 디버그용 강제 붕괴/노출 응답 |

자세한 요청/응답 예시는 [docs/api_spec.md](docs/api_spec.md)를 참고하세요.

## 심리 모델

### HEXACO

NPC의 고정 성향입니다. 심문 시작 전에 클라이언트가 6개 값을 서버로 전달합니다.

```text
honesty_humility
emotionality
extraversion
agreeableness
conscientiousness
openness_to_experience
```

이 값은 압박 민감도, 협조도, 붕괴 저항, 답변 길이, 말투 지시문에 반영됩니다.

### PAD

심문 중 변화하는 현재 감정 상태입니다.

```text
pleasure
arousal
dominance
```

압박, 증거 제시, 모순 발생, 반복 질문에 따라 매 턴 갱신되며 NPC 답변 프롬프트에 함께 전달됩니다.

## 압박과 붕괴 계산

서버는 매 턴 다음 요소를 반영해 압박을 계산합니다.

```text
질문 강도
새로운 증거 언급
반복 증거 언급
하드 모순
소프트 모순
SUE 기반 증거 충격
반복 질문 감쇠
```

누적 압박은 `InterrogationCore.calculate_breakdown_probability()`에서 Sigmoid 함수로 변환됩니다.

```python
p_breakdown = 1.0 / (1.0 + math.exp(-sigmoid_input))
```

초반에는 변화가 작고, 누적 압박이 임계점을 넘으면 붕괴 확률이 빠르게 증가하도록 설계했습니다.

## 실행 방법

### 1. 환경변수 설정

루트에 `.env` 파일을 만들고 OpenAI API 키를 설정합니다.

```env
OPENAI_API_KEY=your_openai_api_key
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 서버 실행

```bash
uvicorn app.main:app --host=0.0.0.0 --port=8000
```

또는:

```bash
python -m app.main
```

### 4. 상태 확인

```bash
curl http://localhost:8000/health
```

정상 응답:

```json
{"ok": true}
```

## 배포

Cloudtype GitHub Actions로 배포합니다.

```yaml
start: uvicorn app.main:app --host=0.0.0.0 --port=8000
healthz: /health
```

배포 환경에는 다음 secret이 필요합니다.

```text
CLOUDTYPE_API_KEY
OPENAI_API_KEY
```

## Unreal 클라이언트 연동

리팩터링 후에도 클라이언트 API 표면은 유지됩니다.

Unreal 클라이언트는 기존과 동일하게 서버 URL과 endpoint를 호출하면 됩니다.  
변경된 것은 서버 내부 모듈 구조와 실행 엔트리포인트입니다.

## 참고 문서

- [API Spec](docs/api_spec.md)
- [Case Schema Guide](docs/case_schema.md)
