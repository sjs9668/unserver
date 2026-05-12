"""서버 실행 환경과 외부 API 모델 설정.

배포 환경과 로컬 시연 환경 모두에서 같은 코드를 사용할 수 있도록
.env 값을 먼저 읽고, 사건/진행 상태 저장 경로와 OpenAI 모델명을 한곳에서 관리한다.
"""

import os
from pathlib import Path
from typing import Union

# 프로젝트 기준 경로와 런타임 저장 경로.
ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_STORE_DIR = ROOT_DIR / "runtime_store"
CASE_STORE_DIR = RUNTIME_STORE_DIR / "cases"
PREBUILT_CASE_DIR = ROOT_DIR / "cases" / "prebuilt"


def load_env(path: Union[Path, str] = ROOT_DIR / ".env") -> None:
    """간단한 KEY=VALUE 형식의 .env 파일을 읽어 환경 변수로 등록한다."""
    env_path = Path(path)
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


load_env()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env (OPENAI_API_KEY=...)")

# 음성 입력(STT), 답변 생성(LLM), 음성 출력(TTS)에 사용할 모델명.
STT_PRIMARY = "gpt-4o-mini-transcribe"
STT_FALLBACK = "whisper-1"
LLM_MODEL = "gpt-5.4-2026-03-05"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "verse"
