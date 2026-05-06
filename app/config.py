import os
from pathlib import Path
from typing import Union

ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_STORE_DIR = ROOT_DIR / "runtime_store"
CASE_STORE_DIR = RUNTIME_STORE_DIR / "cases"
PREBUILT_CASE_DIR = ROOT_DIR / "cases" / "prebuilt"


def load_env(path: Union[Path, str] = ROOT_DIR / ".env") -> None:
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

STT_PRIMARY = "gpt-4o-mini-transcribe"
STT_FALLBACK = "whisper-1"
LLM_MODEL = "gpt-5.4-2026-03-05"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "verse"
