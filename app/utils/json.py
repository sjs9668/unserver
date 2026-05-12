"""JSON 읽기/쓰기 보조 함수."""

import json
import uuid
from pathlib import Path
from typing import Any


def safe_json_loads(s: str, default: Any):
    """JSON 문자열 파싱 실패 시 예외 대신 기본값을 반환한다."""
    try:
        return json.loads(s)
    except Exception:
        return default


def read_json_file(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write_json(path: Path, payload: Any) -> None:
    """임시 파일에 먼저 쓴 뒤 교체해 저장 중 파일이 깨지는 상황을 줄인다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)
