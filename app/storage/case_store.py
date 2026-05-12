"""사건 데이터 캐시와 런타임 저장소.

클라이언트가 선택한 사건을 서버 메모리와 runtime_store/cases에 저장해,
서버 프로세스 안에서 이어지는 심문 요청이 같은 사건 데이터를 참조할 수 있게 한다.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import CASE_STORE_DIR, PREBUILT_CASE_DIR
from app.utils.json import atomic_write_json, read_json_file
from app.utils.text import norm

CASE_CACHE: Dict[str, Dict[str, Any]] = {}

# 런타임 저장 폴더와 사전 제작 사건 폴더를 서버 시작 시 보장한다.
for _store_dir in (CASE_STORE_DIR, PREBUILT_CASE_DIR):
    _store_dir.mkdir(parents=True, exist_ok=True)


def store_key(value: str) -> str:
    """사용자 입력 case_id를 안전한 파일명으로 바꾼다."""
    raw = (value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")[:80] or "item"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{safe}-{digest}"


def case_store_path(case_id: str) -> Path:
    return CASE_STORE_DIR / f"{store_key(case_id)}.json"


def persist_case(case_data: Dict[str, Any]) -> None:
    """정규화된 사건 데이터를 디스크에 저장한다."""
    case_id = norm(case_data.get("case_id", ""))
    if not case_id:
        return
    atomic_write_json(case_store_path(case_id), case_data)


def load_case(case_id: str) -> Optional[Dict[str, Any]]:
    """메모리 캐시를 먼저 보고, 없으면 런타임 저장소에서 사건을 복원한다."""
    cid = norm(case_id)
    if not cid:
        return None

    cached = CASE_CACHE.get(cid)
    if isinstance(cached, dict):
        return cached

    stored = read_json_file(case_store_path(cid), None)
    if not isinstance(stored, dict):
        return None
    if norm(stored.get("case_id", "")) != cid:
        return None

    CASE_CACHE[cid] = stored
    print(f"[CaseStore] Rehydrated case_id='{cid}' from disk")
    return stored
