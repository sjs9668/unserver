import os
import json
import base64
import traceback
import re
import hashlib
import random
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import uuid
from collections import deque

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from openai import OpenAI, BadRequestError

# =========================================================
# ENV
# =========================================================
def load_env(path: str = ".env") -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env (OPENAI_API_KEY=...)")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
# MODEL CONFIG
# =========================================================
STT_PRIMARY = "gpt-4o-mini-transcribe"
STT_FALLBACK = "whisper-1"
LLM_MODEL = "gpt-5.2"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "verse"

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True}

# =========================================================
# IN-MEMORY STORE (서버 재시작하면 초기화)
# =========================================================
CASE_CACHE: Dict[str, Dict[str, Any]] = {}
INTERROGATION_PROGRESS_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
RUNTIME_STORE_DIR = Path(__file__).parent / "runtime_store"
CASE_STORE_DIR = RUNTIME_STORE_DIR / "cases"
PROGRESS_STORE_DIR = RUNTIME_STORE_DIR / "interrogation_progress"

for _store_dir in (CASE_STORE_DIR, PROGRESS_STORE_DIR):
    _store_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# EXAMPLE CASE (템플릿 예시로만 사용)
# =========================================================
EXAMPLE_CASE_PATH = Path(__file__).parent / "cases" / "case_interrogation_01.json"
EXAMPLE_CASE_TEXT = ""

if EXAMPLE_CASE_PATH.exists():
    EXAMPLE_CASE_TEXT = EXAMPLE_CASE_PATH.read_text(encoding="utf-8").strip()
else:
    EXAMPLE_CASE_TEXT = ""

CASE_VARIANT_BLUEPRINTS: List[Dict[str, Any]] = [
    {
        "label": "절도 폭행",
        "type_hint": "절도 후 폭행",
        "place_hint": "원룸 건물 복도, 오피스텔 주차장, 편의점 창고",
        "motive_hint": "생활고, 빚, 들킬까 봐 충동적으로 폭행",
        "allow_fire": False,
    },
    {
        "label": "독살 미수",
        "type_hint": "독살 미수",
        "place_hint": "식당 주방, 사무실 탕비실, 가족 식사 자리",
        "motive_hint": "유산, 원한, 관계 파탄에 따른 계획적 범행",
        "allow_fire": False,
    },
    {
        "label": "유괴 감금",
        "type_hint": "유괴 및 감금",
        "place_hint": "모텔, 외곽 창고, 지하 작업실",
        "motive_hint": "금전 요구, 보복, 비밀을 막기 위한 통제",
        "allow_fire": False,
    },
    {
        "label": "횡령 은폐",
        "type_hint": "횡령 및 증거 은닉",
        "place_hint": "중소기업 사무실, 회계팀, 창고 사무동",
        "motive_hint": "투자 실패, 도박 빚, 장부 조작 은폐",
        "allow_fire": False,
    },
    {
        "label": "뺑소니 은폐",
        "type_hint": "뺑소니 및 은폐",
        "place_hint": "골목길, 공장 진입로, 새벽 도로",
        "motive_hint": "음주 사실 은폐, 면허 문제, 두려움",
        "allow_fire": False,
    },
    {
        "label": "협박 갈취",
        "type_hint": "협박 및 갈취",
        "place_hint": "노래방, 학원 사무실, 골목 흡연 구역",
        "motive_hint": "돈 요구, 약점 이용, 지위 관계 악용",
        "allow_fire": False,
    },
    {
        "label": "밀수 운반",
        "type_hint": "밀수 및 불법 운반",
        "place_hint": "항구 창고, 냉동 탑차, 물류센터 하역장",
        "motive_hint": "고수익 제안, 빚 상환, 조직 압박",
        "allow_fire": False,
    },
    {
        "label": "산업 스파이",
        "type_hint": "산업기밀 유출",
        "place_hint": "연구실, 스타트업 사무실, 공장 설비실",
        "motive_hint": "이직 대가, 경쟁사 거래, 승진 불만",
        "allow_fire": False,
    },
    {
        "label": "스토킹 침입",
        "type_hint": "스토킹 및 주거침입",
        "place_hint": "빌라 복도, 피해자 집 앞, 지하주차장",
        "motive_hint": "집착, 거절에 대한 분노, 관계 망상",
        "allow_fire": False,
    },
    {
        "label": "보험 사기",
        "type_hint": "보험 사기",
        "place_hint": "병원, 공사 현장, 차량 사고 현장",
        "motive_hint": "보험금 편취, 채무 해결, 공모 은폐",
        "allow_fire": False,
    },
    {
        "label": "살인",
        "type_hint": "살인",
        "place_hint": "아파트 내부, 외딴 골목, 공장 사무실, 창고",
        "motive_hint": "원한, 금전 분쟁, 관계 파탄, 충동적 격분",
        "allow_fire": False,
    },
    {
        "label": "방화",
        "type_hint": "방화",
        "place_hint": "상가 창고, 폐건물, 외곽 창고",
        "motive_hint": "보험금, 보복, 증거 인멸",
        "allow_fire": True,
    },
    {
        "label": "살인 교사",
        "type_hint": "살인 교사 미수",
        "place_hint": "사무실, 가족 모임, 거래 현장",
        "motive_hint": "유산, 사업권 분쟁, 관계 청산",
        "allow_fire": False,
    },
]
RECENT_CASE_TYPE_HISTORY: deque[str] = deque(maxlen=5)
FIRE_CASE_MARKERS = ("방화", "화재", "불")

# =========================================================
# UTILS
# =========================================================
def safe_json_loads(s: str, default: Any):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)

def store_key(value: str) -> str:
    raw = (value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")[:80] or "item"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{safe}-{digest}"

def case_store_path(case_id: str) -> Path:
    return CASE_STORE_DIR / f"{store_key(case_id)}.json"

def progress_store_path(case_id: str) -> Path:
    return PROGRESS_STORE_DIR / f"{store_key(case_id)}.json"

def persist_case(case_data: Dict[str, Any]) -> None:
    case_id = norm(case_data.get("case_id", ""))
    if not case_id:
        return
    atomic_write_json(case_store_path(case_id), case_data)

def load_case(case_id: str) -> Optional[Dict[str, Any]]:
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

def normalize_progress_cache(blob: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(blob, dict):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for history_key, state in blob.items():
        if not isinstance(history_key, str) or not isinstance(state, dict):
            continue
        normalized[history_key] = {
            "confession_probability": clamp01(state.get("confession_probability", 0.0)),
            "referenced_evidence_ids": sorted(
                {str(eid).strip() for eid in (state.get("referenced_evidence_ids", []) or []) if str(eid).strip()}
            ),
            "established_contradiction_ids": sorted(
                {
                    str(cid).strip()
                    for cid in (state.get("established_contradiction_ids", []) or [])
                    if str(cid).strip()
                }
            ),
        }
    return normalized

def load_progress_cache(case_id: str) -> Dict[str, Dict[str, Any]]:
    cid = norm(case_id)
    if not cid:
        return {}

    cached = INTERROGATION_PROGRESS_CACHE.get(cid)
    if isinstance(cached, dict):
        return cached

    restored = normalize_progress_cache(read_json_file(progress_store_path(cid), {}))
    if restored:
        INTERROGATION_PROGRESS_CACHE[cid] = restored
        print(f"[ProgressStore] Rehydrated case_id='{cid}' with {len(restored)} states")
    return restored

def persist_progress_cache(case_id: str) -> None:
    cid = norm(case_id)
    if not cid:
        return

    progress_by_history = normalize_progress_cache(INTERROGATION_PROGRESS_CACHE.get(cid, {}))
    if not progress_by_history:
        return
    atomic_write_json(progress_store_path(cid), progress_by_history)

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_for_match(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"[\W_]+", "", s, flags=re.UNICODE)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def trim_to_1_3_sentences(text: str) -> str:
    t = norm(text)
    if not t:
        return t
    parts = re.split(r"(?<=[\.!\?。！？])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 3:
        return t
    return " ".join(parts[:3])

def is_too_ambiguous(user_text: str) -> bool:
    t = norm(user_text)
    if len(t) < 3:
        return True
    if t in {"뭐야", "뭐", "왜", "응", "어", "아니", "그래서", "그럼"}:
        return True
    return False

def detect_repeat(history: List[Dict[str, Any]], user_text: str) -> bool:
    t = norm(user_text)
    if not history:
        return False
    recent = [norm(h.get("user_text", "")) for h in history[-2:] if isinstance(h, dict)]
    return any(t == r and t for r in recent)

def extract_case_keywords(case_data: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    압박도 계산용 키워드:
    - evidences: id/name
    - contradictions: id/description
    """
    if not case_data:
        return [], []

    ev_keywords: List[str] = []
    cd_keywords: List[str] = []

    evidences = case_data.get("evidences", []) or []
    for e in evidences:
        if isinstance(e, dict):
            if e.get("id"):
                ev_keywords.append(str(e["id"]))
            if e.get("name"):
                ev_keywords.append(str(e["name"]))

    contradictions = case_data.get("contradictions", []) or []
    for c in contradictions:
        if isinstance(c, dict):
            if c.get("id"):
                cd_keywords.append(str(c["id"]))
            if c.get("description"):
                cd_keywords.append(str(c["description"]))

    def uniq(xs: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in xs:
            x = (x or "").strip()
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return uniq(ev_keywords), uniq(cd_keywords)

PRESSURE_LEVEL_BONUS = {
    "none": 0.0,
    "low": 0.005,
    "medium": 0.015,
    "high": 0.03,
}

CONFESSION_LEVEL_BONUS = {
    "none": 0.0,
    "low": 0.0,
    "medium": 0.01,
    "high": 0.02,
}

NEW_CONTRADICTION_PRESSURE_DELTA = 0.10
REPEATED_CONTRADICTION_PRESSURE_DELTA = 0.05

DIALOGUE_CONTRADICTION_PRESSURE_BONUS = {
    "detective_highlighted": {
        "none": 0.0,
        "low": 0.02,
        "medium": 0.05,
        "high": 0.07,
    },
    "suspect_self_contradicted": {
        "none": 0.0,
        "low": 0.05,
        "medium": 0.08,
        "high": 0.10,
    },
}

DIALOGUE_CONTRADICTION_CONFESSION_BONUS = {
    "detective_highlighted": {
        "none": 0.0,
        "low": 0.01,
        "medium": 0.03,
        "high": 0.05,
    },
    "suspect_self_contradicted": {
        "none": 0.0,
        "low": 0.02,
        "medium": 0.05,
        "high": 0.08,
    },
}

INTERROGATION_EVAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "before_current_turn": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "referenced_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "established_contradiction_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["referenced_evidence_ids", "established_contradiction_ids"],
        },
        "current_turn": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "referenced_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "established_contradiction_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "pressure_level": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high"],
                },
            },
            "required": [
                "referenced_evidence_ids",
                "established_contradiction_ids",
                "pressure_level",
            ],
        },
        "after_current_turn": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "referenced_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "established_contradiction_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["referenced_evidence_ids", "established_contradiction_ids"],
        },
        "reason": {"type": "string"},
    },
    "required": [
        "before_current_turn",
        "current_turn",
        "after_current_turn",
        "reason",
    ],
}

DIALOGUE_CONTRADICTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "detective_highlighted": {"type": "boolean"},
        "suspect_self_contradicted": {"type": "boolean"},
        "severity": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
        },
        "prior_claim": {"type": "string"},
        "current_claim": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "detective_highlighted",
        "suspect_self_contradicted",
        "severity",
        "prior_claim",
        "current_claim",
        "reason",
    ],
}

SUSPECT_REPLY_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answers_question_directly": {"type": "boolean"},
        "introduces_unrelated_details": {"type": "boolean"},
        "evades_core_issue": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "answers_question_directly",
        "introduces_unrelated_details",
        "evades_core_issue",
        "reason",
    ],
}

def uniq_strings(values: List[str]) -> List[str]:
    out = []
    seen = set()
    for value in values:
        value = (value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out

def _case_evidences(case_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not case_data:
        return []
    return [e for e in (case_data.get("evidences", []) or []) if isinstance(e, dict)]

def _case_contradictions(case_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not case_data:
        return []
    return [c for c in (case_data.get("contradictions", []) or []) if isinstance(c, dict)]

def _dialogue_lines(history: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for h in history:
        if not isinstance(h, dict):
            continue
        user_line = norm(h.get("user_text", ""))
        suspect_line = norm(h.get("suspect_text", ""))
        if user_line:
            lines.append(f"형사: {user_line}")
        if suspect_line:
            lines.append(f"피의자: {suspect_line}")
    return lines

def _dialogue_turn_payload(history: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for entry in history[-limit:]:
        if not isinstance(entry, dict):
            continue
        detective_text = norm(entry.get("user_text", ""))
        suspect_text = norm(entry.get("suspect_text", ""))
        if not detective_text and not suspect_text:
            continue
        out.append(
            {
                "detective_text": detective_text,
                "suspect_text": suspect_text,
            }
        )
    return out

def _sanitize_id_list(values: Any, allowed_ids: set) -> List[str]:
    if not isinstance(values, list):
        return []

    out = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if text and text in allowed_ids and text not in seen:
            seen.add(text)
            out.append(text)
    return out

def _empty_interrogation_signal(reason: str = "") -> Dict[str, Any]:
    return {
        "before_current_turn": {
            "referenced_evidence_ids": [],
            "established_contradiction_ids": [],
        },
        "current_turn": {
            "referenced_evidence_ids": [],
            "established_contradiction_ids": [],
            "pressure_level": "none",
        },
        "after_current_turn": {
            "referenced_evidence_ids": [],
            "established_contradiction_ids": [],
        },
        "reason": reason,
    }

def _empty_dialogue_contradiction_signal(reason: str = "") -> Dict[str, Any]:
    return {
        "detective_highlighted": False,
        "suspect_self_contradicted": False,
        "severity": "none",
        "prior_claim": "",
        "current_claim": "",
        "reason": reason,
    }

def _sanitize_dialogue_contradiction_signal(raw_signal: Any) -> Dict[str, Any]:
    if not isinstance(raw_signal, dict):
        return _empty_dialogue_contradiction_signal("")

    severity = str(raw_signal.get("severity", "none")).strip().lower()
    if severity not in {"none", "low", "medium", "high"}:
        severity = "none"

    return {
        "detective_highlighted": bool(raw_signal.get("detective_highlighted", False)),
        "suspect_self_contradicted": bool(raw_signal.get("suspect_self_contradicted", False)),
        "severity": severity,
        "prior_claim": norm(raw_signal.get("prior_claim", "")),
        "current_claim": norm(raw_signal.get("current_claim", "")),
        "reason": norm(raw_signal.get("reason", "")),
    }

def _sanitize_interrogation_signal(
    case_data: Optional[Dict[str, Any]],
    raw_signal: Dict[str, Any],
) -> Dict[str, Any]:
    valid_evidence_ids = {
        str(e.get("id", "")).strip() for e in _case_evidences(case_data) if e.get("id")
    }
    valid_contradiction_ids = {
        str(c.get("id", "")).strip() for c in _case_contradictions(case_data) if c.get("id")
    }

    def _part(key: str) -> Dict[str, List[str]]:
        part = raw_signal.get(key, {}) if isinstance(raw_signal, dict) else {}
        if not isinstance(part, dict):
            part = {}
        return {
            "referenced_evidence_ids": _sanitize_id_list(
                part.get("referenced_evidence_ids", []),
                valid_evidence_ids,
            ),
            "established_contradiction_ids": _sanitize_id_list(
                part.get("established_contradiction_ids", []),
                valid_contradiction_ids,
            ),
        }

    before = _part("before_current_turn")
    current = _part("current_turn")
    after = _part("after_current_turn")

    after["referenced_evidence_ids"] = uniq_strings(
        before["referenced_evidence_ids"]
        + current["referenced_evidence_ids"]
        + after["referenced_evidence_ids"]
    )
    after["established_contradiction_ids"] = uniq_strings(
        before["established_contradiction_ids"]
        + current["established_contradiction_ids"]
        + after["established_contradiction_ids"]
    )

    pressure_level = "none"
    if isinstance(raw_signal, dict):
        pressure_level = str(
            (raw_signal.get("current_turn", {}) or {}).get("pressure_level", "none")
        ).strip().lower()
    if pressure_level not in PRESSURE_LEVEL_BONUS:
        pressure_level = "none"

    return {
        "before_current_turn": before,
        "current_turn": {
            "referenced_evidence_ids": current["referenced_evidence_ids"],
            "established_contradiction_ids": current["established_contradiction_ids"],
            "pressure_level": pressure_level,
        },
        "after_current_turn": after,
        "reason": norm(raw_signal.get("reason", "")) if isinstance(raw_signal, dict) else "",
    }

def _lexical_evidence_hits(case_data: Optional[Dict[str, Any]], text: str) -> List[str]:
    text_match = norm_for_match(text)
    if not text_match:
        return []

    hits: List[str] = []
    for evidence in _case_evidences(case_data):
        evidence_id = str(evidence.get("id", "")).strip()
        candidates = [evidence_id, str(evidence.get("name", "")).strip()]
        for candidate in candidates:
            candidate_match = norm_for_match(candidate)
            if candidate_match and candidate_match in text_match:
                if evidence_id:
                    hits.append(evidence_id)
                break
    return uniq_strings(hits)

def _fallback_interrogation_signal(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
) -> Dict[str, Any]:
    if not case_data:
        return _empty_interrogation_signal("case unavailable")

    previous_text = " ".join(
        norm(h.get("user_text", "")) for h in history if isinstance(h, dict)
    )
    current_text = norm(user_text)
    after_text = f"{previous_text} {current_text}".strip()

    before_evidence_ids = _lexical_evidence_hits(case_data, previous_text)
    current_evidence_ids = _lexical_evidence_hits(case_data, current_text)
    after_evidence_ids = _lexical_evidence_hits(case_data, after_text)

    pressure_level = "low" if current_evidence_ids else "none"
    if detect_repeat(history, user_text):
        pressure_level = "none"

    return {
        "before_current_turn": {
            "referenced_evidence_ids": before_evidence_ids,
            "established_contradiction_ids": [],
        },
        "current_turn": {
            "referenced_evidence_ids": current_evidence_ids,
            "established_contradiction_ids": [],
            "pressure_level": pressure_level,
        },
        "after_current_turn": {
            "referenced_evidence_ids": after_evidence_ids,
            "established_contradiction_ids": [],
        },
        "reason": "fallback lexical evidence match only",
    }

def llm_evaluate_interrogation(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
) -> Dict[str, Any]:
    if not case_data:
        return _empty_interrogation_signal("case unavailable")

    payload = {
        "false_statement": case_data.get("false_statement", ""),
        "evidences": [
            {
                "id": e.get("id", ""),
                "name": e.get("name", ""),
                "description": e.get("description", ""),
            }
            for e in _case_evidences(case_data)
        ],
        "contradictions": [
            {
                "id": c.get("id", ""),
                "description": c.get("description", ""),
                "related_evidence": c.get("related_evidence", []) or [],
            }
            for c in _case_contradictions(case_data)
        ],
        "history": _dialogue_lines(history[-10:]),
        "current_detective_text": norm(user_text),
    }

    system = (
        "너는 추리 게임의 수사 진행 판정기다.\n"
        "반드시 JSON만 출력한다.\n"
        "판정 규칙:\n"
        "- before_current_turn은 현재 질문을 제외한 기존 대화 기준이다.\n"
        "- current_turn은 이번 형사 발화 한 줄만 보고 판정한다.\n"
        "- after_current_turn은 이번 발화를 반영한 누적 결과다.\n"
        "- 특수문자, 공백, 구두점 차이는 무시하고 같은 표현으로 본다.\n"
        "- exact keyword match만 보지 말고 의미가 같은 요약, 재진술도 허용한다.\n"
        "- evidence는 형사가 증거 이름, id, 또는 증거 핵심 내용을 명시적으로 가리킬 때만 잡는다.\n"
        "- contradiction는 형사가 피의자 진술과 사건 사실/증거의 충돌을 분명히 짚었을 때만 잡는다.\n"
        "- 애매하면 잡지 않는다. 추측 금지.\n"
        "- pressure_level은 none, low, medium, high 중 하나다.\n"
        "  none: 일반 질문, 의미 없는 말, 압박 근거 부족\n"
        "  low: 의심 또는 확인 질문\n"
        "  medium: 증거나 거짓말을 직접 들이대는 압박\n"
        "  high: 모순을 연결해 피의자를 궁지로 모는 압박\n"
        "- id는 제공된 목록에 있는 값만 사용한다.\n"
    )

    try:
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "interrogation_signal",
                    "strict": True,
                    "schema": INTERROGATION_EVAL_SCHEMA,
                }
            },
            max_output_tokens=700,
            store=False,
        )
        raw_signal = safe_json_loads((resp.output_text or "").strip(), {})
        return _sanitize_interrogation_signal(case_data, raw_signal)
    except Exception:
        return _fallback_interrogation_signal(case_data, history, user_text)

def llm_evaluate_dialogue_contradiction(
    history: List[Dict[str, Any]],
    user_text: str,
    suspect_text: str,
) -> Dict[str, Any]:
    current_suspect_text = norm(suspect_text)
    if not current_suspect_text:
        return _empty_dialogue_contradiction_signal("empty suspect reply")

    dialogue_payload = _dialogue_turn_payload(history, limit=8)
    prior_suspect_turns = [turn for turn in dialogue_payload if turn.get("suspect_text")]
    if not prior_suspect_turns:
        return _empty_dialogue_contradiction_signal("no prior suspect statements")

    payload = {
        "history": dialogue_payload,
        "current_detective_text": norm(user_text),
        "current_suspect_text": current_suspect_text,
    }

    system = (
        "You evaluate interrogation dialogue for self-contradictions.\n"
        "Return only JSON.\n"
        "Look for contradictions between the suspect's earlier statements in history and the current turn.\n"
        "detective_highlighted is true only if the detective's latest question clearly points out or strongly implies a contradiction.\n"
        "suspect_self_contradicted is true only if the suspect's new reply directly conflicts with an earlier suspect claim.\n"
        "Do not flag mere elaboration, clarification, or added detail unless both claims cannot both be true.\n"
        "Focus on alibi, time, place, action, presence, and who the suspect met.\n"
        "Severity guide:\n"
        "- none: no contradiction\n"
        "- low: weak or ambiguous mismatch\n"
        "- medium: clear mismatch on one important detail\n"
        "- high: direct alibi, location, or action contradiction\n"
        "prior_claim should summarize the earlier suspect claim that conflicts.\n"
        "current_claim should summarize the current turn's conflicting claim.\n"
    )

    try:
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "dialogue_contradiction",
                    "strict": True,
                    "schema": DIALOGUE_CONTRADICTION_SCHEMA,
                }
            },
            max_output_tokens=250,
            store=False,
        )
        raw_signal = safe_json_loads((resp.output_text or "").strip(), {})
        return _sanitize_dialogue_contradiction_signal(raw_signal)
    except Exception:
        return _empty_dialogue_contradiction_signal("dialogue contradiction eval failed")

def apply_dialogue_contradiction_bonus(
    pressure_delta: float,
    confession_probability: float,
    dialogue_signal: Optional[Dict[str, Any]],
    confession_triggered: bool,
) -> Tuple[float, float]:
    if not isinstance(dialogue_signal, dict):
        return clamp01(pressure_delta), clamp01(confession_probability)

    severity = str(dialogue_signal.get("severity", "none")).strip().lower()
    if severity not in {"none", "low", "medium", "high"}:
        severity = "none"

    pressure_bonus_candidates = [0.0]
    confession_bonus_candidates = [0.0]

    if dialogue_signal.get("detective_highlighted"):
        pressure_bonus_candidates.append(
            DIALOGUE_CONTRADICTION_PRESSURE_BONUS["detective_highlighted"][severity]
        )
        confession_bonus_candidates.append(
            DIALOGUE_CONTRADICTION_CONFESSION_BONUS["detective_highlighted"][severity]
        )

    if dialogue_signal.get("suspect_self_contradicted"):
        pressure_bonus_candidates.append(
            DIALOGUE_CONTRADICTION_PRESSURE_BONUS["suspect_self_contradicted"][severity]
        )
        confession_bonus_candidates.append(
            DIALOGUE_CONTRADICTION_CONFESSION_BONUS["suspect_self_contradicted"][severity]
        )

    pressure_bonus = min(0.10, max(pressure_bonus_candidates))
    confession_bonus = max(confession_bonus_candidates)

    boosted_pressure = clamp01(min(0.20, pressure_delta + pressure_bonus))
    boosted_probability = clamp01(confession_probability + confession_bonus)
    if not confession_triggered and boosted_probability >= 0.85:
        boosted_probability = 0.84

    return boosted_pressure, boosted_probability

def _default_suspect_reply_review(
    reason: str = "",
    answers_question_directly: bool = True,
) -> Dict[str, Any]:
    return {
        "answers_question_directly": bool(answers_question_directly),
        "introduces_unrelated_details": False,
        "evades_core_issue": False,
        "reason": reason,
    }

def _sanitize_suspect_reply_review(raw_signal: Any) -> Dict[str, Any]:
    if not isinstance(raw_signal, dict):
        return _default_suspect_reply_review("")

    return {
        "answers_question_directly": bool(raw_signal.get("answers_question_directly", True)),
        "introduces_unrelated_details": bool(raw_signal.get("introduces_unrelated_details", False)),
        "evades_core_issue": bool(raw_signal.get("evades_core_issue", False)),
        "reason": norm(raw_signal.get("reason", "")),
    }

def llm_review_suspect_reply(user_text: str, suspect_text: str) -> Dict[str, Any]:
    detective_text = norm(user_text)
    reply_text = norm(suspect_text)

    if not reply_text:
        return _default_suspect_reply_review(
            "empty suspect reply",
            answers_question_directly=False,
        )

    payload = {
        "detective_text": detective_text,
        "suspect_reply": reply_text,
    }
    system = (
        "You review whether a suspect's reply directly answers a detective's latest question.\n"
        "Return only JSON.\n"
        "answers_question_directly is true only if the reply clearly addresses the literal question, accusation, or quoted statement.\n"
        "introduces_unrelated_details is true if the reply brings in a new scene, meeting, time, place, person, or backstory that is not needed to answer the latest question.\n"
        "evades_core_issue is true if the reply sidesteps the main point and answers something adjacent instead.\n"
        "A short denial or reinterpretation is acceptable if it directly addresses the question first.\n"
    )

    try:
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "suspect_reply_review",
                    "strict": True,
                    "schema": SUSPECT_REPLY_REVIEW_SCHEMA,
                }
            },
            max_output_tokens=200,
            store=False,
        )
        raw_signal = safe_json_loads((resp.output_text or "").strip(), {})
        return _sanitize_suspect_reply_review(raw_signal)
    except Exception:
        return _default_suspect_reply_review("suspect reply review failed")

def _count_meaningful_turns(history: List[Dict[str, Any]], user_text: str) -> int:
    seen = set()
    count = 0

    for entry in history:
        if not isinstance(entry, dict):
            continue
        text = norm(entry.get("user_text", ""))
        key = norm_for_match(text)
        if not key or is_too_ambiguous(text) or key in seen:
            continue
        seen.add(key)
        count += 1

    text = norm(user_text)
    key = norm_for_match(text)
    if key and not is_too_ambiguous(text) and key not in seen:
        count += 1

    return count

def _normalize_history_for_progress(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "user_text": norm(entry.get("user_text", "")),
                "suspect_text": norm(entry.get("suspect_text", "")),
            }
        )
    return normalized

def _history_progress_key(history: List[Dict[str, Any]]) -> str:
    payload = json.dumps(
        _normalize_history_for_progress(history),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _empty_progress_state() -> Dict[str, Any]:
    return {
        "confession_probability": 0.0,
        "referenced_evidence_ids": [],
        "established_contradiction_ids": [],
    }

def _get_progress_state(case_id: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not case_id:
        return _empty_progress_state()

    progress_by_history = load_progress_cache(case_id)
    state = progress_by_history.get(_history_progress_key(history))
    if not isinstance(state, dict):
        return _empty_progress_state()

    return {
        "confession_probability": clamp01(state.get("confession_probability", 0.0)),
        "referenced_evidence_ids": list(state.get("referenced_evidence_ids", []) or []),
        "established_contradiction_ids": list(
            state.get("established_contradiction_ids", []) or []
        ),
    }

def _store_progress_state(
    case_id: str,
    history: List[Dict[str, Any]],
    confession_probability: float,
    evidence_ids: List[str],
    contradiction_ids: List[str],
) -> None:
    if not case_id:
        return

    progress_by_history = load_progress_cache(case_id)
    progress_by_history[_history_progress_key(history)] = {
        "confession_probability": clamp01(confession_probability),
        "referenced_evidence_ids": sorted(
            {str(eid).strip() for eid in (evidence_ids or []) if str(eid).strip()}
        ),
        "established_contradiction_ids": sorted(
            {str(cid).strip() for cid in (contradiction_ids or []) if str(cid).strip()}
        ),
    }

    while len(progress_by_history) > 64:
        oldest_key = next(iter(progress_by_history))
        progress_by_history.pop(oldest_key, None)

    INTERROGATION_PROGRESS_CACHE[case_id] = progress_by_history
    persist_progress_cache(case_id)

def build_case_context(case_data: Optional[Dict[str, Any]]) -> str:
    """
    LLM에게 줄 컨텍스트(심문 게임용 내부 사건 정보)
    """
    if not case_data:
        return "사건 정보 없음.\n"

    ov = case_data.get("overview", {}) or {}
    suspect = case_data.get("suspect", {}) or {}

    evidences = case_data.get("evidences", []) or []
    contradictions = case_data.get("contradictions", []) or []

    ev_lines = []
    for e in evidences:
        if isinstance(e, dict):
            ev_lines.append(f"- {e.get('name','')} : {e.get('description','')}".strip())
        else:
            ev_lines.append(f"- {str(e)}")

    cd_lines = []
    for c in contradictions:
        if isinstance(c, dict):
            rel = c.get("related_evidence", [])
            rel_s = f" (related:{rel})" if rel else ""
            cd_lines.append(f"- {c.get('description','')}{rel_s}".strip())
        else:
            cd_lines.append(f"- {str(c)}")

    return (
        "=== 사건 정보(내부) ===\n"
        f"[개요] 시간:{ov.get('time','')} 장소:{ov.get('place','')} 유형:{ov.get('type','')}\n"
        f"[동기] {case_data.get('motive','')}\n"
        f"[범행 흐름(사실)] {case_data.get('crime_flow','')}\n"
        f"[용의자] 이름:{suspect.get('name','')} 나이:{suspect.get('age','')} 직업:{suspect.get('job','')} 관계:{suspect.get('relation','')}\n"
        f"[기본 거짓 진술] {case_data.get('false_statement','')}\n"
        "[증거 목록]\n" + ("\n".join(ev_lines) if ev_lines else "- 없음") + "\n"
        "[모순점 목록]\n" + ("\n".join(cd_lines) if cd_lines else "- 없음") + "\n"
        "======================\n"
    )

def calc_pressure_and_prob(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    interrogation_signal: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """
    2주차 초기 버전 + 3주차 안정화(반복 질문 반감)
    - 증거 언급: +0.20 * hits
    - 모순 찌름: +0.35 * hits
    - 기본 확률: 0.12 + min(0.30, 0.03 * 질문수)
    - 최종 확률 = base + pressure, 0~1 클램프
    """
    ev_kws, cd_kws = extract_case_keywords(case_data)

    all_text = user_text + " " + " ".join(
        [(h.get("user_text", "") or "") for h in history if isinstance(h, dict)]
    )
    all_text_match = norm_for_match(all_text)

    ev_hits = 0
    cd_hits = 0

    for kw in ev_kws:
        kw_match = norm_for_match(kw)
        if kw_match and kw_match in all_text_match:
            ev_hits += 1
    for kw in cd_kws:
        kw_match = norm_for_match(kw)
        if kw_match and kw_match in all_text_match:
            cd_hits += 1

    pressure = 0.0
    pressure += 0.20 * ev_hits
    pressure += 0.35 * cd_hits
    pressure = clamp01(pressure)

    base = 0.12 + min(0.30, 0.03 * len(history))
    prob = clamp01(base + pressure)

    # 반복 질문은 압박 반감(치팅 방지)
    if detect_repeat(history, user_text):
        pressure = clamp01(pressure * 0.5)
        prob = clamp01(base + pressure)

    return pressure, prob

def calc_pressure_and_prob_v2(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    interrogation_signal: Optional[Dict[str, Any]] = None,
    prior_progress: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, List[str], List[str]]:
    if not case_data:
        return 0.0, 0.0, [], []

    signal = interrogation_signal or llm_evaluate_interrogation(case_data, history, user_text)
    prior_progress = prior_progress or _empty_progress_state()

    prior_confession_probability = clamp01(
        prior_progress.get("confession_probability", 0.0)
    )
    prior_evidence_ids = {
        str(eid).strip()
        for eid in (prior_progress.get("referenced_evidence_ids", []) or [])
        if str(eid).strip()
    }
    prior_contradiction_ids = {
        str(cid).strip()
        for cid in (prior_progress.get("established_contradiction_ids", []) or [])
        if str(cid).strip()
    }

    current_evidence_ids = set(signal["current_turn"]["referenced_evidence_ids"])
    after_evidence_ids = set(signal["after_current_turn"]["referenced_evidence_ids"])

    current_contradiction_ids = set(
        signal["current_turn"]["established_contradiction_ids"]
    )
    after_contradiction_ids = set(
        signal["after_current_turn"]["established_contradiction_ids"]
    )

    cumulative_evidence_ids = prior_evidence_ids | after_evidence_ids
    cumulative_contradiction_ids = prior_contradiction_ids | after_contradiction_ids

    new_evidence_ids = cumulative_evidence_ids - prior_evidence_ids
    repeated_evidence_ids = current_evidence_ids & prior_evidence_ids
    new_contradiction_ids = cumulative_contradiction_ids - prior_contradiction_ids
    repeated_contradiction_ids = current_contradiction_ids & prior_contradiction_ids

    pressure_level = signal["current_turn"].get("pressure_level", "none")
    pressure_delta = (
        0.04 * len(new_evidence_ids)
        + 0.01 * len(repeated_evidence_ids)
        + NEW_CONTRADICTION_PRESSURE_DELTA * len(new_contradiction_ids)
        + REPEATED_CONTRADICTION_PRESSURE_DELTA * len(repeated_contradiction_ids)
        + PRESSURE_LEVEL_BONUS.get(pressure_level, 0.0)
    )

    if not user_text or is_too_ambiguous(user_text):
        pressure_delta = 0.0
    elif detect_repeat(history, user_text):
        pressure_delta *= 0.35

    pressure_delta = clamp01(min(0.20, pressure_delta))

    meaningful_turns = _count_meaningful_turns(history, user_text)
    computed_probability = (
        0.05 * len(cumulative_evidence_ids)
        + 0.20 * len(cumulative_contradiction_ids)
        + min(0.08, 0.01 * meaningful_turns)
        + CONFESSION_LEVEL_BONUS.get(pressure_level, 0.0)
    )

    max_gain = 0.10
    if new_evidence_ids:
        max_gain += 0.03
    if new_contradiction_ids:
        max_gain += 0.06
    if pressure_level == "high":
        max_gain += 0.02
    max_gain = min(0.20, max_gain)

    confession_probability = max(
        prior_confession_probability,
        min(computed_probability, prior_confession_probability + max_gain),
    )

    if detect_repeat(history, user_text):
        confession_probability = min(confession_probability, prior_confession_probability + 0.02)

    return (
        pressure_delta,
        clamp01(confession_probability),
        sorted(cumulative_evidence_ids),
        sorted(cumulative_contradiction_ids),
    )

# =========================================================
# STT
# =========================================================
def stt_transcribe(audio_bytes: bytes) -> str:
    """
    1) gpt-4o-mini-transcribe 시도
    2) 포맷/호환 에러면 whisper-1로 폴백
    """
    def _call(model_name: str) -> str:
        bio = BytesIO(audio_bytes)
        bio.name = "user.wav"
        res = client.audio.transcriptions.create(
            model=model_name,
            file=bio,
            response_format="json",
            language="ko",
        )
        return (getattr(res, "text", "") or "").strip()

    try:
        return _call(STT_PRIMARY)
    except BadRequestError as e:
        msg = str(e)
        if "unsupported_format" in msg or "messages" in msg:
            return _call(STT_FALLBACK)
        raise

# =========================================================
# TTS
# =========================================================
async def tts_to_b64(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    speech = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )

    if hasattr(speech, "read") and callable(getattr(speech, "read")):
        wav_bytes = speech.read()
    elif hasattr(speech, "content"):
        wav_bytes = speech.content
    else:
        wav_bytes = bytes(speech)

    if not wav_bytes:
        return ""

    return base64.b64encode(wav_bytes).decode("ascii")

# =========================================================
# CASE GENERATION (Structured Outputs) - 현재 JSON 스키마
# =========================================================
CASE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "case_id": {"type": "string"},
        "overview": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "time": {"type": "string"},
                "place": {"type": "string"},
                "type": {"type": "string"},
            },
            "required": ["time", "place", "type"],
        },
        "motive": {"type": "string"},
        "crime_flow": {"type": "string"},
        "suspect": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "job": {"type": "string"},
                "relation": {"type": "string"},
            },
            "required": ["name", "age", "job", "relation"],
        },
        "false_statement": {"type": "string"},
        "evidences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["id", "name", "description"],
            },
            "minItems": 2,
            "maxItems": 6,
        },
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "related_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                # ✅ 여기만 수정: related_evidence를 required에 포함
                "required": ["id", "description", "related_evidence"],
            },
            "minItems": 2,
            "maxItems": 5,
        },
    },
    "required": [
        "case_id",
        "overview",
        "motive",
        "crime_flow",
        "suspect",
        "false_statement",
        "evidences",
        "contradictions",
    ],
}

def coerce_case_payload(case_blob: Any, fallback_case_id: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(case_blob, dict):
        return None

    overview = case_blob.get("overview", {}) if isinstance(case_blob.get("overview"), dict) else {}
    suspect = case_blob.get("suspect", {}) if isinstance(case_blob.get("suspect"), dict) else {}
    evidences = case_blob.get("evidences", []) if isinstance(case_blob.get("evidences"), list) else []
    contradictions = case_blob.get("contradictions", []) if isinstance(case_blob.get("contradictions"), list) else []

    def to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    normalized_evidences: List[Dict[str, str]] = []
    for evidence in evidences:
        if not isinstance(evidence, dict):
            continue
        normalized_evidences.append(
            {
                "id": norm(evidence.get("id", "")),
                "name": norm(evidence.get("name", "")),
                "description": norm(evidence.get("description", "")),
            }
        )

    normalized_contradictions: List[Dict[str, Any]] = []
    for contradiction in contradictions:
        if not isinstance(contradiction, dict):
            continue
        related_evidence = contradiction.get("related_evidence", [])
        if not isinstance(related_evidence, list):
            related_evidence = []
        normalized_contradictions.append(
            {
                "id": norm(contradiction.get("id", "")),
                "description": norm(contradiction.get("description", "")),
                "related_evidence": uniq_strings([str(item).strip() for item in related_evidence]),
            }
        )

    case_id = norm(case_blob.get("case_id", "")) or norm(fallback_case_id)
    if not case_id:
        return None

    return {
        "case_id": case_id,
        "overview": {
            "time": norm(overview.get("time", "")),
            "place": norm(overview.get("place", "")),
            "type": norm(overview.get("type", "")),
        },
        "motive": norm(case_blob.get("motive", "")),
        "crime_flow": norm(case_blob.get("crime_flow", "")),
        "suspect": {
            "name": norm(suspect.get("name", "")),
            "age": to_int(suspect.get("age", 0)),
            "job": norm(suspect.get("job", "")),
            "relation": norm(suspect.get("relation", "")),
        },
        "false_statement": norm(case_blob.get("false_statement", "")),
        "evidences": normalized_evidences,
        "contradictions": normalized_contradictions,
    }

def _case_type_key(text: str) -> str:
    return norm_for_match(text)

def _is_fire_case_type(text: str) -> bool:
    normalized = norm(text)
    return any(marker in normalized for marker in FIRE_CASE_MARKERS)

def _recent_case_type_keys() -> set:
    return {_case_type_key(case_type) for case_type in RECENT_CASE_TYPE_HISTORY if case_type}

def _pick_case_blueprint(excluded_labels: Optional[set] = None) -> Dict[str, Any]:
    excluded_labels = excluded_labels or set()
    recent_keys = _recent_case_type_keys()
    eligible = [
        blueprint
        for blueprint in CASE_VARIANT_BLUEPRINTS
        if blueprint["label"] not in excluded_labels
        and _case_type_key(blueprint["type_hint"]) not in recent_keys
    ]
    if not eligible:
        eligible = [
            blueprint
            for blueprint in CASE_VARIANT_BLUEPRINTS
            if blueprint["label"] not in excluded_labels
        ]
    if not eligible:
        eligible = CASE_VARIANT_BLUEPRINTS
    return random.choice(eligible)

def _register_generated_case_type(case_data: Dict[str, Any]) -> None:
    overview = case_data.get("overview", {}) if isinstance(case_data, dict) else {}
    case_type = norm(overview.get("type", "")) if isinstance(overview, dict) else ""
    if case_type:
        RECENT_CASE_TYPE_HISTORY.append(case_type)

def llm_generate_case() -> Dict[str, Any]:
    """
    seed 없이, case_interrogation_01.json 예시를 보고 자동 생성.
    - 구조/톤/필드 사용방식은 예시를 따르되
    - 내용(사건/인물/증거/모순)은 반드시 새로 만들기
    """
    system = (
        "너는 '음성 기반 심문 게임'의 사건 생성기다.\n"
        "현실적인 한국 수사/심문 톤의 사건을 만든다.\n"
        "규칙:\n"
        "- 출력은 반드시 JSON이며, 주어진 스키마를 엄격히 준수한다.\n"
        "- 아래 제공되는 '예시 사건 JSON'은 형식/톤/밀도(정보량)의 기준이다.\n"
        "- 예시와 '구조'는 비슷하게 유지하되, '내용'은 완전히 다른 새 사건으로 만들 것.\n"
        "- evidences는 2~6개, contradictions는 2~5개.\n"
        "- contradictions.related_evidence에는 연결되는 evidence id를 넣어라(가능하면 1개 이상).\n"
        "- false_statement는 초반엔 그럴듯하지만, 증거/모순 누적으로 무너지게 설계.\n"
        "- crime_flow는 '사실(정답)' 요약이며, 플레이어에게 직접 노출되지 않는 내부 참고용.\n"
        "- case_id는 'case_'로 시작하는 문자열.\n"
    )

    example_block = (
        f"\n[예시 사건 JSON]\n{EXAMPLE_CASE_TEXT}\n"
        if EXAMPLE_CASE_TEXT else
        "\n[예시 사건 JSON]\n(예시 파일이 없어도 생성은 해야 한다.)\n"
    )
    user = (
        "예시를 참고해서, 완전히 새로운 사건 1개를 생성하라.\n"
        "중요: 예시의 문장/표현/사건 디테일을 그대로 복사하지 말고, 다른 사건으로 만들 것.\n"
        "사건 유형/장소/증거/모순 조합이 뻔하지 않게 다양하게 만들 것.\n"
        + example_block +
        "\n이제 새 사건 JSON을 출력하라.\n"
    )

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "interrogation_case",
                "strict": True,
                "schema": CASE_JSON_SCHEMA,
            }
        },
        #max_output_tokens=1100,
        max_output_tokens=4000,
        store=False,
        temperature=1.0,
    )

    text = (resp.output_text or "").strip()
    return json.loads(text)

# Override the legacy generator with a diversity-aware version.
def llm_generate_case() -> Dict[str, Any]:
    """
    Generate a structured interrogation case while actively varying crime type.
    """
    system = (
        "You generate structured interrogation-game cases.\n"
        "Output must be valid JSON matching the provided schema.\n"
        "The example JSON is only a schema/style reference, not a crime-type template.\n"
        "Vary the crime type, setting, suspect relationship, and motive across generations.\n"
        "Do not keep generating arson or fire cases unless the target profile explicitly asks for one.\n"
        "evidences must contain 2 to 6 items.\n"
        "contradictions must contain 2 to 5 items.\n"
        "Each contradiction.related_evidence should reference at least one evidence id when possible.\n"
        "false_statement should sound plausible at first but collapse under evidence or contradiction pressure.\n"
        "crime_flow is the hidden ground truth summary.\n"
        "case_id must start with 'case_'.\n"
    )

    example_block = (
        f"\n[Example case JSON]\n{EXAMPLE_CASE_TEXT}\n"
        if EXAMPLE_CASE_TEXT else
        "\n[Example case JSON]\n(No example file available.)\n"
    )
    recent_types = list(RECENT_CASE_TYPE_HISTORY)
    tried_labels = set()
    last_case_data: Optional[Dict[str, Any]] = None

    for _attempt in range(3):
        blueprint = _pick_case_blueprint(tried_labels)
        tried_labels.add(blueprint["label"])
        recent_types_text = ", ".join(recent_types) if recent_types else "(none)"
        fire_rule = (
            "Fire or arson is allowed for this run."
            if blueprint["allow_fire"]
            else "Do not generate a fire, arson, blaze, or burn case for this run."
        )
        user = (
            "Create one new interrogation case.\n"
            "Important: do not copy the example's wording, facts, or crime type.\n"
            "Make the case meaningfully different from recent generations.\n"
            f"\n[Target crime profile]\n- Crime type: {blueprint['type_hint']}\n"
            f"- Recommended setting: {blueprint['place_hint']}\n"
            f"- Recommended motive direction: {blueprint['motive_hint']}\n"
            f"\n[Recent crime types to avoid repeating]\n- {recent_types_text}\n"
            f"\n[Hard diversity rule]\n- {fire_rule}\n"
            "- overview.type must clearly match the target crime profile.\n"
            + example_block +
            "\nNow output only the new case JSON.\n"
        )

        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "interrogation_case",
                    "strict": True,
                    "schema": CASE_JSON_SCHEMA,
                }
            },
            max_output_tokens=4000,
            store=False,
            temperature=1.0,
        )

        text = (resp.output_text or "").strip()
        case_data = json.loads(text)
        last_case_data = case_data
        generated_type = norm((case_data.get("overview", {}) or {}).get("type", ""))
        generated_key = _case_type_key(generated_type)

        should_retry = False
        if generated_key and generated_key in _recent_case_type_keys():
            should_retry = True
        if not blueprint["allow_fire"] and _is_fire_case_type(generated_type):
            should_retry = True

        if not should_retry:
            _register_generated_case_type(case_data)
            return case_data

    if last_case_data:
        _register_generated_case_type(last_case_data)
        return last_case_data
    raise RuntimeError("Failed to generate case")


# =========================================================
# LLM (Suspect QnA / Confession)
# =========================================================
def _collect_allowed_evidence_words(case_data: Optional[Dict[str, Any]], detective_mentions: List[str]) -> List[str]:
    allowed: List[str] = []
    if not case_data:
        return allowed

    for e in (case_data.get("evidences", []) or []):
        if not isinstance(e, dict):
            continue
        name = (e.get("name", "") or "").strip()
        eid = (e.get("id", "") or "").strip()
        for m in detective_mentions:
            m_match = norm_for_match(m)
            name_match = norm_for_match(name)
            eid_match = norm_for_match(eid)
            if name_match and name_match in m_match:
                allowed.append(name)
            if eid_match and eid_match in m_match:
                allowed.append(eid)

    # uniq preserve order
    out = []
    seen = set()
    for w in allowed:
        if w and w not in seen:
            out.append(w)
            seen.add(w)
    return out

def _all_evidence_words(case_data: Optional[Dict[str, Any]]) -> List[str]:
    if not case_data:
        return []
    words = []
    for e in (case_data.get("evidences", []) or []):
        if isinstance(e, dict):
            if e.get("id"):
                words.append(str(e["id"]))
            if e.get("name"):
                words.append(str(e["name"]))
    out = []
    seen = set()
    for w in words:
        w = (w or "").strip()
        if w and w not in seen:
            out.append(w)
            seen.add(w)
    return out

def _contains_banned_evidence(text: str, all_words: List[str], allowed_words: List[str]) -> bool:
    t = norm_for_match(text)
    allowed = set(norm_for_match(w) for w in (allowed_words or []) if w)
    for w in all_words:
        w_match = norm_for_match(w)
        if w_match and w_match in t and w_match not in allowed:
            return True
    return False

def _build_turn_pressure_context(
    case_data: Optional[Dict[str, Any]],
    interrogation_signal: Optional[Dict[str, Any]],
) -> str:
    if not case_data or not interrogation_signal:
        return ""

    current_turn = interrogation_signal.get("current_turn", {}) or {}
    pressure_level = str(current_turn.get("pressure_level", "none")).strip().lower()
    current_evidence_ids = current_turn.get("referenced_evidence_ids", []) or []
    current_contradiction_ids = current_turn.get("established_contradiction_ids", []) or []

    evidence_map = {
        str(e.get("id", "")).strip(): e
        for e in _case_evidences(case_data)
        if e.get("id")
    }
    contradiction_map = {
        str(c.get("id", "")).strip(): c
        for c in _case_contradictions(case_data)
        if c.get("id")
    }

    evidence_lines = []
    for evidence_id in current_evidence_ids:
        evidence = evidence_map.get(str(evidence_id).strip())
        if evidence:
            evidence_lines.append(
                f"- {evidence.get('name', evidence_id)}: {evidence.get('description', '')}".strip()
            )

    contradiction_lines = []
    for contradiction_id in current_contradiction_ids:
        contradiction = contradiction_map.get(str(contradiction_id).strip())
        if contradiction:
            contradiction_lines.append(f"- {contradiction.get('description', '')}".strip())

    directives = []
    if current_contradiction_ids:
        directives.append(
            "- 이번 답변에서는 방금 지적된 모순을 직접 다뤄라. 짧은 부인이나 말 돌리기는 금지다."
        )
    elif current_evidence_ids:
        directives.append(
            "- 이번 답변에서는 방금 언급된 증거에 직접 반응하라. 일반론으로만 회피하지 마라."
        )

    if pressure_level in {"medium", "high"}:
        directives.append(
            "- 이번 답변에서는 시간, 장소, 행동 중 최소 하나를 구체적으로 말해라."
        )
    if pressure_level == "high":
        directives.append(
            "- 완전 자백은 아니어도 진술 일부 수정, 사실 일부 인정, 감정 흔들림이 자연스럽다."
        )

    if not evidence_lines and not contradiction_lines and not directives:
        return ""

    return (
        f"\n[이번 턴 압박 수준] {pressure_level}\n"
        "[이번 턴에 형사가 꺼낸 증거]\n"
        + ("\n".join(evidence_lines) if evidence_lines else "(없음)")
        + "\n[이번 턴에 형사가 찌른 모순]\n"
        + ("\n".join(contradiction_lines) if contradiction_lines else "(없음)")
        + "\n[이번 답변 지침]\n"
        + ("\n".join(directives) if directives else "(없음)")
    )

def _is_overly_evasive_answer(text: str) -> bool:
    t = norm(text)
    if not t:
        return True

    t_match = norm_for_match(t)
    evasive_patterns = [
        "모르겠습니다",
        "아닙니다",
        "기억 안 납니다",
        "잘 모르겠습니다",
        "할 말 없습니다",
        "그건 아닙니다",
    ]

    return len(t) <= 28 and any(
        norm_for_match(pattern) in t_match for pattern in evasive_patterns
    )

def _build_turn_pressure_context_v2(
    case_data: Optional[Dict[str, Any]],
    interrogation_signal: Optional[Dict[str, Any]],
) -> str:
    if not case_data or not interrogation_signal:
        return ""

    current_turn = interrogation_signal.get("current_turn", {}) or {}
    pressure_level = str(current_turn.get("pressure_level", "none")).strip().lower()
    current_evidence_ids = current_turn.get("referenced_evidence_ids", []) or []
    current_contradiction_ids = current_turn.get("established_contradiction_ids", []) or []

    evidence_map = {
        str(e.get("id", "")).strip(): e
        for e in _case_evidences(case_data)
        if e.get("id")
    }
    contradiction_map = {
        str(c.get("id", "")).strip(): c
        for c in _case_contradictions(case_data)
        if c.get("id")
    }

    evidence_lines = []
    for evidence_id in current_evidence_ids:
        evidence = evidence_map.get(str(evidence_id).strip())
        if evidence:
            evidence_lines.append(
                f"- {evidence.get('name', evidence_id)}: {evidence.get('description', '')}".strip()
            )

    contradiction_lines = []
    for contradiction_id in current_contradiction_ids:
        contradiction = contradiction_map.get(str(contradiction_id).strip())
        if contradiction:
            contradiction_lines.append(f"- {contradiction.get('description', '')}".strip())

    directives = []
    if current_contradiction_ids:
        directives.append(
            "- Prioritize the contradiction above everything else. Answer that inconsistency directly in polite Korean."
        )
        directives.append(
            "- Keep the reply short and defensive. Do not widen the answer with unrelated evidence or narration."
        )
    elif current_evidence_ids:
        directives.append(
            "- React to the evidence briefly in polite Korean. Do not overexplain unless it clearly exposes a contradiction."
        )

    if pressure_level in {"medium", "high"}:
        directives.append(
            "- Give only one concrete detail, and only if it helps explain the pressure point or contradiction."
        )
    if pressure_level == "high":
        directives.append(
            "- Let the story wobble a little. You may partially correct yourself, but do not fully confess."
        )

    if not evidence_lines and not contradiction_lines and not directives:
        return ""

    return (
        f"\n[Pressure level this turn] {pressure_level}\n"
        "[Evidence raised this turn]\n"
        + ("\n".join(evidence_lines) if evidence_lines else "(none)")
        + "\n[Contradictions raised this turn]\n"
        + ("\n".join(contradiction_lines) if contradiction_lines else "(none)")
        + "\n[Response guidance]\n"
        + ("\n".join(directives) if directives else "(none)")
    )

def _build_suspect_answer_system_prompt() -> str:
    return (
        "You are a suspect being interrogated by a detective.\n"
        "Reply in natural spoken Korean using polite speech (존댓말) every turn.\n"
        "Use 1 or 2 short sentences, at most 3.\n"
        "Never use banmal.\n"
        "Do not sound like a narrator, a report, or an AI assistant.\n"
        "Answer the detective's latest question or accusation directly in the first sentence.\n"
        "Stay inside the scope of the latest question.\n"
        "Do not introduce a new scene, time, place, meeting, motive, or backstory unless the detective asked about it or it is strictly needed to answer.\n"
        "If the detective quotes a line, message, or statement, explain that exact line first.\n"
        "If the detective's pressure is weak, deny briefly or deflect politely.\n"
        "If a contradiction is raised, prioritize that contradiction over everything else.\n"
        "React to contradictions more strongly than to evidence lists.\n"
        "If evidence is mentioned without a clear contradiction, answer briefly and do not overexplain.\n"
        "If pressure is strong, let your story shake slightly and reveal only one small concrete fact.\n"
        "Do not mention evidence the detective has not brought up yet.\n"
        "Do not fully confess unless the confession trigger is reached.\n"
    )

def _build_confession_system_prompt() -> str:
    return (
        "You are a suspect finally breaking under interrogation.\n"
        "Reply in natural spoken Korean using polite speech (존댓말).\n"
        "Use 1 or 2 short sentences, at most 3.\n"
        "Never use banmal.\n"
        "Do not sound theatrical, poetic, or robotic.\n"
        "Confess the crime directly instead of circling around it.\n"
        "You may include guilt, fear, regret, or resignation, but keep it grounded.\n"
    )

def llm_suspect_answer(
    case_context: str,
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    confession_probability: float,
    interrogation_signal: Optional[Dict[str, Any]] = None,
) -> str:
    """
    용의자 답변 생성(2~3주차)
    - 형사가 꺼내지 않은 증거를 먼저 말하지 않게 가드
    - 1~3문장
    """
    system = (
        "너는 심문실에 앉아있는 '용의자'다.\n"
        "유저는 담당 형사다.\n"
        "규칙:\n"
        "- 한국어\n"
        "- 1~3문장\n"
        "- 너무 소설처럼 장황하게 말하지 말 것\n"
        "- 형사가 언급하지 않은 증거(예: CCTV, 목격자, 물증)를 네가 먼저 꺼내지 마라.\n"
        "- 증거/모순 압박이 약하면 기본 거짓 진술을 유지하며 버틴다.\n"
        "- 압박이 강하면 변명/흔들림/부분 인정(핵심 범행 자백은 금지)을 한다.\n"
        "- 자백 트리거가 아니면 범행을 완전 자백하지 말 것.\n"
    )

    system = _build_suspect_answer_system_prompt()
    recent = history[-4:] if isinstance(history, list) else []
    hist_lines = []
    detective_mentions = []
    for h in recent:
        if not isinstance(h, dict):
            continue
        u = (h.get("user_text", "") or "").strip()
        s = (h.get("suspect_text", "") or "").strip()
        if u:
            hist_lines.append(f"형사: {u}")
            detective_mentions.append(u)
        if s:
            hist_lines.append(f"용의자: {s}")

    detective_mentions.append(user_text)

    allowed_evidence_words = _collect_allowed_evidence_words(case_data, detective_mentions)
    all_evidence_words = _all_evidence_words(case_data)
    turn_pressure_context = _build_turn_pressure_context_v2(case_data, interrogation_signal)
    current_turn = (interrogation_signal or {}).get("current_turn", {}) if interrogation_signal else {}
    pressure_level = str(current_turn.get("pressure_level", "none")).strip().lower()
    has_current_contradiction = bool(current_turn.get("established_contradiction_ids", []))
    extra_guard = ""
    if turn_pressure_context:
        extra_guard += turn_pressure_context
    if has_current_contradiction:
        extra_guard += "\n- 방금 지적된 모순을 직접 해명하거나 진술 일부를 수정해라. 짧은 부인만 하지 마라."
    elif pressure_level in {"medium", "high"}:
        extra_guard += "\n- 이번에는 질문을 피해 가지 말고 구체적인 사실 하나는 내놓아라."
    if has_current_contradiction:
        extra_guard += "\n- Answer the contradiction first. Keep the reply polite, short, and focused on that inconsistency."
    elif pressure_level in {"medium", "high"}:
        extra_guard += "\n- Stay polite and give only one concrete detail."
    extra_guard += "\n- Answer only the detective's latest question or accusation."
    extra_guard += "\n- Put the direct answer in the first sentence."
    extra_guard += "\n- Do not widen into unrelated scenes, meetings, background stories, or prior events unless they are strictly needed to answer."
    if is_too_ambiguous(user_text):
        extra_guard += "\n- 질문이 모호하면 무슨 뜻인지 되묻고 질문을 구체화하게 유도해라."
    if detect_repeat(history, user_text):
        extra_guard += "\n- 같은 질문을 반복하면 이미 답했다고 짧게 말해라."

    user = (
        case_context +
        f"\n[현재 자백 확률(참고용)] {confession_probability:.2f}\n"
        "[최근 대화]\n" + ("\n".join(hist_lines) if hist_lines else "(없음)") + "\n"
        f"\n형사의 질문: {user_text}\n"
        f"\n[형사가 이미 꺼낸 증거 키워드(이것만 언급 가능)] {allowed_evidence_words}\n"
        f"\n[추가 가드]\n{extra_guard}\n"
        "용의자의 답변만 출력하라."
    )

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_output_tokens=220,
    )
    out = trim_to_1_3_sentences((resp.output_text or "").strip())
    should_retry_for_specificity = (
        pressure_level in {"medium", "high"} or has_current_contradiction
    ) and _is_overly_evasive_answer(out)

    # 금지 증거 스포가 나오면 1회 재시도
    if _contains_banned_evidence(out, all_evidence_words, allowed_evidence_words):
        retry_user = (
            user +
            "\n\n[경고] 방금 답변에 '형사가 언급하지 않은 증거'가 포함됐다. "
            "그런 단어는 절대 말하지 말고, 일반적인 부인/변명으로만 다시 답해라."
        )
        resp2 = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": retry_user},
            ],
            max_output_tokens=200,
        )
        out = trim_to_1_3_sentences((resp2.output_text or "").strip())

    if should_retry_for_specificity and _is_overly_evasive_answer(out):
        retry_user = (
            user +
            "\n\n[경고] 방금 답변은 너무 회피적이다. "
            "이번에는 방금 제시된 증거나 모순에 직접 반응하고, 시간, 장소, 행동 중 하나는 구체적으로 말해라."
        )
        resp3 = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": retry_user},
            ],
            max_output_tokens=200,
        )
        out = trim_to_1_3_sentences((resp3.output_text or "").strip())

    return out or "…모릅니다. 정말 집에 있었습니다."

# Override the legacy suspect answer generator with a question-focused version.
def llm_suspect_answer(
    case_context: str,
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    confession_probability: float,
    interrogation_signal: Optional[Dict[str, Any]] = None,
) -> str:
    system = _build_suspect_answer_system_prompt()
    recent = history[-4:] if isinstance(history, list) else []
    hist_lines: List[str] = []
    detective_mentions: List[str] = []
    for h in recent:
        if not isinstance(h, dict):
            continue
        u = norm(h.get("user_text", ""))
        s = norm(h.get("suspect_text", ""))
        if u:
            hist_lines.append(f"Detective: {u}")
            detective_mentions.append(u)
        if s:
            hist_lines.append(f"Suspect: {s}")

    detective_mentions.append(user_text)

    allowed_evidence_words = _collect_allowed_evidence_words(case_data, detective_mentions)
    all_evidence_words = _all_evidence_words(case_data)
    turn_pressure_context = _build_turn_pressure_context_v2(case_data, interrogation_signal)
    current_turn = (interrogation_signal or {}).get("current_turn", {}) if interrogation_signal else {}
    pressure_level = str(current_turn.get("pressure_level", "none")).strip().lower()
    has_current_contradiction = bool(current_turn.get("established_contradiction_ids", []))

    extra_guard = ""
    if turn_pressure_context:
        extra_guard += turn_pressure_context
    if has_current_contradiction:
        extra_guard += "\n- Answer the contradiction directly and first."
        extra_guard += "\n- Keep the reply polite, short, and focused on that inconsistency."
    elif pressure_level in {"medium", "high"}:
        extra_guard += "\n- Respond to the pressure point directly instead of broadening the story."
        extra_guard += "\n- Stay polite and give only one concrete detail."
    extra_guard += "\n- Answer only the detective's latest question or accusation."
    extra_guard += "\n- Put the direct answer in the first sentence."
    extra_guard += "\n- If the detective quotes a line or message, explain that exact line first."
    extra_guard += "\n- Do not widen into unrelated scenes, meetings, background stories, or prior events unless they are strictly needed to answer."
    if is_too_ambiguous(user_text):
        extra_guard += "\n- If the question is vague, ask what point the detective wants clarified."
    if detect_repeat(history, user_text):
        extra_guard += "\n- If the same question is repeated, answer that same point plainly."

    user = (
        case_context
        + f"\n[Current confession probability reference] {confession_probability:.2f}\n"
        + "[Recent dialogue]\n"
        + ("\n".join(hist_lines) if hist_lines else "(none)")
        + f"\n[Latest detective question]\n{user_text}\n"
        + f"\n[Evidence words already raised by the detective; do not go beyond these unless strictly needed] {allowed_evidence_words}\n"
        + f"\n[Response guardrails]\n{extra_guard}\n"
        + "Output only the suspect's reply."
    )

    def _generate_suspect_reply(prompt_text: str, max_output_tokens: int = 220) -> str:
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_text},
            ],
            max_output_tokens=max_output_tokens,
        )
        return trim_to_1_3_sentences((resp.output_text or "").strip())

    out = _generate_suspect_reply(user, 220)

    if _contains_banned_evidence(out, all_evidence_words, allowed_evidence_words):
        retry_user = (
            user
            + "\n\n[Correction] The previous reply mentioned evidence or facts the detective did not bring up. Remove that and answer with only a general denial or limited explanation."
        )
        out = _generate_suspect_reply(retry_user, 200)

    review = llm_review_suspect_reply(user_text, out)
    should_retry_for_specificity = (
        pressure_level in {"medium", "high"} or has_current_contradiction
    ) and _is_overly_evasive_answer(out)
    should_retry_for_relevance = (
        not review.get("answers_question_directly", True)
        or review.get("introduces_unrelated_details", False)
        or review.get("evades_core_issue", False)
    )

    if should_retry_for_specificity or should_retry_for_relevance:
        warnings: List[str] = []
        if should_retry_for_relevance:
            warnings.append(
                "The previous reply drifted away from the detective's latest question. Answer that exact question first and remove unrelated details."
            )
            if review.get("reason"):
                warnings.append(f"Reviewer note: {review['reason']}")
        if should_retry_for_specificity:
            warnings.append(
                "Do not stay vague. Respond directly to the pressure point with only one necessary concrete detail."
            )
        retry_user = user + "\n\n[Correction] " + " ".join(warnings)
        out = _generate_suspect_reply(retry_user, 200)

        final_review = llm_review_suspect_reply(user_text, out)
        if (
            not is_too_ambiguous(user_text)
            and (
                not final_review.get("answers_question_directly", True)
                or final_review.get("introduces_unrelated_details", False)
                or final_review.get("evades_core_issue", False)
            )
        ):
            final_retry_user = (
                user
                + "\n\n[Final correction] Do not add any new background. Reply with only the direct answer to the latest question in one or two short polite sentences."
            )
            out = _generate_suspect_reply(final_retry_user, 180)

    return out or "죄송하지만 그 질문에는 바로 답드리기 어렵습니다."

# Override the slow post-answer evaluators with local heuristics.
_CONTRADICTION_CUE_PATTERNS = (
    "아까", "방금", "전에", "처음엔", "그런데", "근데", "모순", "말이다르",
    "말이다르잖", "했잖", "라면서", "라고했", "아니라며", "왜이제", "왜지금",
)
_HOME_REST_PATTERNS = ("자고있", "잤", "집에있", "집에만있", "안나갔", "밖에안나갔")
_PRESENCE_DENY_PATTERNS = ("안갔", "간적없", "간일없", "안왔", "안들렀", "방문안")
_PRESENCE_ADMIT_PATTERNS = ("갔", "가서", "왔", "들렀", "방문", "찾아갔", "확인하러", "나갔")
_MEET_DENY_PATTERNS = ("안만났", "만난적없", "본적없", "마주친적없", "연락안했")
_MEET_ADMIT_PATTERNS = ("만났", "봤", "마주쳤", "연락했", "통화했")
_ALONE_PATTERNS = ("혼자있", "혼자였", "저혼자")
_WITH_OTHER_PATTERNS = ("같이있", "함께있", "둘이있", "누구랑있", "누구와있")
_REPLY_SCOPE_MARKERS = ("메신저", "문자", "카톡", "대화", "말", "문구", "뜻", "의미")
_BACKGROUND_DETAIL_MARKERS = (
    "전날", "어제", "오늘", "회의실", "사무실", "회사", "집", "골목", "주차장",
    "카페", "술집", "대표권", "말다툼", "목소리", "전화", "따로", "현장",
)
_TOKEN_STOPWORDS = {
    "그", "그거", "그건", "그게", "이", "이거", "이건", "저", "저는", "제가",
    "당신", "당신이", "그때", "지금", "왜", "뭐", "무슨", "어떻게", "그리고",
    "근데", "그런데", "그냥", "정말", "진짜", "그말", "그문구", "그메시지",
}

def _split_short_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!\?])\s+|(?<=[다요니다까])\s+", norm(text))
    return [part.strip() for part in parts if part and part.strip()]

def _tokenize_koreanish(text: str) -> List[str]:
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", norm(text))
    return [
        token for token in tokens
        if len(token) >= 2 and token not in _TOKEN_STOPWORDS
    ]

def _squash_korean_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", norm(text).lower())

def _contains_any_match(text: str, patterns: Tuple[str, ...]) -> bool:
    target = _squash_korean_text(text)
    return any(pattern in target for pattern in patterns)

def _extract_dialogue_flags(text: str) -> Dict[str, bool]:
    target = _squash_korean_text(text)
    presence_denied = any(pattern in target for pattern in _PRESENCE_DENY_PATTERNS)
    meet_denied = any(pattern in target for pattern in _MEET_DENY_PATTERNS)
    return {
        "home_rest": any(pattern in target for pattern in _HOME_REST_PATTERNS),
        "presence_denied": presence_denied,
        "presence_admitted": (not presence_denied) and any(
            pattern in target for pattern in _PRESENCE_ADMIT_PATTERNS
        ),
        "meet_denied": meet_denied,
        "meet_admitted": (not meet_denied) and any(
            pattern in target for pattern in _MEET_ADMIT_PATTERNS
        ),
        "alone": any(pattern in target for pattern in _ALONE_PATTERNS),
        "with_other": any(pattern in target for pattern in _WITH_OTHER_PATTERNS),
    }

def llm_review_suspect_reply(user_text: str, suspect_text: str) -> Dict[str, Any]:
    detective_text = norm(user_text)
    reply_text = norm(suspect_text)
    if not reply_text:
        return _default_suspect_reply_review(
            "empty suspect reply",
            answers_question_directly=False,
        )

    sentences = _split_short_sentences(reply_text)
    first_sentence = sentences[0] if sentences else reply_text
    remaining = " ".join(sentences[1:]).strip()
    question_tokens = set(_tokenize_koreanish(detective_text))
    first_tokens = set(_tokenize_koreanish(first_sentence))
    remaining_tokens = set(_tokenize_koreanish(remaining))
    overlap_first = len(question_tokens & first_tokens)
    overlap_remaining = len(question_tokens & remaining_tokens)

    quote_or_message_question = (
        any(marker in detective_text for marker in _REPLY_SCOPE_MARKERS)
        or "\"" in detective_text
        or "'" in detective_text
    )
    direct_explanation = any(marker in first_sentence for marker in ("뜻", "의미", "아니", "아닙", "그말은", "그건"))
    evasive = _is_overly_evasive_answer(reply_text)

    answers_question_directly = not evasive
    if quote_or_message_question:
        answers_question_directly = direct_explanation or overlap_first > 0
    elif overlap_first == 0 and len(first_sentence) > 24 and not direct_explanation:
        answers_question_directly = False

    introduces_unrelated_details = False
    if remaining:
        remaining_has_background = any(marker in remaining for marker in _BACKGROUND_DETAIL_MARKERS)
        if remaining_has_background and overlap_remaining == 0:
            introduces_unrelated_details = True
        elif len(sentences) >= 3 and overlap_remaining <= 1:
            introduces_unrelated_details = True
        elif len(remaining) >= 28 and overlap_remaining == 0 and quote_or_message_question:
            introduces_unrelated_details = True

    evades_core_issue = evasive or (not answers_question_directly and not introduces_unrelated_details)
    reason_parts: List[str] = []
    if not answers_question_directly:
        reason_parts.append("reply does not directly address the latest question first")
    if introduces_unrelated_details:
        reason_parts.append("reply adds unrelated background details")
    if evades_core_issue:
        reason_parts.append("reply stays too indirect or evasive")

    return {
        "answers_question_directly": answers_question_directly,
        "introduces_unrelated_details": introduces_unrelated_details,
        "evades_core_issue": evades_core_issue,
        "reason": "; ".join(reason_parts),
    }

def llm_evaluate_dialogue_contradiction(
    history: List[Dict[str, Any]],
    user_text: str,
    suspect_text: str,
) -> Dict[str, Any]:
    current_suspect_text = norm(suspect_text)
    if not current_suspect_text:
        return _empty_dialogue_contradiction_signal("empty suspect reply")

    current_flags = _extract_dialogue_flags(current_suspect_text)
    if not any(current_flags.values()):
        return _empty_dialogue_contradiction_signal("no contradiction slots in current reply")

    detective_highlighted = _contains_any_match(user_text, _CONTRADICTION_CUE_PATTERNS)
    prior_turns = [
        norm(entry.get("suspect_text", ""))
        for entry in history[-8:]
        if isinstance(entry, dict) and norm(entry.get("suspect_text", ""))
    ]
    if not prior_turns:
        return _empty_dialogue_contradiction_signal("no prior suspect statements")

    for prior_text in reversed(prior_turns):
        prior_flags = _extract_dialogue_flags(prior_text)
        severity = "none"

        if current_flags["presence_admitted"] and (prior_flags["presence_denied"] or prior_flags["home_rest"]):
            severity = "high"
        elif (current_flags["presence_denied"] or current_flags["home_rest"]) and prior_flags["presence_admitted"]:
            severity = "high"
        elif current_flags["meet_admitted"] and prior_flags["meet_denied"]:
            severity = "high"
        elif current_flags["meet_denied"] and prior_flags["meet_admitted"]:
            severity = "high"
        elif current_flags["alone"] and prior_flags["with_other"]:
            severity = "medium"
        elif current_flags["with_other"] and prior_flags["alone"]:
            severity = "medium"

        if severity != "none":
            return {
                "detective_highlighted": detective_highlighted,
                "suspect_self_contradicted": True,
                "severity": severity,
                "prior_claim": prior_text,
                "current_claim": current_suspect_text,
                "reason": "local heuristic contradiction detected",
            }

    return {
        "detective_highlighted": False,
        "suspect_self_contradicted": False,
        "severity": "none",
        "prior_claim": "",
        "current_claim": "",
        "reason": "no local contradiction detected",
    }

def llm_confession(case_context: str, history: List[Dict[str, Any]], user_text: str) -> str:
    system = (
        "너는 심문실에 앉아있는 '용의자'다.\n"
        "유저는 담당 형사다.\n"
        "지금은 '자백하는 순간'이다.\n"
        "규칙:\n"
        "- 한국어\n"
        "- 1~3문장\n"
        "- 변명보다 '인정' 중심\n"
        "- 감정(체념/후회/두려움) 중 하나를 살짝 포함\n"
    )

    recent = history[-4:] if isinstance(history, list) else []
    hist_lines = []
    for h in recent:
        if not isinstance(h, dict):
            continue
        u = (h.get("user_text", "") or "").strip()
        s = (h.get("suspect_text", "") or "").strip()
        if u:
            hist_lines.append(f"형사: {u}")
        if s:
            hist_lines.append(f"용의자: {s}")

    user = (
        case_context +
        "\n[최근 대화]\n" + ("\n".join(hist_lines) if hist_lines else "(없음)") + "\n"
        f"\n형사의 마지막 압박/질문: {user_text}\n"
        "이제 범행을 자백하라."
    )

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": _build_confession_system_prompt()},
            {"role": "user", "content": user},
        ],
        max_output_tokens=200,
    )
    return trim_to_1_3_sentences((resp.output_text or "").strip()) or "…제가 했습니다. 더는 숨길 수 없어요."

# =========================================================
# ENDPOINTS
# =========================================================
@app.post("/case/generate")
async def case_generate():
    """
    seed 없이 자동 사건 생성:
    - 예시 JSON을 프롬프트에 넣어서 구조/톤을 맞춘다
    - case_id는 서버에서 고유하게 강제
    """
    try:
        case_data = llm_generate_case()

        # case_id 고유 강제 (중복 방지)
        cid = (case_data.get("case_id", "") or "").strip()
        if not cid or cid in CASE_CACHE or case_store_path(cid).exists():
            cid = f"case_{uuid.uuid4().hex[:8]}"
            case_data["case_id"] = cid

        CASE_CACHE[cid] = case_data
        persist_case(case_data)
        return JSONResponse(status_code=200, content={"case": case_data})

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb[-2000:]})

@app.post("/interrogation/qna")
async def interrogation_qna(
    file: Optional[UploadFile] = File(None),
    user_text: str = Form(""),
    case_id: str = Form(""),
    case_json: str = Form(""),
    history_json: str = Form("[]"),
):
    """
    Request(Form-data):
      - file: wav (optional)
      - user_text: text (optional, file 없을 때만 사용)
      - case_id: string  (/case/generate 로 받은 값)
      - history_json: JSON string (list)

    Response(JSON):
      - user_text
      - suspect_text
      - pressure_delta
      - confession_probability
      - confession_triggered
      - audio_wav_b64
    """
    try:
        # 0) history
        history = safe_json_loads(history_json, [])
        if not isinstance(history, list):
            history = []
        history = history[-20:]  # 너무 길면 토큰/비용 폭발 방지

        # 1) 입력 텍스트(STT or direct)
        final_user_text = (user_text or "").strip()
        if file is not None:
            audio_bytes = await file.read()
            if audio_bytes:
                final_user_text = stt_transcribe(audio_bytes)
        final_user_text = norm(final_user_text)

        case_id = norm(case_id)
        case_data = None
        client_case_payload = safe_json_loads(case_json, None) if case_json.strip() else None
        client_case_data = coerce_case_payload(client_case_payload, case_id)
        if client_case_data:
            payload_case_id = norm(client_case_data.get("case_id", ""))
            if case_id and payload_case_id != case_id:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "case_id_mismatch",
                        "message": "case_id and case_json.case_id do not match.",
                        "case_id": case_id,
                        "case_json_case_id": payload_case_id,
                    },
                )
            case_id = payload_case_id
            case_data = client_case_data
            CASE_CACHE[case_id] = case_data
            persist_case(case_data)
            print(f"[CaseStore] Refreshed case_id='{case_id}' from client case_json")
        elif case_id:
            case_data = load_case(case_id)
        if case_id and not case_data:
            print(f"[CaseStore] Missing case_id='{case_id}' in cache and disk")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "case_not_found",
                    "message": "Unknown case_id. The server no longer has the generated case for this session.",
                    "case_id": case_id,
                },
            )
        case_context = build_case_context(case_data)
        prior_progress = _get_progress_state(case_id, history)
        prior_confession_probability = float(prior_progress["confession_probability"])

        if not final_user_text:
            msg = "…잘 안 들립니다. 다시 말씀해 주세요."
            return JSONResponse(
                status_code=200,
                content={
                    "user_text": "",
                    "suspect_text": msg,
                    "pressure_delta": 0.0,
                    "confession_probability": prior_confession_probability,
                    "confession_triggered": False,
                    "audio_wav_b64": await tts_to_b64(msg),
                },
            )

        if is_too_ambiguous(final_user_text):
            msg = "무슨 뜻인지 잘 모르겠습니다. 시간이나 장소를 구체적으로 말해 보세요."
            return JSONResponse(
                status_code=200,
                content={
                    "user_text": final_user_text,
                    "suspect_text": msg,
                    "pressure_delta": 0.0,
                    "confession_probability": prior_confession_probability,
                    "confession_triggered": False,
                    "audio_wav_b64": await tts_to_b64(msg),
                },
            )

        # 2) case load (cache 우선)
        # 3) calc pressure/prob
        interrogation_signal = llm_evaluate_interrogation(case_data, history, final_user_text)
        pressure_delta, confession_probability, cumulative_evidence_ids, cumulative_contradiction_ids = calc_pressure_and_prob_v2(
            case_data,
            history,
            final_user_text,
            interrogation_signal,
            prior_progress,
        )
        confession_triggered = confession_probability >= 0.85

        # 4) LLM answer
        if confession_triggered:
            suspect_text = llm_confession(case_context, history, final_user_text)
        else:
            suspect_text = llm_suspect_answer(
                case_context,
                case_data,
                history,
                final_user_text,
                confession_probability,
                interrogation_signal,
            )

        suspect_text = trim_to_1_3_sentences(suspect_text)
        dialogue_contradiction_signal = llm_evaluate_dialogue_contradiction(
            history,
            final_user_text,
            suspect_text,
        )
        pressure_delta, confession_probability = apply_dialogue_contradiction_bonus(
            pressure_delta,
            confession_probability,
            dialogue_contradiction_signal,
            confession_triggered,
        )

        # 5) TTS
        wav_b64 = await tts_to_b64(suspect_text)
        updated_history = history + [
            {
                "user_text": final_user_text,
                "suspect_text": suspect_text,
            }
        ]
        _store_progress_state(
            case_id,
            updated_history,
            confession_probability,
            cumulative_evidence_ids,
            cumulative_contradiction_ids,
        )

        return JSONResponse(
            status_code=200,
            content={
                "user_text": final_user_text,
                "suspect_text": suspect_text,
                "pressure_delta": float(pressure_delta),
                "confession_probability": float(confession_probability),
                "confession_triggered": bool(confession_triggered),
                "audio_wav_b64": wav_b64,
            },
        )

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb[-2000:]},
        )

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")
