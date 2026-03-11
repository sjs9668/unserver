import os
import json
import base64
import traceback
import re
import hashlib
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
import uuid

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
        + 0.14 * len(new_contradiction_ids)
        + 0.03 * len(repeated_contradiction_ids)
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
