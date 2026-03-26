import os
import json
import base64
import traceback
import re
import hashlib
import random
import math
import copy
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
LLM_MODEL = "gpt-5.4-2026-03-05"
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
INTERROGATION_PROGRESS_CACHE: Dict[str, Dict[str, Any]] = {}
RUNTIME_STORE_DIR = Path(__file__).parent / "runtime_store"
CASE_STORE_DIR = RUNTIME_STORE_DIR / "cases"
PREBUILT_CASE_DIR = Path(__file__).parent / "cases" / "prebuilt"

for _store_dir in (CASE_STORE_DIR, PREBUILT_CASE_DIR):
    _store_dir.mkdir(parents=True, exist_ok=True)

# Final server shape:
# - cases come from prebuilt JSON files
# - the interrogation engine is still the existing server-side flow
#   (question analysis -> suspect answer -> contradiction detection -> scoring)
# - documents / personality / PAD are extra case metadata for briefing and UI linkage

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

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def normalize_progress_state(blob: Any) -> Dict[str, Any]:
    if not isinstance(blob, dict):
        return {}

    try:
        turn_count = int(blob.get("turn_count", 0) or 0)
    except (TypeError, ValueError):
        turn_count = 0

    return {
        "breakdown_probability": clamp01(blob.get("breakdown_probability", 0.0)),
        "referenced_evidence_ids": sorted(
            {str(eid).strip() for eid in (blob.get("referenced_evidence_ids", []) or []) if str(eid).strip()}
        ),
        "established_contradiction_ids": sorted(
            {
                str(cid).strip()
                for cid in (blob.get("established_contradiction_ids", []) or [])
                if str(cid).strip()
            }
        ),
        "stress_score": clamp01(blob.get("stress_score", 0.0)),
        "cooperation_score": clamp01(blob.get("cooperation_score", DEFAULT_COOPERATION_SCORE)),
        "cumulative_pressure": clamp01(blob.get("cumulative_pressure", 0.0)),
        "fsm_state": norm(blob.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE,
        "last_raw_odds": safe_float(blob.get("last_raw_odds", 0.0), 0.0),
        "last_sue_impact": safe_float(blob.get("last_sue_impact", 0.0), 0.0),
        "statement_collapse_stage": max(0, min(5, int(safe_float(blob.get("statement_collapse_stage", 0), 0.0)))),
        "core_fact_exposed": bool(
            blob.get("core_fact_exposed", False)
            or max(0, min(5, int(safe_float(blob.get("statement_collapse_stage", 0), 0.0)))) >= 5
            or clamp01(blob.get("breakdown_probability", 0.0)) >= BREAKDOWN_EXPOSURE_THRESHOLD
        ),
        "pad_state": _normalize_pad_state_blob(blob.get("pad_state", {})),
        "final_psychological_reaction": norm(blob.get("final_psychological_reaction", "")),
        "selected_personality": _normalize_selected_personality_blob(
            blob.get("selected_personality", {})
        ),
        "statement_records": _normalize_statement_records(blob.get("statement_records", [])),
        "submitted_judgment": _normalize_submitted_judgment(blob.get("submitted_judgment", {})),
        "final_psychological_report": _normalize_final_report_blob(blob.get("final_psychological_report", {})),
        "turn_count": max(0, turn_count),
    }

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_for_match(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"[\W_]+", "", s, flags=re.UNICODE)

def is_truthy_string(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "debug"}

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

PRESSURE_LEVEL_BONUS = {
    "none": 0.0,
    "low": 0.005,
    "medium": 0.01,
    "high": 0.02,
}

NEW_EVIDENCE_PRESSURE_DELTA = 0.08
REPEATED_EVIDENCE_PRESSURE_DELTA = 0.02
NEW_CONTRADICTION_PRESSURE_DELTA = 0.12
REPEATED_CONTRADICTION_PRESSURE_DELTA = 0.04
MAX_TURN_PRESSURE_DELTA = 0.22
MAX_GAME_TURNS = 10
BREAKDOWN_EXPOSURE_THRESHOLD = 0.85
STRESS_IDLE_DECAY = 0.01
STRESS_WEAK_TURN_DECAY = 0.003
STRESS_REPEAT_DECAY = 0.01
HIGH_IMPACT_SUE_THRESHOLD = 3.0
HIGH_IMPACT_PRESSURED_MIN_BREAKDOWN = 0.18
PRESSURE_SIGMOID_STEEPNESS = 9.5
PRESSURE_SIGMOID_MIDPOINT = 0.58
CORE_QUESTION_PRESSURE_BONUS = 0.03
NEW_EVIDENCE_PRESSURE_PROGRESS_BONUS = 0.04
NEW_CONTRADICTION_PRESSURE_PROGRESS_BONUS = 0.08
SOFT_DIALOGUE_PRESSURE_PROGRESS_BONUS = 0.02
SUE_PRESSURE_BONUS_SCALE = 0.02
MAX_TURN_CUMULATIVE_PRESSURE_GAIN = 0.30
VALID_CONTRADICTION_TYPES = {
    "claim_vs_evidence",
    "claim_vs_truth",
    "timeline_mismatch",
    "alibi_mismatch",
}

LEGACY_CONFESSION_COMPAT_THRESHOLD = BREAKDOWN_EXPOSURE_THRESHOLD
EXPOSURE_FSM_STATE = "Breakdown / Core Fact Exposure"

DIALOGUE_CONTRADICTION_PRESSURE_BONUS = {
    "detective_highlighted": {
        "none": 0.0,
        "low": 0.005,
        "medium": 0.01,
        "high": 0.015,
    },
    "suspect_self_contradicted": {
        "none": 0.0,
        "low": 0.01,
        "medium": 0.02,
        "high": 0.03,
    },
}

DIALOGUE_CONTRADICTION_CONFESSION_BONUS = {
    "detective_highlighted": {
        "none": 0.0,
        "low": 0.0,
        "medium": 0.005,
        "high": 0.01,
    },
    "suspect_self_contradicted": {
        "none": 0.0,
        "low": 0.005,
        "medium": 0.01,
        "high": 0.02,
    },
}

class InterrogationCore:
    def __init__(
        self,
        pressure_steepness: float = PRESSURE_SIGMOID_STEEPNESS,
        pressure_midpoint: float = PRESSURE_SIGMOID_MIDPOINT,
    ):
        self.w = pressure_steepness
        self.midpoint = pressure_midpoint
        self.sue_matrix = {
            "low_spec_low_src": 0.5,
            "low_spec_high_src": 1.2,
            "high_spec_low_src": 1.5,
            "high_spec_high_src": 3.0,
        }

    def calculate_sue_impact(self, specificity: str, source: str) -> float:
        key = f"{specificity}_spec_{source}_src"
        return float(self.sue_matrix.get(key, 0.0))

    def calculate_raw_odds(
        self,
        cumulative_pressure: float,
    ) -> float:
        return self.w * (clamp01(cumulative_pressure) - self.midpoint)

    def calculate_breakdown_probability(
        self,
        cumulative_pressure: float,
    ) -> Tuple[float, float]:
        sigmoid_input = max(-60.0, min(60.0, self.calculate_raw_odds(cumulative_pressure)))
        p_breakdown = 1.0 / (1.0 + math.exp(-sigmoid_input))
        return sigmoid_input, clamp01(p_breakdown)

    def evaluate_fsm_state(
        self,
        p_breakdown: float,
        contradictions: int,
        player_intent: str,
        cooperation: float,
        latest_sue_impact: float = 0.0,
    ) -> str:
        if p_breakdown >= BREAKDOWN_EXPOSURE_THRESHOLD:
            return EXPOSURE_FSM_STATE
        if (
            latest_sue_impact >= HIGH_IMPACT_SUE_THRESHOLD
            and contradictions >= 1
            and p_breakdown >= HIGH_IMPACT_PRESSURED_MIN_BREAKDOWN
        ):
            return "Pressured / Shaken"
        if player_intent == "Intimidate" and cooperation < 0.2:
            return "Angry / Uncooperative"
        if 0.4 <= p_breakdown < BREAKDOWN_EXPOSURE_THRESHOLD:
            return "Pressured / Shaken"
        return "Idle / Evasion"

DEFAULT_COOPERATION_SCORE = 0.55
DEFAULT_FSM_STATE = "Idle / Evasion"

HIGH_DEFENSE_JOB_HINTS = (
    "변호사",
    "법무",
    "경찰",
    "형사",
    "기자",
    "홍보",
    "영업",
    "팀장",
    "대표",
    "관리",
)

LOW_DEFENSE_JOB_HINTS = (
    "학생",
    "무직",
    "알바",
    "인턴",
)

HIGH_SPECIFICITY_EVIDENCE_HINTS = (
    "cctv",
    "영상",
    "카메라",
    "혈흔",
    "dna",
    "지문",
    "접이식",
    "위치기록",
    "기지국",
    "메시지",
    "문자",
    "통화기록",
)

HIGH_SOURCE_EVIDENCE_HINTS = (
    "cctv",
    "영상",
    "카메라",
    "기록",
    "로그",
    "메시지",
    "문자",
    "혈흔",
    "dna",
    "지문",
    "기지국",
    "gps",
)

def _build_interrogation_core(case_data: Optional[Dict[str, Any]]) -> InterrogationCore:
    return InterrogationCore()

def _infer_defense_intelligence(case_data: Optional[Dict[str, Any]]) -> float:
    suspect = case_data.get("suspect", {}) if isinstance(case_data, dict) else {}
    job = norm(str(suspect.get("job", ""))).lower() if isinstance(suspect, dict) else ""
    if any(hint in job for hint in HIGH_DEFENSE_JOB_HINTS):
        return 0.8
    if any(hint in job for hint in LOW_DEFENSE_JOB_HINTS):
        return 0.5
    return 0.65

def _infer_player_intent(interrogation_signal: Optional[Dict[str, Any]]) -> str:
    signal = interrogation_signal or {}
    intent = str(signal.get("intent", "irrelevant")).strip().lower()
    pressure_level = str(signal.get("pressure_level", "none")).strip().lower()
    if intent == "small_talk":
        return "Rapport"
    if intent == "point_contradiction" or pressure_level == "high":
        return "Intimidate"
    if intent == "present_evidence" or pressure_level == "medium":
        return "Confront"
    if intent in {
        "ask_time",
        "ask_place",
        "ask_weapon",
        "ask_relation",
        "ask_alibi",
        "ask_action",
        "ask_last_seen_place",
        "ask_meeting",
    }:
        return "Probe"
    return "Neutral"

def _infer_evidence_specificity(evidence: Dict[str, Any]) -> str:
    text = norm(f"{evidence.get('name', '')} {evidence.get('description', '')}").lower()
    if any(hint in text for hint in HIGH_SPECIFICITY_EVIDENCE_HINTS):
        return "high"
    return "low"

def _infer_evidence_source(evidence: Dict[str, Any]) -> str:
    text = norm(f"{evidence.get('name', '')} {evidence.get('description', '')}").lower()
    if any(hint in text for hint in HIGH_SOURCE_EVIDENCE_HINTS):
        return "high"
    return "low"

def _calculate_latest_sue_impact(
    case_data: Optional[Dict[str, Any]],
    mentioned_evidence_ids: List[str],
    contradiction_ids: List[str],
    core: InterrogationCore,
) -> float:
    if not case_data or not contradiction_ids:
        return 0.0

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

    candidate_evidence_ids = {
        str(eid).strip()
        for eid in (mentioned_evidence_ids or [])
        if str(eid).strip()
    }
    for contradiction_id in contradiction_ids:
        contradiction = contradiction_map.get(str(contradiction_id).strip())
        if not contradiction:
            continue
        for evidence_id in contradiction.get("related_evidence", []) or []:
            evidence_id = str(evidence_id).strip()
            if evidence_id:
                candidate_evidence_ids.add(evidence_id)

    impacts: List[float] = []
    for evidence_id in candidate_evidence_ids:
        evidence = evidence_map.get(evidence_id)
        if not evidence:
            continue
        specificity = _infer_evidence_specificity(evidence)
        source = _infer_evidence_source(evidence)
        impacts.append(core.calculate_sue_impact(specificity, source))

    return max(impacts) if impacts else 0.0

def _update_cooperation_score(
    prior_cooperation: float,
    player_intent: str,
    pressure_delta: float,
    new_evidence_count: int,
    new_contradiction_count: int,
    history: List[Dict[str, Any]],
    user_text: str,
) -> float:
    cooperation = clamp01(prior_cooperation)
    intent_delta = {
        "Rapport": 0.03,
        "Probe": -0.015,
        "Confront": -0.03,
        "Intimidate": -0.05,
        "Neutral": -0.005,
    }.get(player_intent, -0.005)
    cooperation += intent_delta
    cooperation -= pressure_delta * 0.25
    cooperation -= 0.01 * float(new_evidence_count)
    cooperation -= 0.04 * float(new_contradiction_count)
    if detect_repeat(history, user_text):
        cooperation -= 0.02
    return clamp01(cooperation)

def _update_stress_score(
    prior_stress_score: float,
    pressure_delta: float,
    pressure_level: str,
    current_evidence_ids: set,
    current_contradiction_ids: set,
    history: List[Dict[str, Any]],
    user_text: str,
) -> float:
    stress_delta = pressure_delta

    if not current_contradiction_ids and not current_evidence_ids:
        if pressure_level == "none":
            stress_delta = -STRESS_IDLE_DECAY
        elif pressure_level == "low":
            stress_delta = -STRESS_WEAK_TURN_DECAY

    if detect_repeat(history, user_text):
        stress_delta -= STRESS_REPEAT_DECAY

    return clamp01(prior_stress_score + stress_delta)

TRUTH_SLOT_NAMES = [
    "crime_time",
    "crime_place",
    "weapon",
    "victim_relation",
    "alibi_claim",
    "actual_action",
    "last_seen_place",
    "met_victim_that_day",
]

QUESTION_INTENTS = [
    "ask_time",
    "ask_place",
    "ask_weapon",
    "ask_relation",
    "ask_alibi",
    "ask_action",
    "ask_last_seen_place",
    "ask_meeting",
    "present_evidence",
    "point_contradiction",
    "small_talk",
    "irrelevant",
]

SLOT_LABELS = {
    "crime_time": "crime time",
    "crime_place": "crime place",
    "weapon": "weapon",
    "victim_relation": "victim relation",
    "alibi_claim": "alibi claim",
    "actual_action": "actual action",
    "last_seen_place": "last seen place",
    "met_victim_that_day": "whether the suspect met the victim that day",
}

QUESTION_SLOT_GUIDE: List[Dict[str, str]] = [
    {
        "slot": "crime_time",
        "meaning": "Questions about when the incident happened or what time the suspect was somewhere.",
    },
    {
        "slot": "crime_place",
        "meaning": "Questions about where the incident happened or where the suspect was.",
    },
    {
        "slot": "weapon",
        "meaning": "Questions about the object, tool, or weapon used or carried.",
    },
    {
        "slot": "victim_relation",
        "meaning": "Questions about the relationship, familiarity, or history with the victim.",
    },
    {
        "slot": "alibi_claim",
        "meaning": "Questions about what the suspect says they were doing or where they say they were.",
    },
    {
        "slot": "actual_action",
        "meaning": "Questions about what the suspect actually did during the incident.",
    },
    {
        "slot": "last_seen_place",
        "meaning": "Questions about where the victim or suspect was last seen before the incident.",
    },
    {
        "slot": "met_victim_that_day",
        "meaning": "Questions about whether the suspect met, saw, or contacted the victim that day.",
    },
]

QUESTION_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": QUESTION_INTENTS,
        },
        "target_slot": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": TRUTH_SLOT_NAMES,
                },
                {
                    "type": "null",
                },
            ]
        },
        "mentioned_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "pressure_level": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
        },
        "reason": {"type": "string"},
    },
    "required": [
        "intent",
        "target_slot",
        "mentioned_evidence_ids",
        "pressure_level",
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

def _empty_dialogue_contradiction_signal(reason: str = "") -> Dict[str, Any]:
    return {
        "detective_highlighted": False,
        "suspect_self_contradicted": False,
        "severity": "none",
        "prior_claim": "",
        "current_claim": "",
        "reason": reason,
    }

EVIDENCE_LEXICAL_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("cctv", "영상", "카메라", "camera", "감시"), ("cctv", "영상", "카메라", "감시카메라")),
    (("문자", "메시지", "카톡", "채팅"), ("문자", "메시지", "카톡", "채팅")),
    (("통화", "전화"), ("통화", "전화", "통화기록", "전화기록")),
    (("위치", "gps", "기지국"), ("위치", "위치기록", "gps", "기지국")),
    (("혈흔", "피", "혈액", "dna"), ("혈흔", "dna", "혈액")),
    (("칼", "흉기", "나이프", "knife"), ("칼", "흉기", "나이프")),
)

def _evidence_lexical_candidates(evidence: Dict[str, Any]) -> List[str]:
    evidence_id = str(evidence.get("id", "")).strip()
    name = str(evidence.get("name", "")).strip()
    description = str(evidence.get("description", "")).strip()
    aliases = evidence.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []

    candidates = [evidence_id, name, description]
    candidates.extend(str(alias).strip() for alias in aliases if str(alias).strip())
    lowered = norm(
        " ".join([name, description] + [str(alias).strip() for alias in aliases if str(alias).strip()])
    ).lower()
    for trigger_patterns, hint_terms in EVIDENCE_LEXICAL_HINTS:
        if any(pattern in lowered for pattern in trigger_patterns):
            candidates.extend(hint_terms)

    return uniq_strings([candidate for candidate in candidates if norm(candidate)])

def _lexical_evidence_hits(case_data: Optional[Dict[str, Any]], text: str) -> List[str]:
    text_match = norm_for_match(text)
    if not text_match:
        return []

    hits: List[str] = []
    for evidence in _case_evidences(case_data):
        evidence_id = str(evidence.get("id", "")).strip()
        candidates = _evidence_lexical_candidates(evidence)
        for candidate in candidates:
            candidate_match = norm_for_match(candidate)
            if candidate_match and candidate_match in text_match:
                if evidence_id:
                    hits.append(evidence_id)
                break
    return uniq_strings(hits)

def _default_truth_slots() -> Dict[str, str]:
    return {slot_name: "" for slot_name in TRUTH_SLOT_NAMES}

def _case_truth_slots(case_data: Optional[Dict[str, Any]]) -> Dict[str, str]:
    slots = _default_truth_slots()
    if not case_data:
        return slots

    raw_slots = case_data.get("truth_slots", {})
    if not isinstance(raw_slots, dict):
        return slots

    for slot_name in TRUTH_SLOT_NAMES:
        slots[slot_name] = norm(raw_slots.get(slot_name, ""))
    return slots

def _empty_question_analysis(reason: str = "") -> Dict[str, Any]:
    return {
        "intent": "irrelevant",
        "target_slot": None,
        "mentioned_evidence_ids": [],
        "pressure_level": "none",
        "reason": reason,
    }

QUESTION_SLOT_HINTS: Dict[str, Tuple[str, ...]] = {
    "crime_time": ("언제", "몇 시", "시각", "시간", "타임", "time", "clock"),
    "crime_place": ("어디", "장소", "현장", "place", "location", "복도", "편의점", "공장"),
    "weapon": ("무기", "흉기", "칼", "weapon"),
    "victim_relation": ("피해자", "관계", "사이", "relation"),
    "alibi_claim": (
        "알리바이",
        "뭐 했",
        "뭐하고 있었",
        "어디 있었",
        "어디 있었어",
        "어디에 있었",
        "그때 어디",
        "그 시간",
        "그 시간에",
        "그때 뭐",
        "집에",
        "집에 있었",
        "집에 있었다",
        "거실",
        "거실에 있었",
        "거실에 있었다",
        "식사 내내",
        "내내",
        "계속",
        "줄곧",
        "에만 있었다",
        "에만 있었",
        "거기 있었다",
        "거기 있었",
        "있었다고 했",
        "있었다 했",
        "있다 했",
        "있다며",
        "있었다며",
        "했다며",
        "했다고 했",
        "라고 했",
        "라고 했는데",
        "했는데",
        "라면서",
        "없었다고 했",
        "없었다며",
        "아까",
        "아까는",
        "아까 말했다",
        "말했잖",
        "말이 다르",
        "말이 다른데",
        "말 바뀌",
        "진술",
        "진술이 다른데",
        "alibi",
        "cctv",
        "cctv에 찍혔",
        "cctv에 찍혔다",
        "찍혔던데",
        "찍혀",
        "찍혀있",
        "찍힌 거",
        "찍혔는데",
        "영상에",
        "영상에 찍혀",
    ),
    "actual_action": ("무슨 짓", "무엇을 했", "행동", "actual action"),
    "last_seen_place": ("마지막", "봤던 곳", "last seen"),
    "met_victim_that_day": ("만났", "봤", "접촉", "met", "seen"),
}

QUESTION_INTENT_BY_SLOT = {
    "crime_time": "ask_time",
    "crime_place": "ask_place",
    "weapon": "ask_weapon",
    "victim_relation": "ask_relation",
    "alibi_claim": "ask_alibi",
    "actual_action": "ask_action",
    "last_seen_place": "ask_last_seen_place",
    "met_victim_that_day": "ask_meeting",
}

QUESTION_CATEGORY_LABELS = {
    "basic_fact": "기본 사실 질문",
    "statement_lock": "진술 고정 질문",
    "pressure": "압박 질문",
    "confirmation": "확인 질문",
    "repeat": "반복 질문",
    "misc": "기타 질문",
}

QUESTION_CONFIRMATION_HINTS = (
    "다시 말씀",
    "다시 말",
    "다시 한번",
    "한 번 더",
    "정리하면",
    "확인하자면",
    "다시 확인",
    "재차",
)

QUESTION_STATEMENT_LOCK_HINTS = (
    "맞지",
    "맞죠",
    "맞습니까",
    "맞나요",
    "맞다고",
    "맞잖",
    "였지",
    "였죠",
    "였습니까",
    "있었던 거 맞",
    "있었다는 거지",
    "있었다고 보면",
    "간 거 맞",
    "본 거 맞",
    "라고 했지",
    "라고 한 거지",
    "그 말 맞",
)

SLOT_HINT_PRIORITY = {
    "alibi_claim": 5,
    "crime_place": 4,
    "last_seen_place": 3,
    "crime_time": 2,
    "actual_action": 2,
    "victim_relation": 1,
    "weapon": 1,
    "met_victim_that_day": 1,
}

SMALL_TALK_HINTS = (
    "안녕",
    "반갑",
    "이름",
    "긴장",
    "괜찮",
    "hello",
    "hi",
)

CONTRADICTION_HINTS = (
    "모순",
    "말이 안",
    "앞뒤가 안",
    "거짓말",
    "왜 다르",
    "증거랑",
    "cctv랑",
    "contradiction",
    "inconsisten",
)

ACCUSATION_HINTS = (
    "거짓말",
    "숨기",
    "들켰",
    "왜 속였",
    "admit",
    "explain",
)

ALIBI_ATTACK_HINTS = (
    "있었다는데",
    "있었다 했는데",
    "있었다고 했는데",
    "있다는데",
    "했다는데",
    "아까",
    "진술",
    "cctv에 찍혔",
    "cctv에 찍혀",
    "말이 바뀌",
    "말이 바뀌었",
    "없었다며",
)

def _question_has_confirmation_cues(text: str) -> bool:
    lowered = norm(text).lower()
    squashed = norm_for_match(text)
    return any(
        pattern in lowered or norm_for_match(pattern) in squashed
        for pattern in QUESTION_CONFIRMATION_HINTS
    )

def _question_has_statement_lock_cues(text: str) -> bool:
    lowered = norm(text).lower()
    squashed = norm_for_match(text)
    return any(
        pattern in lowered or norm_for_match(pattern) in squashed
        for pattern in QUESTION_STATEMENT_LOCK_HINTS
    )

def _classify_question_category(
    user_text: str,
    question_analysis: Optional[Dict[str, Any]],
    repeated_question: bool = False,
) -> Dict[str, str]:
    analysis = question_analysis or {}
    intent = norm(analysis.get("intent", "")).lower()
    pressure_level = norm(analysis.get("pressure_level", "")).lower()
    target_slot = norm(analysis.get("target_slot", ""))

    if repeated_question:
        key = "repeat"
        reason = "same or very similar question repeated"
    elif _question_has_confirmation_cues(user_text):
        key = "confirmation"
        reason = "question explicitly asks for restatement or clarification"
    elif target_slot and _question_has_statement_lock_cues(user_text) and intent != "point_contradiction":
        key = "statement_lock"
        reason = "question tries to pin the suspect to a direct factual claim"
    elif intent in {"point_contradiction", "present_evidence"} or pressure_level in {"medium", "high"}:
        key = "pressure"
        reason = "question presses with contradiction or evidence-backed pressure"
    elif intent in QUESTION_INTENT_BY_SLOT.values():
        key = "basic_fact"
        reason = "question asks for a core factual detail"
    else:
        key = "misc"
        reason = "question does not map cleanly to the main interrogation categories"

    return {
        "key": key,
        "label": QUESTION_CATEGORY_LABELS.get(key, QUESTION_CATEGORY_LABELS["misc"]),
        "reason": reason,
    }

def _sanitize_question_analysis(
    case_data: Optional[Dict[str, Any]],
    raw_signal: Dict[str, Any],
) -> Dict[str, Any]:
    valid_evidence_ids = {
        str(e.get("id", "")).strip() for e in _case_evidences(case_data) if e.get("id")
    }
    intent = str(raw_signal.get("intent", "irrelevant")).strip().lower() if isinstance(raw_signal, dict) else "irrelevant"
    if intent not in QUESTION_INTENTS:
        intent = "irrelevant"

    target_slot = raw_signal.get("target_slot") if isinstance(raw_signal, dict) else None
    if target_slot is None:
        sanitized_slot = None
    else:
        sanitized_slot = str(target_slot).strip()
        if sanitized_slot not in TRUTH_SLOT_NAMES:
            sanitized_slot = None

    pressure_level = str(raw_signal.get("pressure_level", "none")).strip().lower() if isinstance(raw_signal, dict) else "none"
    if pressure_level not in PRESSURE_LEVEL_BONUS:
        pressure_level = "none"

    return {
        "intent": intent,
        "target_slot": sanitized_slot,
        "mentioned_evidence_ids": _sanitize_id_list(
            raw_signal.get("mentioned_evidence_ids", []) if isinstance(raw_signal, dict) else [],
            valid_evidence_ids,
        ),
        "pressure_level": pressure_level,
        "reason": norm(raw_signal.get("reason", "")) if isinstance(raw_signal, dict) else "",
    }

def _detect_slot_from_text(text: str) -> Optional[str]:
    normalized = norm(text).lower()
    best_slot = None
    best_score = 0
    best_priority = -1
    for slot_name, patterns in QUESTION_SLOT_HINTS.items():
        score = sum(1 for pattern in patterns if pattern in normalized)
        priority = SLOT_HINT_PRIORITY.get(slot_name, 0)
        if score > best_score or (score > 0 and score == best_score and priority > best_priority):
            best_slot = slot_name
            best_score = score
            best_priority = priority
    return best_slot if best_score > 0 else None

def _question_has_contradiction_cues(text: str) -> bool:
    lowered = norm(text).lower()
    extra_cues = (
        "했는데",
        "했다며",
        "라고 했는데",
        "라면서",
        "있었다고 했",
        "있었다며",
        "없었다고 했",
        "없었다며",
        "말이 다르",
        "말 바뀌",
        "진술이 다른데",
        "찍혔",
        "찍혀",
        "찍혀있",
        "영상에",
    )
    return any(
        pattern in lowered
        for pattern in CONTRADICTION_HINTS + ACCUSATION_HINTS + ALIBI_ATTACK_HINTS + extra_cues
    )

def _question_has_alibi_attack_cues(text: str) -> bool:
    lowered = norm(text).lower()
    extra_cues = (
        "에만 있었다",
        "내내",
        "계속",
        "줄곧",
        "말했잖",
        "말이 다른데",
        "거실",
        "편의점",
        "집에 있었다",
    )
    return any(pattern in lowered for pattern in ALIBI_ATTACK_HINTS + extra_cues)

def _best_contradiction_slot_for_evidence(
    case_data: Optional[Dict[str, Any]],
    evidence_ids: List[str],
    user_text: str = "",
) -> Optional[str]:
    evidence_id_set = {
        str(evidence_id).strip()
        for evidence_id in (evidence_ids or [])
        if str(evidence_id).strip()
    }
    if not case_data or not evidence_id_set:
        return None

    slot_scores: Dict[str, int] = {}
    alibi_attack = _question_has_alibi_attack_cues(user_text)
    for contradiction in _case_contradictions(case_data):
        related_evidence = {
            str(evidence_id).strip()
            for evidence_id in (contradiction.get("related_evidence", []) or [])
            if str(evidence_id).strip()
        }
        if not related_evidence or not related_evidence.intersection(evidence_id_set):
            continue

        slot_name = str(contradiction.get("slot", "")).strip()
        if slot_name not in TRUTH_SLOT_NAMES:
            continue

        score = 1
        contradiction_type = norm(contradiction.get("contradiction_type", "")).lower()
        if contradiction_type == "alibi_mismatch" and slot_name == "alibi_claim":
            score += 4 if alibi_attack else 2
        elif contradiction_type in {"claim_vs_evidence", "timeline_mismatch"}:
            score += 1
        if alibi_attack and slot_name == "alibi_claim":
            score += 2
        slot_scores[slot_name] = slot_scores.get(slot_name, 0) + score

    if not slot_scores:
        return None

    return max(
        slot_scores.items(),
        key=lambda item: (item[1], SLOT_HINT_PRIORITY.get(item[0], 0)),
    )[0]

def _backfill_question_analysis(
    case_data: Optional[Dict[str, Any]],
    user_text: str,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    backfilled = dict(analysis or {})
    lexical_slot = _detect_slot_from_text(user_text)
    lexical_evidence_ids = _lexical_evidence_hits(case_data, user_text)
    contradiction_cues = _question_has_contradiction_cues(user_text)
    alibi_attack_cues = _question_has_alibi_attack_cues(user_text)

    if lexical_slot and not backfilled.get("target_slot"):
        backfilled["target_slot"] = lexical_slot
        if backfilled.get("intent") in {"irrelevant", "small_talk"}:
            backfilled["intent"] = QUESTION_INTENT_BY_SLOT.get(lexical_slot, "irrelevant")
        if backfilled.get("pressure_level") == "none":
            backfilled["pressure_level"] = "low"
        reason = norm(backfilled.get("reason", ""))
        suffix = f"lexical slot backfill:{lexical_slot}"
        backfilled["reason"] = f"{reason}; {suffix}" if reason else suffix

    if lexical_evidence_ids and not backfilled.get("mentioned_evidence_ids"):
        backfilled["mentioned_evidence_ids"] = lexical_evidence_ids
        if backfilled.get("intent") == "irrelevant":
            backfilled["intent"] = "present_evidence"
        if backfilled.get("pressure_level") == "none":
            backfilled["pressure_level"] = "medium"

    effective_evidence_ids = uniq_strings(
        list(backfilled.get("mentioned_evidence_ids", []) or []) + lexical_evidence_ids
    )
    contradiction_slot = _best_contradiction_slot_for_evidence(
        case_data,
        effective_evidence_ids,
        user_text,
    )
    current_slot = backfilled.get("target_slot")
    if contradiction_slot:
        evidence_backed_alibi_push = (
            contradiction_slot == "alibi_claim"
            and bool(effective_evidence_ids)
            and (
                alibi_attack_cues
                or current_slot in {"alibi_claim", "crime_place"}
                or backfilled.get("intent") in {"present_evidence", "ask_place", "ask_alibi"}
            )
        )
        contradiction_promotion = (
            contradiction_cues
            or evidence_backed_alibi_push
        )
        override_implies_contradiction = bool(effective_evidence_ids) and current_slot not in {None, "", contradiction_slot}
        should_override = (
            current_slot in {None, "", "crime_place", "crime_time", "last_seen_place"}
            or (
                contradiction_slot == "alibi_claim"
                and current_slot in {"actual_action", "crime_place"}
            )
            or (
                contradiction_cues
                and backfilled.get("intent") in {"present_evidence", "point_contradiction", "ask_place"}
            )
            or (
                contradiction_slot == "alibi_claim"
                and effective_evidence_ids
                and (alibi_attack_cues or current_slot in {"crime_place", "alibi_claim"})
            )
        )
        if should_override and current_slot != contradiction_slot:
            backfilled["target_slot"] = contradiction_slot
            if contradiction_promotion or override_implies_contradiction:
                backfilled["intent"] = "point_contradiction"
                backfilled["pressure_level"] = "high"
            elif backfilled.get("intent") in {"irrelevant", "small_talk"}:
                backfilled["intent"] = QUESTION_INTENT_BY_SLOT.get(contradiction_slot, "irrelevant")
                if backfilled.get("pressure_level") == "none":
                    backfilled["pressure_level"] = "low"
            reason = norm(backfilled.get("reason", ""))
            suffix = f"evidence contradiction slot override:{contradiction_slot}"
            backfilled["reason"] = f"{reason}; {suffix}" if reason else suffix
        elif (
            current_slot == contradiction_slot
            and effective_evidence_ids
            and contradiction_promotion
            and backfilled.get("intent") != "point_contradiction"
        ):
            backfilled["intent"] = "point_contradiction"
            backfilled["pressure_level"] = "high"
            reason = norm(backfilled.get("reason", ""))
            suffix = f"evidence contradiction cue promotion:{contradiction_slot}"
            backfilled["reason"] = f"{reason}; {suffix}" if reason else suffix
    elif effective_evidence_ids and alibi_attack_cues and backfilled.get("target_slot") == "alibi_claim":
        backfilled["intent"] = "point_contradiction"
        backfilled["pressure_level"] = "high"
        reason = norm(backfilled.get("reason", ""))
        suffix = "alibi attack cue promotion"
        backfilled["reason"] = f"{reason}; {suffix}" if reason else suffix

    return _sanitize_question_analysis(case_data, backfilled)

def _fallback_question_analysis(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
) -> Dict[str, Any]:
    if not case_data:
        return _empty_question_analysis("case unavailable")

    current_text = norm(user_text)
    lowered = current_text.lower()
    current_evidence_ids = _lexical_evidence_hits(case_data, current_text)
    target_slot = _detect_slot_from_text(current_text)

    if any(pattern in lowered for pattern in SMALL_TALK_HINTS):
        intent = "small_talk"
    elif any(pattern in lowered for pattern in CONTRADICTION_HINTS):
        intent = "point_contradiction"
    elif current_evidence_ids:
        intent = "present_evidence"
    elif target_slot:
        intent = QUESTION_INTENT_BY_SLOT.get(target_slot, "irrelevant")
    else:
        intent = "irrelevant"

    pressure_level = "none"
    if intent in QUESTION_INTENT_BY_SLOT.values():
        pressure_level = "low"
    if current_evidence_ids or any(pattern in lowered for pattern in ACCUSATION_HINTS):
        pressure_level = "medium"
    if intent == "point_contradiction" and (current_evidence_ids or target_slot):
        pressure_level = "high"
    if detect_repeat(history, user_text):
        pressure_level = "none"

    return {
        "intent": intent,
        "target_slot": target_slot,
        "mentioned_evidence_ids": current_evidence_ids,
        "pressure_level": pressure_level,
        "reason": "fallback question analysis",
    }

def llm_evaluate_interrogation(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
) -> Dict[str, Any]:
    if not case_data:
        return _empty_question_analysis("case unavailable")

    payload = {
        "overview": case_data.get("overview", {}) or {},
        "available_slots": TRUTH_SLOT_NAMES,
        "slot_guide": QUESTION_SLOT_GUIDE,
        "evidences": [
            {
                "id": e.get("id", ""),
                "name": e.get("name", ""),
                "description": e.get("description", ""),
                "aliases": e.get("aliases", []),
            }
            for e in _case_evidences(case_data)
        ],
        "history": _dialogue_lines(history[-10:]),
        "current_detective_text": norm(user_text),
    }

    system = (
        "You analyze the detective's latest question in a free-form voice interrogation game.\n"
        "Return only JSON matching the schema.\n"
        "Your role is only natural-language structuring support.\n"
        "Do not decide contradiction ids. Do not judge whether the suspect is actually lying.\n"
        "Do not invent hidden facts beyond the detective's wording and the provided evidence list.\n"
        "Use only the available_slots provided in the payload.\n"
        "Use the slot_guide to map paraphrased Korean questions to the closest slot.\n"
        "Identify: the main intent, the target slot if any, evidence ids explicitly mentioned by the detective, and a coarse pressure level.\n"
        "Use mentioned_evidence_ids only when the detective clearly mentions an evidence id, evidence name, or a very specific evidence description.\n"
        "Use point_contradiction only when the detective explicitly points out an inconsistency or lie.\n"
        "Use target_slot null when the latest turn does not clearly target one slot.\n"
        "pressure_level guide: none=casual or irrelevant, low=basic factual question, medium=evidence-backed pressure, high=explicit contradiction push.\n"
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
                    "name": "question_analysis",
                    "strict": True,
                    "schema": QUESTION_ANALYSIS_SCHEMA,
                }
            },
            max_output_tokens=400,
            store=False,
        )
        raw_signal = safe_json_loads((resp.output_text or "").strip(), {})
        sanitized = _sanitize_question_analysis(case_data, raw_signal)
        return _backfill_question_analysis(case_data, user_text, sanitized)
    except Exception:
        return _fallback_question_analysis(case_data, history, user_text)

def _dialogue_contradiction_bonus_values(
    dialogue_signal: Optional[Dict[str, Any]],
) -> Tuple[float, float, bool]:
    if not isinstance(dialogue_signal, dict):
        return 0.0, 0.0, False

    severity = str(dialogue_signal.get("severity", "none")).strip().lower()
    if severity not in {"none", "low", "medium", "high"}:
        severity = "none"

    pressure_bonus_candidates = [0.0]
    breakdown_bonus_candidates = [0.0]
    soft_dialogue_contradiction = False

    if dialogue_signal.get("detective_highlighted"):
        soft_dialogue_contradiction = True
        pressure_bonus_candidates.append(
            DIALOGUE_CONTRADICTION_PRESSURE_BONUS["detective_highlighted"][severity]
        )
        breakdown_bonus_candidates.append(
            DIALOGUE_CONTRADICTION_CONFESSION_BONUS["detective_highlighted"][severity]
        )

    if dialogue_signal.get("suspect_self_contradicted"):
        soft_dialogue_contradiction = True
        pressure_bonus_candidates.append(
            DIALOGUE_CONTRADICTION_PRESSURE_BONUS["suspect_self_contradicted"][severity]
        )
        breakdown_bonus_candidates.append(
            DIALOGUE_CONTRADICTION_CONFESSION_BONUS["suspect_self_contradicted"][severity]
        )

    pressure_bonus = min(0.03, max(pressure_bonus_candidates))
    breakdown_bonus = max(breakdown_bonus_candidates)
    if severity == "none":
        soft_dialogue_contradiction = False
    return pressure_bonus, breakdown_bonus, soft_dialogue_contradiction

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

def _empty_progress_state() -> Dict[str, Any]:
    return {
        "breakdown_probability": 0.0,
        "referenced_evidence_ids": [],
        "established_contradiction_ids": [],
        "stress_score": 0.0,
        "cooperation_score": DEFAULT_COOPERATION_SCORE,
        "cumulative_pressure": 0.0,
        "fsm_state": DEFAULT_FSM_STATE,
        "last_raw_odds": 0.0,
        "last_sue_impact": 0.0,
        "statement_collapse_stage": 0,
        "core_fact_exposed": False,
        "pad_state": _default_mental_state(),
        "final_psychological_reaction": "",
        "selected_personality": {},
        "statement_records": [],
        "submitted_judgment": _normalize_submitted_judgment({}),
        "final_psychological_report": _normalize_final_report_blob({}),
        "turn_count": 0,
    }

def _get_progress_state(case_id: str) -> Dict[str, Any]:
    if not case_id:
        return _empty_progress_state()

    state = normalize_progress_state(INTERROGATION_PROGRESS_CACHE.get(case_id))
    if not state:
        return _empty_progress_state()
    return state

def _store_progress_state(
    case_id: str,
    breakdown_probability: float,
    evidence_ids: List[str],
    contradiction_ids: List[str],
    stress_score: float = 0.0,
    cooperation_score: float = DEFAULT_COOPERATION_SCORE,
    cumulative_pressure: float = 0.0,
    fsm_state: str = DEFAULT_FSM_STATE,
    last_raw_odds: float = 0.0,
    last_sue_impact: float = 0.0,
    turn_count: int = 0,
    statement_collapse_stage: int = 0,
    pad_state: Optional[Dict[str, float]] = None,
    final_psychological_reaction: str = "",
    selected_personality: Optional[Dict[str, float]] = None,
    statement_records: Optional[List[Dict[str, Any]]] = None,
    submitted_judgment: Optional[Dict[str, Any]] = None,
    final_psychological_report: Optional[Dict[str, Any]] = None,
) -> None:
    if not case_id:
        return

    INTERROGATION_PROGRESS_CACHE[case_id] = normalize_progress_state(
        {
            "breakdown_probability": clamp01(breakdown_probability),
            "referenced_evidence_ids": sorted(
                {str(eid).strip() for eid in (evidence_ids or []) if str(eid).strip()}
            ),
            "established_contradiction_ids": sorted(
                {str(cid).strip() for cid in (contradiction_ids or []) if str(cid).strip()}
            ),
            "stress_score": clamp01(stress_score),
            "cooperation_score": clamp01(cooperation_score),
            "cumulative_pressure": clamp01(cumulative_pressure),
            "fsm_state": norm(fsm_state) or DEFAULT_FSM_STATE,
            "last_raw_odds": float(last_raw_odds),
            "last_sue_impact": float(last_sue_impact),
            "statement_collapse_stage": max(0, min(5, int(statement_collapse_stage or 0))),
            "core_fact_exposed": bool(
                max(0, min(5, int(statement_collapse_stage or 0))) >= 5
                or clamp01(breakdown_probability) >= BREAKDOWN_EXPOSURE_THRESHOLD
            ),
            "pad_state": _normalize_pad_state_blob(pad_state or {}),
            "final_psychological_reaction": norm(final_psychological_reaction),
            "selected_personality": _normalize_selected_personality_blob(selected_personality or {}),
            "statement_records": _normalize_statement_records(statement_records or []),
            "submitted_judgment": _normalize_submitted_judgment(submitted_judgment or {}),
            "final_psychological_report": _normalize_final_report_blob(final_psychological_report or {}),
            "turn_count": max(0, int(turn_count or 0)),
        }
    )

def _persist_progress_snapshot(case_id: str, progress_state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_progress_state(progress_state or {})
    _store_progress_state(
        case_id,
        normalized.get("breakdown_probability", 0.0),
        normalized.get("referenced_evidence_ids", []),
        normalized.get("established_contradiction_ids", []),
        normalized.get("stress_score", 0.0),
        normalized.get("cooperation_score", DEFAULT_COOPERATION_SCORE),
        normalized.get("cumulative_pressure", 0.0),
        normalized.get("fsm_state", DEFAULT_FSM_STATE),
        normalized.get("last_raw_odds", 0.0),
        normalized.get("last_sue_impact", 0.0),
        normalized.get("turn_count", 0),
        normalized.get("statement_collapse_stage", 0),
        normalized.get("pad_state", {}),
        normalized.get("final_psychological_reaction", ""),
        normalized.get("selected_personality", {}),
        normalized.get("statement_records", []),
        normalized.get("submitted_judgment", {}),
        normalized.get("final_psychological_report", {}),
    )
    return _get_progress_state(case_id)

def _slot_value_tokens(value: str) -> List[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", norm(value).lower())

SLOT_EXTRACTION_MIN_SCORE = 0.58

PLACE_TEXT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("물류창고", "창고"),
    ("창고입구", "창고근처"),
    ("창고앞", "창고근처"),
    ("창고앞쪽", "창고근처"),
    ("창고주변", "창고근처"),
    ("창고인근", "창고근처"),
    ("창고부근", "창고근처"),
    ("창고옆", "창고근처"),
    ("창고쪽", "창고근처"),
    ("창고뒤쪽", "창고뒤편"),
    ("회사입구", "회사근처"),
    ("회사앞", "회사근처"),
    ("집앞", "집근처"),
)

WEAPON_CANONICAL_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("칼", ("칼", "식칼", "주방칼", "과도", "회칼", "knife", "나이프")),
    ("망치", ("망치", "해머", "hammer")),
    ("둔기", ("둔기", "쇠파이프", "파이프", "렌치")),
)

RELATION_CANONICAL_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "직장동료",
        (
            "직장동료",
            "회사동료",
            "동료직원",
            "회사사람",
            "직장사람",
            "같이일하던사람",
            "같이일하던분",
            "같은회사사람",
        ),
    ),
    ("전연인", ("전연인", "옛연인", "전애인", "전남자친구", "전여자친구", "헤어진사이")),
    ("친구", ("친구", "지인", "아는사람")),
    ("가족", ("가족", "친척", "혈연")),
)

PLACE_CLAIM_KEYWORDS = (
    "집",
    "회사",
    "사무실",
    "창고",
    "항만",
    "주차장",
    "골목",
    "거리",
    "건물",
    "옥상",
    "현장",
    "매장",
    "공원",
)

def _canonicalize_with_rules(
    text: str,
    rules: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> str:
    for canonical, patterns in rules:
        if any(pattern in text for pattern in patterns):
            return canonical
    return text

def _normalize_place_text(value: str) -> str:
    normalized = norm_for_match(value)
    if not normalized:
        return ""
    for source, replacement in PLACE_TEXT_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    normalized = normalized.replace("인근", "근처")
    normalized = normalized.replace("부근", "근처")
    normalized = normalized.replace("주변", "근처")
    normalized = normalized.replace("뒤쪽", "뒤편")
    normalized = re.sub(r"([0-9A-Za-z가-힣]+)(?:앞|입구|옆)", r"\1근처", normalized)
    normalized = re.sub(r"(근처)+", "근처", normalized)
    return normalized

def _normalize_slot_value_for_slot(value: str, target_slot: str = "") -> str:
    text = norm(value)
    if not text:
        return ""

    time_match = re.search(r"(?P<hour>\d{1,2})\s*(?:시|:)\s*(?P<minute>\d{1,2})?", text)
    if time_match:
        hour = int(time_match.group("hour"))
        minute = int(time_match.group("minute") or 0)
        return f"{hour:02d}:{minute:02d}"

    lowered = text.lower()
    if lowered in {"yes", "y", "true", "1", "네", "예"}:
        return "yes"
    if lowered in {"no", "n", "false", "0", "아니요", "아니오"}:
        return "no"

    normalized = norm_for_match(text)
    if target_slot in {"crime_place", "last_seen_place", "alibi_claim"}:
        return _normalize_place_text(text)
    if target_slot == "weapon":
        return _canonicalize_with_rules(normalized, WEAPON_CANONICAL_RULES)
    if target_slot == "victim_relation":
        return _canonicalize_with_rules(normalized, RELATION_CANONICAL_RULES)
    return normalized

def normalize_slot_value(value: str) -> str:
    return _normalize_slot_value_for_slot(value)

def _slot_values_match(claimed_value: str, truth_value: str, target_slot: str = "") -> bool:
    claimed_norm = _normalize_slot_value_for_slot(claimed_value, target_slot)
    truth_norm = _normalize_slot_value_for_slot(truth_value, target_slot)
    if not claimed_norm or not truth_norm:
        return False
    if claimed_norm == truth_norm:
        return True
    if truth_norm in claimed_norm or claimed_norm in truth_norm:
        return True

    claimed_tokens = set(_slot_value_tokens(claimed_norm))
    truth_tokens = set(_slot_value_tokens(truth_norm))
    if not claimed_tokens or not truth_tokens:
        return False

    overlap = len(claimed_tokens & truth_tokens)
    coverage = overlap / max(1, min(len(claimed_tokens), len(truth_tokens)))
    return coverage >= 0.7

def _extract_boolean_claim(answer: str) -> str:
    text = norm(answer)
    negative_patterns = (
        "안 만났",
        "만난 적 없",
        "본 적 없",
        "접촉한 적 없",
        "안 봤",
    )
    positive_patterns = (
        "만났",
        "봤",
        "마주쳤",
        "접촉했",
        "통화했",
    )
    if any(pattern in text for pattern in negative_patterns):
        return "no"
    if any(pattern in text for pattern in positive_patterns):
        return "yes"
    return ""

def _extract_time_like_value(text: str) -> str:
    match = re.search(r"(?P<hour>\d{1,2})\s*(?:시|:)\s*(?P<minute>\d{1,2})?", norm(text))
    if not match:
        return ""
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    return f"{hour:02d}:{minute:02d}"

def _extract_place_like_claim(answer: str) -> Tuple[str, float]:
    text = norm(answer)
    if not text:
        return "", 0.0

    patterns = (
        r"([0-9A-Za-z가-힣 ]{0,24}(?:근처|앞|앞쪽|뒤편|뒤쪽|뒤|옆|입구|인근|부근))\s*(?:에|에서)",
        r"([0-9A-Za-z가-힣 ]{0,24}(?:집|회사|사무실|창고|항만|주차장|골목|거리|건물|옥상|현장|매장|공원))\s*(?:에|에서)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        place = norm(match.group(1))
        place = re.sub(r"^(그쪽|저쪽|거기|그 근처|저 근처)\s*", "", place)
        if place:
            return place, 0.82

    normalized = _normalize_place_text(text)
    if any(keyword in normalized for keyword in PLACE_CLAIM_KEYWORDS):
        return text, 0.7
    return "", 0.0

def _extract_relation_like_claim(answer: str) -> Tuple[str, float]:
    normalized = _normalize_slot_value_for_slot(answer, "victim_relation")
    if normalized and normalized in {canonical for canonical, _patterns in RELATION_CANONICAL_RULES}:
        return normalized, 0.85

    match = re.search(r"([0-9A-Za-z가-힣 ]{1,24}(?:동료|친구|가족|지인|연인|친척|사람))", norm(answer))
    if match:
        return norm(match.group(1)), 0.72
    return "", 0.0

def _extract_weapon_like_claim(answer: str) -> Tuple[str, float]:
    normalized = _normalize_slot_value_for_slot(answer, "weapon")
    if normalized and normalized in {canonical for canonical, _patterns in WEAPON_CANONICAL_RULES}:
        return normalized, 0.9
    return "", 0.0

def _slot_candidate_values(target_slot: str, case_data: Optional[Dict[str, Any]]) -> List[str]:
    if not case_data:
        return []

    truth_slots = _case_truth_slots(case_data)
    candidates: List[str] = []
    related_slots = {target_slot}

    if target_slot and truth_slots.get(target_slot):
        candidates.append(truth_slots[target_slot])

    overview = case_data.get("overview", {}) or {}
    suspect = case_data.get("suspect", {}) or {}
    if target_slot == "crime_time":
        candidates.append(str(overview.get("time", "")))
    elif target_slot == "crime_place":
        candidates.append(str(overview.get("place", "")))
    elif target_slot == "victim_relation":
        candidates.append(str(suspect.get("relation", "")))
    elif target_slot == "alibi_claim":
        related_slots |= {"crime_place", "crime_time", "last_seen_place", "met_victim_that_day"}
    elif target_slot == "last_seen_place":
        related_slots.add("crime_place")

    if target_slot in {"alibi_claim", "crime_place", "crime_time", "last_seen_place"}:
        candidates.append(str(case_data.get("false_statement", "")))
    for slot_name in related_slots:
        candidates.append(truth_slots.get(slot_name, ""))
    for contradiction in _case_contradictions(case_data):
        if target_slot and norm(contradiction.get("slot", "")) == target_slot:
            candidates.append(str(contradiction.get("truth_value", "")))

    return uniq_strings([norm(candidate) for candidate in candidates if norm(candidate)])

def _pick_best_slot_candidate_with_score(
    text: str,
    candidates: List[str],
    target_slot: str,
) -> Tuple[str, float]:
    text_norm = _normalize_slot_value_for_slot(text, target_slot)
    text_tokens = set(_slot_value_tokens(text_norm))

    best_candidate = ""
    best_score = 0.0
    for candidate in candidates:
        candidate = norm(candidate)
        if not candidate:
            continue
        candidate_norm = _normalize_slot_value_for_slot(candidate, target_slot)
        if not candidate_norm:
            continue
        if candidate_norm == text_norm:
            return candidate, 1.0
        if candidate_norm and candidate_norm in text_norm:
            score = 0.96
            if score > best_score:
                best_score = score
                best_candidate = candidate
            continue

        candidate_tokens = set(_slot_value_tokens(candidate_norm))
        if not candidate_tokens:
            continue
        overlap = len(candidate_tokens & text_tokens)
        if overlap <= 0:
            continue
        coverage = overlap / max(1, len(candidate_tokens))
        precision = overlap / max(1, len(text_tokens))
        score = (coverage * 0.8) + (precision * 0.2)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate, best_score

def _extract_claimed_slot_value_with_confidence(
    answer: str,
    target_slot: str,
    case_data: Optional[Dict[str, Any]],
) -> Tuple[str, float]:
    if not target_slot or not answer:
        return "", 0.0

    text = norm(answer)
    if not text:
        return "", 0.0

    if target_slot == "met_victim_that_day":
        claim = _extract_boolean_claim(text)
        return (claim, 0.95) if claim else ("", 0.0)

    if target_slot == "crime_time":
        claim = _extract_time_like_value(text)
        if claim:
            return claim, 1.0
    elif target_slot in {"crime_place", "last_seen_place", "alibi_claim"}:
        claim, confidence = _extract_place_like_claim(text)
        if confidence >= SLOT_EXTRACTION_MIN_SCORE:
            return claim, confidence
    elif target_slot == "victim_relation":
        claim, confidence = _extract_relation_like_claim(text)
        if confidence >= SLOT_EXTRACTION_MIN_SCORE:
            return claim, confidence
    elif target_slot == "weapon":
        claim, confidence = _extract_weapon_like_claim(text)
        if confidence >= SLOT_EXTRACTION_MIN_SCORE:
            return claim, confidence

    candidate, score = _pick_best_slot_candidate_with_score(
        text,
        _slot_candidate_values(target_slot, case_data),
        target_slot,
    )
    if score < SLOT_EXTRACTION_MIN_SCORE:
        return "", score
    return candidate, score

def extract_claimed_slot_value(answer: str, target_slot: str, case_data: Optional[Dict[str, Any]]) -> str:
    claimed_value, _confidence = _extract_claimed_slot_value_with_confidence(
        answer,
        target_slot,
        case_data,
    )
    return claimed_value

def _contradiction_requires_evidence(
    target_slot: str,
    contradiction_type: str,
    related_evidence: set,
    mentioned_evidence_set: set,
) -> bool:
    if not contradiction_type:
        return bool(related_evidence) and not (mentioned_evidence_set & related_evidence)
    if contradiction_type in {"claim_vs_evidence", "alibi_mismatch", "timeline_mismatch"}:
        if not related_evidence or not (mentioned_evidence_set & related_evidence):
            return True
    if contradiction_type == "claim_vs_truth":
        if related_evidence:
            return not (mentioned_evidence_set & related_evidence)
        return target_slot not in {"crime_time", "crime_place"}
    if contradiction_type == "timeline_mismatch":
        if target_slot not in {"crime_time", "crime_place", "last_seen_place"}:
            return True
        return False
    if contradiction_type == "alibi_mismatch":
        if target_slot != "alibi_claim":
            return True
        return False
    if contradiction_type == "claim_vs_evidence":
        return False
    return False

def detect_contradictions_from_slot(
    target_slot: str,
    claimed_value: str,
    case_data: Optional[Dict[str, Any]],
    mentioned_evidence_ids: List[str],
) -> List[str]:
    if not case_data or not target_slot or not claimed_value:
        return []

    truth_slots = _case_truth_slots(case_data)
    slot_truth_value = truth_slots.get(target_slot, "")
    slot_truth_matches = bool(
        slot_truth_value and _slot_values_match(claimed_value, slot_truth_value, target_slot)
    )

    mentioned_evidence_set = {
        str(evidence_id).strip()
        for evidence_id in (mentioned_evidence_ids or [])
        if str(evidence_id).strip()
    }
    allowed_slots = {target_slot}
    if target_slot == "alibi_claim":
        allowed_slots |= {"crime_place", "crime_time", "last_seen_place"}
    elif target_slot == "last_seen_place":
        allowed_slots |= {"crime_place"}

    contradiction_candidates: List[Tuple[int, str]] = []
    for contradiction in _case_contradictions(case_data):
        contradiction_id = norm(contradiction.get("id", ""))
        contradiction_slot = norm(contradiction.get("slot", ""))
        contradiction_truth = norm(contradiction.get("truth_value", "")) or slot_truth_value
        contradiction_type = norm(contradiction.get("contradiction_type", ""))
        related_evidence = {
            str(evidence_id).strip()
            for evidence_id in (contradiction.get("related_evidence", []) or [])
            if str(evidence_id).strip()
        }

        if not contradiction_id:
            continue
        if contradiction_slot and contradiction_slot not in allowed_slots:
            continue
        if not contradiction_truth:
            continue
        if contradiction_truth and _slot_values_match(
            claimed_value,
            contradiction_truth,
            contradiction_slot or target_slot,
        ):
            continue
        if _contradiction_requires_evidence(
            target_slot,
            contradiction_type,
            related_evidence,
            mentioned_evidence_set,
        ):
            continue

        score = 0
        if contradiction_slot == target_slot:
            score += 4
        elif contradiction_slot in allowed_slots:
            score += 1
        if mentioned_evidence_set & related_evidence:
            score += 2
        if contradiction_type == "alibi_mismatch" and target_slot == "alibi_claim":
            score += 2
        elif contradiction_type == "timeline_mismatch" and target_slot in {"crime_time", "crime_place", "last_seen_place"}:
            score += 1
        contradiction_candidates.append((score, contradiction_id))

    if slot_truth_matches and not contradiction_candidates:
        return []
    contradiction_candidates.sort(key=lambda item: (-item[0], item[1]))
    contradiction_ids = uniq_strings([contradiction_id for _score, contradiction_id in contradiction_candidates])
    return contradiction_ids[:1]

def analyze_interrogation_turn_rule_based(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    suspect_text: str,
    question_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    analysis = question_analysis or llm_evaluate_interrogation(case_data, history, user_text)
    target_slot = analysis.get("target_slot")
    mentioned_evidence_ids = list(analysis.get("mentioned_evidence_ids", []) or [])
    truth_slots = _case_truth_slots(case_data)
    claimed_value = ""
    claimed_value_confidence = 0.0
    contradiction_ids: List[str] = []
    if target_slot:
        claimed_value, claimed_value_confidence = _extract_claimed_slot_value_with_confidence(
            suspect_text,
            target_slot,
            case_data,
        )
        if claimed_value and claimed_value_confidence >= SLOT_EXTRACTION_MIN_SCORE:
            contradiction_ids = detect_contradictions_from_slot(
                target_slot,
                claimed_value,
                case_data,
                mentioned_evidence_ids,
            )
    hard_contradiction_ids = uniq_strings(contradiction_ids)
    return {
        "question_analysis": analysis,
        "target_slot": target_slot,
        "claimed_value": claimed_value,
        "claimed_value_confidence": round(claimed_value_confidence, 3),
        "truth_value": truth_slots.get(target_slot or "", ""),
        "mentioned_evidence_ids": mentioned_evidence_ids,
        "hard_contradiction_ids": hard_contradiction_ids,
        "detected_contradiction_ids": hard_contradiction_ids,
        "soft_dialogue_contradiction": False,
    }

def build_case_context(
    case_data: Optional[Dict[str, Any]],
    include_hidden_truth: bool = False,
) -> str:
    if not case_data:
        return "No case data.\n"

    overview = case_data.get("overview", {}) or {}
    suspect = case_data.get("suspect", {}) or {}
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data.get("suspect_profile"), dict) else {}
    personality = _case_personality(case_data)
    mental_state = suspect_profile.get("mental_state", {}) if isinstance(suspect_profile.get("mental_state"), dict) else {}
    evidences = _case_evidences(case_data)

    evidence_lines = [
        f"- {e.get('name', e.get('id', ''))}: {e.get('description', '')}".strip()
        for e in evidences
    ]

    return (
        "=== Internal case brief ===\n"
        + f"[Overview] time:{overview.get('time', '')} place:{overview.get('place', '')} type:{overview.get('type', '')}\n"
        + f"[Motive] {case_data.get('motive', '')}\n"
        + (
            f"[Hidden crime flow summary] {case_data.get('crime_flow', '')}\n"
            if include_hidden_truth else ""
        )
        + f"[Suspect] name:{suspect.get('name', '')} age:{suspect.get('age', '')} job:{suspect.get('job', '')} relation:{suspect.get('relation', '')}\n"
        + f"[Suspect role] {suspect_profile.get('case_role', '')}\n"
        + (
            "[Personality] "
            + " ".join(
                f"{trait}:{float(personality.get(trait, 0.5)):.2f}"
                for trait in BIG_FIVE_TRAITS
            )
            + "\n"
        )
        + (
            "[Baseline PAD] "
            + " ".join(
                f"{field}:{float(mental_state.get(field, 0.5)):.2f}"
                for field in PAD_STATE_FIELDS
            )
            + "\n"
        )
        + f"[Default false statement] {case_data.get('false_statement', '')}\n"
        + "[Evidence list]\n"
        + ("\n".join(evidence_lines) if evidence_lines else "- none")
        + "\n[Response boundaries]\n"
        + (
            "- Use the hidden crime flow and suspect profile only to stay internally consistent.\n"
            if include_hidden_truth else
            "- Stay consistent with the suspect profile and default false statement.\n"
        )
        + "- Never volunteer hidden facts or unseen evidence unless the detective directly asks that point.\n"
        + "- When pressure rises, the story may wobble, but do not reveal core facts too early.\n"
        + "\n======================\n"
    )

def evaluate_interrogation_progress_v3(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    interrogation_signal: Optional[Dict[str, Any]] = None,
    prior_progress: Optional[Dict[str, Any]] = None,
    contradiction_ids: Optional[List[str]] = None,
    dialogue_contradiction_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not case_data:
        return {
            "pressure_delta": 0.0,
            "breakdown_probability": 0.0,
            "cumulative_evidence_ids": [],
            "cumulative_contradiction_ids": [],
            "hard_contradiction_ids": [],
            "soft_dialogue_contradiction": False,
            "soft_dialogue_severity": "none",
            "stress_score": 0.0,
            "cooperation_score": DEFAULT_COOPERATION_SCORE,
            "cumulative_pressure": 0.0,
            "turn_pressure_gain": 0.0,
            "defense_intelligence": 0.65,
            "latest_sue_impact": 0.0,
            "raw_odds": 0.0,
            "player_intent": "Neutral",
            "fsm_state": DEFAULT_FSM_STATE,
            "statement_collapse_stage": 0,
            "statement_collapse_label": _statement_collapse_label(0),
            "core_fact_exposed": False,
            "pad_state": _default_mental_state(),
            "final_psychological_reaction": "",
            "personality_response_factors": {},
            "personality_response_breakdown": {},
            "pressure_components": {
                "evidence": 0.0,
                "hard_contradiction": 0.0,
                "soft_dialogue": 0.0,
                "pressure_level": 0.0,
                "progression": 0.0,
                "sue": 0.0,
            },
        }

    signal = interrogation_signal or llm_evaluate_interrogation(case_data, history, user_text)
    prior_progress = prior_progress or _empty_progress_state()

    prior_stress_score = clamp01(prior_progress.get("stress_score", 0.0))
    prior_cooperation_score = clamp01(
        prior_progress.get("cooperation_score", DEFAULT_COOPERATION_SCORE)
    )
    prior_cumulative_pressure = clamp01(
        prior_progress.get("cumulative_pressure", prior_progress.get("stress_score", 0.0))
    )
    prior_statement_collapse_stage = max(
        0,
        min(5, int(safe_float(prior_progress.get("statement_collapse_stage", 0), 0.0))),
    )
    baseline_pad_state = _case_baseline_pad_state(case_data)
    prior_pad_state = _normalize_pad_state_blob(prior_progress.get("pad_state", baseline_pad_state))
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

    current_evidence_ids = {
        str(eid).strip()
        for eid in (signal.get("mentioned_evidence_ids", []) or [])
        if str(eid).strip()
    }
    current_contradiction_ids = {
        str(cid).strip()
        for cid in (contradiction_ids or [])
        if str(cid).strip()
    }

    cumulative_evidence_ids = prior_evidence_ids | current_evidence_ids
    cumulative_contradiction_ids = prior_contradiction_ids | current_contradiction_ids

    new_evidence_ids = cumulative_evidence_ids - prior_evidence_ids
    repeated_evidence_ids = current_evidence_ids & prior_evidence_ids
    new_contradiction_ids = cumulative_contradiction_ids - prior_contradiction_ids
    repeated_contradiction_ids = current_contradiction_ids & prior_contradiction_ids

    pressure_level = str(signal.get("pressure_level", "none")).strip().lower()
    if pressure_level not in PRESSURE_LEVEL_BONUS:
        pressure_level = "none"

    player_intent = _infer_player_intent(signal)
    personality_response_factors = _calculate_personality_response_factors(
        case_data,
        player_intent,
    )
    repeated_question = detect_repeat(history, user_text)

    dialogue_pressure_bonus, dialogue_breakdown_bonus, soft_dialogue_contradiction = (
        _dialogue_contradiction_bonus_values(dialogue_contradiction_signal)
    )
    evidence_pressure = (
        NEW_EVIDENCE_PRESSURE_DELTA * len(new_evidence_ids)
        + REPEATED_EVIDENCE_PRESSURE_DELTA * len(repeated_evidence_ids)
    )
    hard_contradiction_pressure = (
        NEW_CONTRADICTION_PRESSURE_DELTA * len(new_contradiction_ids)
        + REPEATED_CONTRADICTION_PRESSURE_DELTA * len(repeated_contradiction_ids)
    )
    pressure_level_bonus = PRESSURE_LEVEL_BONUS.get(pressure_level, 0.0)
    pressure_components = {
        "evidence": evidence_pressure * personality_response_factors["pressure_multiplier"],
        "hard_contradiction": hard_contradiction_pressure * personality_response_factors["pressure_multiplier"],
        "soft_dialogue": dialogue_pressure_bonus * max(0.85, min(1.35, personality_response_factors["arousal_sensitivity"])),
        "pressure_level": pressure_level_bonus * personality_response_factors["pressure_multiplier"],
    }
    pressure_delta = (
        pressure_components["evidence"]
        + pressure_components["hard_contradiction"]
        + pressure_components["soft_dialogue"]
        + pressure_components["pressure_level"]
    )

    if not user_text or is_too_ambiguous(user_text):
        pressure_components = {key: 0.0 for key in pressure_components}
        pressure_delta = 0.0
    elif repeated_question:
        pressure_components = {
            key: value * 0.35 for key, value in pressure_components.items()
        }
        pressure_delta *= 0.35

    if pressure_delta > MAX_TURN_PRESSURE_DELTA and pressure_delta > 0:
        cap_scale = MAX_TURN_PRESSURE_DELTA / pressure_delta
        pressure_components = {
            key: value * cap_scale for key, value in pressure_components.items()
        }
        pressure_delta = MAX_TURN_PRESSURE_DELTA

    pressure_delta = clamp01(pressure_delta)
    stress_score = _update_stress_score(
        prior_stress_score,
        pressure_delta,
        pressure_level,
        current_evidence_ids,
        current_contradiction_ids,
        history,
        user_text,
    )
    stress_score = _apply_personality_scaled_delta(
        prior_stress_score,
        stress_score,
        personality_response_factors["stress_multiplier"],
    )
    cooperation_score = _update_cooperation_score(
        prior_cooperation_score,
        player_intent,
        pressure_delta,
        len(new_evidence_ids),
        len(new_contradiction_ids),
        history,
        user_text,
    )
    cooperation_score = clamp01(
        cooperation_score + personality_response_factors["cooperation_shift"]
    )
    pad_state = _update_pad_state(
        prior_pad_state,
        case_data,
        player_intent,
        pressure_delta,
        len(new_evidence_ids),
        len(new_contradiction_ids),
        soft_dialogue_contradiction,
        repeated_question,
    )
    defense_intelligence = _infer_defense_intelligence(case_data)
    core = _build_interrogation_core(case_data)
    latest_sue_impact = _calculate_latest_sue_impact(
        case_data,
        list(current_evidence_ids),
        list(current_contradiction_ids),
        core,
    )
    progression_pressure = 0.0
    if signal.get("intent") in {"ask_time", "ask_place", "ask_alibi", "ask_action"}:
        progression_pressure += CORE_QUESTION_PRESSURE_BONUS
    if new_evidence_ids:
        progression_pressure += NEW_EVIDENCE_PRESSURE_PROGRESS_BONUS
    if new_contradiction_ids:
        progression_pressure += NEW_CONTRADICTION_PRESSURE_PROGRESS_BONUS
    if soft_dialogue_contradiction:
        progression_pressure += SOFT_DIALOGUE_PRESSURE_PROGRESS_BONUS

    progression_pressure *= personality_response_factors["direct_bonus_multiplier"]
    sue_pressure = 0.0
    if current_contradiction_ids and latest_sue_impact > 0.0:
        sue_pressure = min(0.08, latest_sue_impact * SUE_PRESSURE_BONUS_SCALE)

    if repeated_question:
        progression_pressure *= 0.35
        sue_pressure *= 0.5

    pressure_components["progression"] = progression_pressure
    pressure_components["sue"] = sue_pressure
    turn_cumulative_pressure_gain = min(
        MAX_TURN_CUMULATIVE_PRESSURE_GAIN,
        pressure_delta + progression_pressure + sue_pressure + dialogue_breakdown_bonus,
    )
    cumulative_pressure = clamp01(prior_cumulative_pressure + turn_cumulative_pressure_gain)
    raw_odds, breakdown_probability = core.calculate_breakdown_probability(
        cumulative_pressure
    )
    statement_collapse_stage = _calculate_statement_collapse_stage(
        prior_statement_collapse_stage,
        cumulative_pressure,
        breakdown_probability,
        len(cumulative_contradiction_ids),
        len(current_contradiction_ids),
        soft_dialogue_contradiction,
        pad_state,
        case_data,
    )
    statement_collapse_stage = max(
        statement_collapse_stage,
        _soft_dialogue_stage_floor(
            dialogue_contradiction_signal,
            cumulative_pressure,
            pad_state,
            repeated_question,
        ),
    )
    fsm_state = core.evaluate_fsm_state(
        breakdown_probability,
        len(cumulative_contradiction_ids),
        player_intent,
        cooperation_score,
        latest_sue_impact,
    )
    if statement_collapse_stage >= 4 and fsm_state != EXPOSURE_FSM_STATE:
        fsm_state = "Pressured / Shaken"
    elif statement_collapse_stage >= 2 and fsm_state == DEFAULT_FSM_STATE:
        fsm_state = "Pressured / Shaken"
    final_psychological_reaction = ""
    core_fact_exposed = bool(
        statement_collapse_stage >= 5 or breakdown_probability >= BREAKDOWN_EXPOSURE_THRESHOLD
    )
    if core_fact_exposed:
        statement_collapse_stage = 5
        fsm_state = EXPOSURE_FSM_STATE
        final_psychological_reaction = _generate_final_psychological_reaction(
            case_data,
            statement_collapse_stage,
            pad_state,
            True,
        )

    return {
        "pressure_delta": turn_cumulative_pressure_gain,
        "breakdown_probability": breakdown_probability,
        "cumulative_evidence_ids": sorted(cumulative_evidence_ids),
        "cumulative_contradiction_ids": sorted(cumulative_contradiction_ids),
        "hard_contradiction_ids": sorted(current_contradiction_ids),
        "soft_dialogue_contradiction": soft_dialogue_contradiction,
        "soft_dialogue_severity": str(
            (dialogue_contradiction_signal or {}).get("severity", "none")
        ).strip().lower() if isinstance(dialogue_contradiction_signal, dict) else "none",
        "stress_score": stress_score,
        "cooperation_score": cooperation_score,
        "cumulative_pressure": cumulative_pressure,
        "turn_pressure_gain": turn_cumulative_pressure_gain,
        "defense_intelligence": defense_intelligence,
        "latest_sue_impact": latest_sue_impact,
        "raw_odds": raw_odds,
        "player_intent": player_intent,
        "fsm_state": fsm_state,
        "statement_collapse_stage": statement_collapse_stage,
        "statement_collapse_label": _statement_collapse_label(statement_collapse_stage),
        "core_fact_exposed": core_fact_exposed,
        "pad_state": pad_state,
        "final_psychological_reaction": final_psychological_reaction,
        "personality_response_factors": {
            key: round(value, 4)
            for key, value in personality_response_factors.items()
        },
        "personality_response_breakdown": _build_personality_response_breakdown(
            case_data,
            player_intent,
        ),
        "pressure_components": {
            "evidence": round(pressure_components["evidence"], 4),
            "hard_contradiction": round(pressure_components["hard_contradiction"], 4),
            "soft_dialogue": round(pressure_components["soft_dialogue"], 4),
            "pressure_level": round(pressure_components["pressure_level"], 4),
        },
    }

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
        msg = str(e).lower()
        if (
            "unsupported" in msg
            or "corrupted" in msg
            or "invalid_value" in msg
            or "audio file might be corrupted" in msg
            or "unsupported_format" in msg
            or "messages" in msg
        ):
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

BIG_FIVE_TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

PAD_STATE_FIELDS = (
    "pleasure",
    "arousal",
    "dominance",
)

STATEMENT_COLLAPSE_LABELS = {
    0: "안정적 부인",
    1: "경계",
    2: "흔들림",
    3: "부분 수정",
    4: "진술 붕괴",
    5: "핵심 사실 노출",
}

JUDGMENT_CHOICE_LABELS = {
    "principal": "주범",
    "accomplice_or_coverup": "공범 또는 은폐 가능성",
    "not_directly_involved": "범행과 직접 관련 없음",
}

JUDGMENT_CHOICE_ALIASES = {
    "principal": "principal",
    "주범": "principal",
    "main": "principal",
    "accomplice_or_coverup": "accomplice_or_coverup",
    "accomplice": "accomplice_or_coverup",
    "coverup": "accomplice_or_coverup",
    "공범": "accomplice_or_coverup",
    "은폐": "accomplice_or_coverup",
    "공범 또는 은폐 가능성": "accomplice_or_coverup",
    "not_directly_involved": "not_directly_involved",
    "not_involved": "not_directly_involved",
    "irrelevant": "not_directly_involved",
    "무관": "not_directly_involved",
    "범행과 직접 관련 없음": "not_directly_involved",
}

MAX_STATEMENT_RECORDS = 40
def _to_clamped_float(value: Any, default: float = 0.5) -> float:
    try:
        return clamp01(float(value))
    except Exception:
        return clamp01(default)

def _normalize_string_list(values: Any) -> List[str]:
    if isinstance(values, list):
        return uniq_strings([norm(str(value)) for value in values if norm(str(value))])
    if isinstance(values, str):
        text = norm(values)
        return [text] if text else []
    return []

def _default_personality() -> Dict[str, float]:
    return {trait: 0.5 for trait in BIG_FIVE_TRAITS}

def _default_mental_state() -> Dict[str, float]:
    return {field: 0.5 for field in PAD_STATE_FIELDS}

def _normalize_pad_state_blob(blob: Any) -> Dict[str, float]:
    raw = blob if isinstance(blob, dict) else {}
    normalized = _default_mental_state()
    for field in PAD_STATE_FIELDS:
        normalized[field] = _to_clamped_float(raw.get(field, normalized[field]), normalized[field])
    return normalized

def _normalize_personality_blob(blob: Any) -> Dict[str, float]:
    raw = blob if isinstance(blob, dict) else {}
    normalized = _default_personality()
    for trait in BIG_FIVE_TRAITS:
        normalized[trait] = _to_clamped_float(raw.get(trait, normalized[trait]), normalized[trait])
    return normalized

def _missing_big_five_traits(blob: Any) -> List[str]:
    raw = blob if isinstance(blob, dict) else {}
    return [trait for trait in BIG_FIVE_TRAITS if trait not in raw]

def _validate_selected_personality_payload(blob: Any) -> Tuple[Dict[str, float], List[str]]:
    missing_traits = _missing_big_five_traits(blob)
    if missing_traits:
        return {}, missing_traits
    return _normalize_personality_blob(blob), []

def _normalize_selected_personality_blob(blob: Any) -> Dict[str, float]:
    raw = blob if isinstance(blob, dict) else {}
    if _missing_big_five_traits(raw):
        return {}
    return _normalize_personality_blob(raw)

def _selected_personality_from_progress(progress_state: Optional[Dict[str, Any]]) -> Dict[str, float]:
    progress = progress_state if isinstance(progress_state, dict) else {}
    return _normalize_selected_personality_blob(
        progress.get("selected_personality", {})
    )

def _resolve_selected_personality(
    case_data: Optional[Dict[str, Any]],
    progress_state: Optional[Dict[str, Any]] = None,
    personality_blob: Any = None,
) -> Dict[str, float]:
    raw = personality_blob if isinstance(personality_blob, dict) else None
    if raw is None:
        return _selected_personality_from_progress(progress_state)
    if _missing_big_five_traits(raw):
        return {}
    return _normalize_personality_blob(raw)

def _normalize_statement_record(record: Any) -> Dict[str, Any]:
    raw = record if isinstance(record, dict) else {}
    try:
        turn_index = int(raw.get("turn_index", 0) or 0)
    except (TypeError, ValueError):
        turn_index = 0

    hard_contradiction_ids = raw.get("hard_contradiction_ids", [])
    if not isinstance(hard_contradiction_ids, list):
        hard_contradiction_ids = []

    question_category = norm(raw.get("question_category", ""))
    if question_category not in QUESTION_CATEGORY_LABELS:
        question_category = _classify_question_category(
            norm(raw.get("question_text", "")),
            {
                "intent": raw.get("question_intent", ""),
                "target_slot": raw.get("target_slot", ""),
                "pressure_level": raw.get("pressure_level", ""),
            },
            False,
        )["key"]

    return {
        "turn_index": max(0, turn_index),
        "question_text": norm(raw.get("question_text", "")),
        "question_intent": norm(raw.get("question_intent", "")),
        "question_category": question_category,
        "question_category_label": QUESTION_CATEGORY_LABELS.get(
            question_category,
            QUESTION_CATEGORY_LABELS["misc"],
        ),
        "target_slot": norm(raw.get("target_slot", "")),
        "pressure_level": norm(raw.get("pressure_level", "")),
        "npc_answer": norm(raw.get("npc_answer", "")),
        "core_claim": norm(raw.get("core_claim", "")),
        "claimed_value": norm(raw.get("claimed_value", "")),
        "hard_contradiction_ids": uniq_strings([norm(str(value)) for value in hard_contradiction_ids if norm(str(value))]),
        "soft_dialogue_contradiction": bool(raw.get("soft_dialogue_contradiction", False)),
        "pad_state": _normalize_pad_state_blob(raw.get("pad_state", {})),
        "statement_collapse_stage": max(
            0,
            min(5, int(safe_float(raw.get("statement_collapse_stage", 0), 0.0))),
        ),
        "cumulative_pressure": clamp01(raw.get("cumulative_pressure", 0.0)),
        "statement_collapse_label": norm(raw.get("statement_collapse_label", "")),
        "fsm_state": norm(raw.get("fsm_state", "")),
        "breakdown_probability": clamp01(
            raw.get("breakdown_probability", 0.0)
        ),
        "core_fact_exposed": bool(raw.get("core_fact_exposed", False)),
    }

def _normalize_statement_records(values: Any) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    records = [_normalize_statement_record(value) for value in values]
    return records[-MAX_STATEMENT_RECORDS:]

def _normalize_judgment_choice(value: Any) -> str:
    key = norm(str(value))
    return JUDGMENT_CHOICE_ALIASES.get(key, "")

def _normalize_submitted_judgment(blob: Any) -> Dict[str, Any]:
    raw = blob if isinstance(blob, dict) else {}
    choice_key = _normalize_judgment_choice(raw.get("choice_key") or raw.get("choice") or raw.get("label"))
    return {
        "choice_key": choice_key,
        "choice_label": JUDGMENT_CHOICE_LABELS.get(choice_key, ""),
        "notes": norm(raw.get("notes", "")),
        "turn_count": max(0, int(safe_float(raw.get("turn_count", 0), 0.0))),
    }

def _normalize_final_report_blob(blob: Any) -> Dict[str, Any]:
    raw = blob if isinstance(blob, dict) else {}
    return {
        "case_id": norm(raw.get("case_id", "")),
        "case_role": norm(raw.get("case_role", "")),
        "turn_count": max(0, int(safe_float(raw.get("turn_count", 0), 0.0))),
        "breakdown_probability": clamp01(
            raw.get("breakdown_probability", 0.0)
        ),
        "statement_collapse_stage": max(
            0,
            min(5, int(safe_float(raw.get("statement_collapse_stage", 0), 0.0))),
        ),
        "statement_collapse_label": norm(raw.get("statement_collapse_label", "")),
        "core_fact_exposed": bool(raw.get("core_fact_exposed", False)),
        "pad_state": _normalize_pad_state_blob(raw.get("pad_state", {})),
        "final_psychological_reaction": norm(raw.get("final_psychological_reaction", "")),
        "hard_contradiction_count": max(0, int(safe_float(raw.get("hard_contradiction_count", 0), 0.0))),
        "evidence_reference_count": max(0, int(safe_float(raw.get("evidence_reference_count", 0), 0.0))),
        "cooperation_score": clamp01(raw.get("cooperation_score", DEFAULT_COOPERATION_SCORE)),
        "stress_score": clamp01(raw.get("stress_score", 0.0)),
        "personality": _normalize_personality_blob(raw.get("personality", {})),
        "summary_tags": _normalize_string_list(raw.get("summary_tags", [])),
        "judgment_ready": bool(raw.get("judgment_ready", False)),
    }

def _case_default_personality(case_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data, dict) else {}
    personality = {}
    if isinstance(suspect_profile, dict):
        if isinstance(suspect_profile.get("default_personality"), dict):
            personality = suspect_profile.get("default_personality", {})
        elif isinstance(suspect_profile.get("personality"), dict):
            personality = suspect_profile.get("personality", {})
    normalized = _default_personality()
    for trait in BIG_FIVE_TRAITS:
        normalized[trait] = _to_clamped_float(personality.get(trait, normalized[trait]), normalized[trait])
    return normalized

def _case_personality(case_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data, dict) else {}
    selected = suspect_profile.get("selected_personality", {}) if isinstance(suspect_profile, dict) else {}
    normalized_selected = _normalize_selected_personality_blob(selected)
    if normalized_selected:
        return normalized_selected
    return _case_default_personality(case_data)

def _case_baseline_pad_state(case_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data, dict) else {}
    mental_state = suspect_profile.get("mental_state", {}) if isinstance(suspect_profile, dict) else {}
    return _normalize_pad_state_blob(mental_state)

def _apply_selected_personality_to_case(
    case_data: Optional[Dict[str, Any]],
    selected_personality: Optional[Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    if not case_data:
        return case_data
    selected = _normalize_personality_blob(selected_personality or {})
    if not selected_personality:
        return case_data

    effective_case = copy.deepcopy(case_data)
    suspect_profile = effective_case.get("suspect_profile", {})
    if not isinstance(suspect_profile, dict):
        suspect_profile = {}
        effective_case["suspect_profile"] = suspect_profile
    suspect_profile["selected_personality"] = selected
    return effective_case

def _effective_case_from_progress(
    case_data: Optional[Dict[str, Any]],
    progress_state: Optional[Dict[str, Any]] = None,
    selected_personality: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    selected = selected_personality
    if selected is None and isinstance(progress_state, dict):
        selected = _selected_personality_from_progress(progress_state)
    return _apply_selected_personality_to_case(case_data, selected)

def _extract_core_claim_text(suspect_text: str, claimed_value: str = "") -> str:
    claim = norm(claimed_value)
    if claim:
        return claim
    text = trim_to_1_3_sentences(suspect_text)
    if not text:
        return ""
    parts = re.split(r"(?<=[\.!\?。！？])\s+", text)
    return norm(parts[0] if parts else text)

def _case_result_choice_key(case_data: Optional[Dict[str, Any]]) -> str:
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data, dict) else {}
    role = norm(suspect_profile.get("case_role", ""))
    if any(keyword in role for keyword in ("주범", "실행범", "범인")):
        return "principal"
    if any(keyword in role for keyword in ("공범", "은폐", "교사", "방조")):
        return "accomplice_or_coverup"
    return "not_directly_involved"

def _build_final_psychological_report(
    case_data: Optional[Dict[str, Any]],
    progress_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    progress = normalize_progress_state(progress_state or {})
    effective_case = _effective_case_from_progress(case_data, progress)
    personality = _case_personality(effective_case)
    stage = max(0, min(5, int(safe_float(progress.get("statement_collapse_stage", 0), 0.0))))
    summary_tags: List[str] = []
    if stage >= 4:
        summary_tags.append("statement_breakdown")
    elif stage >= 2:
        summary_tags.append("statement_wobble")
    else:
        summary_tags.append("stable_denial")
    if progress.get("cumulative_pressure", 0.0) >= 0.7:
        summary_tags.append("high_interrogation_pressure")
    if progress.get("core_fact_exposed", False):
        summary_tags.append("core_fact_exposed")
    if progress.get("pad_state", {}).get("arousal", 0.0) >= 0.7:
        summary_tags.append("high_arousal")
    if progress.get("pad_state", {}).get("dominance", 1.0) <= 0.35:
        summary_tags.append("low_dominance")

    return _normalize_final_report_blob(
        {
            "case_id": norm((effective_case or {}).get("case_id", "")),
            "case_role": norm(((effective_case or {}).get("suspect_profile", {}) or {}).get("case_role", "")),
            "turn_count": progress.get("turn_count", 0),
            "breakdown_probability": progress.get(
                "breakdown_probability",
                0.0,
            ),
            "statement_collapse_stage": stage,
            "cumulative_pressure": progress.get("cumulative_pressure", 0.0),
            "statement_collapse_label": _statement_collapse_label(stage),
            "core_fact_exposed": bool(progress.get("core_fact_exposed", False) or stage >= 5),
            "pad_state": progress.get("pad_state", _default_mental_state()),
            "final_psychological_reaction": progress.get("final_psychological_reaction", ""),
            "hard_contradiction_count": len(progress.get("established_contradiction_ids", []) or []),
            "evidence_reference_count": len(progress.get("referenced_evidence_ids", []) or []),
            "cooperation_score": progress.get("cooperation_score", DEFAULT_COOPERATION_SCORE),
            "stress_score": progress.get("stress_score", 0.0),
            "personality": personality,
            "summary_tags": summary_tags,
            "judgment_ready": bool(
                progress.get("turn_count", 0) >= MAX_GAME_TURNS
                or progress.get("core_fact_exposed", False)
                or stage >= 4
            ),
        }
    )

def _build_statement_record(
    turn_index: int,
    user_text: str,
    question_analysis: Dict[str, Any],
    suspect_text: str,
    rule_based_turn: Dict[str, Any],
    progress_eval: Dict[str, Any],
    repeated_question: bool = False,
) -> Dict[str, Any]:
    stage = max(0, min(5, int(safe_float(progress_eval.get("statement_collapse_stage", 0), 0.0))))
    question_category = _classify_question_category(
        user_text,
        question_analysis,
        repeated_question,
    )
    return _normalize_statement_record(
        {
            "turn_index": turn_index,
            "question_text": user_text,
            "question_intent": question_analysis.get("intent", ""),
            "question_category": question_category["key"],
            "question_category_label": question_category["label"],
            "target_slot": question_analysis.get("target_slot", ""),
            "pressure_level": question_analysis.get("pressure_level", ""),
            "npc_answer": suspect_text,
            "core_claim": _extract_core_claim_text(
                suspect_text,
                rule_based_turn.get("claimed_value", ""),
            ),
            "claimed_value": rule_based_turn.get("claimed_value", ""),
            "hard_contradiction_ids": rule_based_turn.get("hard_contradiction_ids", []),
            "soft_dialogue_contradiction": rule_based_turn.get("soft_dialogue_contradiction", False),
            "pad_state": progress_eval.get("pad_state", {}),
            "statement_collapse_stage": stage,
            "statement_collapse_label": _statement_collapse_label(stage),
            "fsm_state": progress_eval.get("fsm_state", ""),
            "breakdown_probability": progress_eval.get(
                "breakdown_probability",
                0.0,
            ),
            "core_fact_exposed": progress_eval.get("core_fact_exposed", False),
        }
    )

def _statement_collapse_label(stage: int) -> str:
    return STATEMENT_COLLAPSE_LABELS.get(max(0, min(5, int(stage))), STATEMENT_COLLAPSE_LABELS[0])

def _calculate_personality_response_factors(
    case_data: Optional[Dict[str, Any]],
    player_intent: str,
) -> Dict[str, float]:
    personality = _case_personality(case_data)
    openness = personality["openness"]
    conscientiousness = personality["conscientiousness"]
    extraversion = personality["extraversion"]
    agreeableness = personality["agreeableness"]
    neuroticism = personality["neuroticism"]

    pressure_multiplier = 1.0 + (0.28 * neuroticism) - (0.18 * conscientiousness)
    if player_intent == "Rapport":
        pressure_multiplier -= 0.12 * agreeableness
    elif player_intent == "Intimidate":
        pressure_multiplier += 0.08 + (0.10 * neuroticism)
    elif player_intent == "Confront":
        pressure_multiplier += 0.05 + (0.08 * openness)

    return {
        "pressure_multiplier": max(0.65, min(1.65, pressure_multiplier)),
        "stress_multiplier": max(0.6, min(1.85, 0.82 + (0.85 * neuroticism) - (0.28 * conscientiousness))),
        "direct_bonus_multiplier": max(0.72, min(1.6, 0.92 + (0.34 * neuroticism) + (0.12 * agreeableness) - (0.24 * conscientiousness))),
        "cooperation_shift": max(-0.18, min(0.18, ((agreeableness - 0.5) * 0.18) + ((extraversion - 0.5) * 0.07) - ((neuroticism - 0.5) * 0.08))),
        "arousal_sensitivity": max(0.72, min(1.95, 0.88 + (0.92 * neuroticism) + (0.16 * openness))),
        "dominance_resistance": max(0.65, min(1.5, 0.86 + (0.58 * conscientiousness) - (0.18 * agreeableness))),
        "collapse_resistance": max(0.58, min(1.7, 0.88 + (0.96 * conscientiousness) - (0.34 * neuroticism))),
        "rapport_affinity": max(0.72, min(1.45, 0.88 + (0.62 * agreeableness))),
        "reply_length_bias": max(0.62, min(1.65, 0.70 + (0.95 * extraversion))),
        "rigidity_bias": max(0.65, min(1.55, 1.05 + (0.62 * conscientiousness) - (0.34 * openness))),
        "friendliness_bias": max(0.6, min(1.55, 0.78 + (0.92 * agreeableness) - (0.12 * neuroticism))),
        "volatility_bias": max(0.6, min(1.65, 0.80 + (0.78 * neuroticism) - (0.24 * conscientiousness))),
    }

def _build_personality_speaking_directives(case_data: Optional[Dict[str, Any]]) -> List[str]:
    personality = _case_personality(case_data)
    openness = personality["openness"]
    conscientiousness = personality["conscientiousness"]
    extraversion = personality["extraversion"]
    agreeableness = personality["agreeableness"]
    neuroticism = personality["neuroticism"]

    directives: List[str] = [
        "- Make the personality difference obvious on the surface of the reply, not only in hidden state.",
    ]

    if openness >= 0.7:
        directives.append("- High openness: reframe the situation more readily and reach for alternative angles or side explanations.")
    elif openness <= 0.3:
        directives.append("- Low openness: stay rigid and repetitive. Reuse the same denial frame and avoid new angles.")

    if conscientiousness >= 0.7:
        directives.append("- High conscientiousness: sound careful, controlled, and precise. Keep the wording organized and internally consistent.")
    elif conscientiousness <= 0.3:
        directives.append("- Low conscientiousness: allow looser wording, slight messiness, and small mid-sentence self-corrections.")

    if extraversion >= 0.75:
        directives.append("- High extraversion: be visibly more talkative. Usually give 2 or 3 spoken sentences, and add one short follow-up explanation after the direct answer.")
    elif extraversion <= 0.3:
        directives.append("- Low extraversion: be visibly clipped. Usually stop after 1 short sentence and avoid voluntary follow-up explanation.")

    if agreeableness >= 0.7:
        directives.append("- High agreeableness: sound softer, more cooperative, and more accommodating. Mild apologies or deference are acceptable.")
    elif agreeableness <= 0.3:
        directives.append("- Low agreeableness: sound curt, prickly, and resistant. Avoid apologetic softeners unless absolutely necessary.")

    if neuroticism >= 0.7:
        directives.append("- High neuroticism: show nervousness, hedging, and visible strain under pressure. Let uncertainty markers and verbal wobble appear.")
    elif neuroticism <= 0.3:
        directives.append("- Low neuroticism: stay flat, composed, and hard to rattle. Avoid nervous hedging.")

    return directives

def _postprocess_reply_by_personality(
    text: str,
    case_data: Optional[Dict[str, Any]],
    pressure_level: str,
    has_current_contradiction: bool,
) -> str:
    out = trim_to_1_3_sentences(text)
    sentences = _split_short_sentences(out)
    if not sentences:
        return out

    extraversion = _case_personality(case_data)["extraversion"]

    if extraversion <= 0.25 and len(sentences) >= 2:
        return norm(sentences[0])

    if extraversion <= 0.4 and len(sentences) >= 3:
        return norm(" ".join(sentences[:2]))

    if extraversion >= 0.8:
        return out

    if (pressure_level in {"medium", "high"} or has_current_contradiction) and len(sentences) >= 3:
        return norm(" ".join(sentences[:2]))

    return out

def _build_personality_response_breakdown(
    case_data: Optional[Dict[str, Any]],
    player_intent: str,
) -> Dict[str, Any]:
    personality = _case_personality(case_data)
    factors = _calculate_personality_response_factors(case_data, player_intent)
    intent_shift = {
        "Rapport": "agreeableness lowers pressure and supports cooperation",
        "Probe": "neutral factual pressure",
        "Confront": "openness slightly raises response volatility",
        "Intimidate": "neuroticism raises pressure and arousal sensitivity",
        "Neutral": "baseline interpretation",
    }.get(player_intent, "baseline interpretation")
    return {
        "traits": {key: round(value, 3) for key, value in personality.items()},
        "intent": player_intent,
        "intent_effect": intent_shift,
        "computed_factors": {key: round(value, 3) for key, value in factors.items()},
        "readout": {
            "pressure": "higher neuroticism increases felt pressure; higher conscientiousness resists it",
            "cooperation": "agreeableness and extraversion soften cooperation loss",
            "collapse": "conscientiousness delays collapse, neuroticism accelerates wobble",
            "pad": "neuroticism pushes arousal up faster, agreeableness lowers dominance under pressure",
            "speech": "extraversion changes response length, agreeableness changes warmth, openness changes reframing, conscientiousness changes consistency, neuroticism changes shakiness",
            "model": "turn pressure accumulates into cumulative_pressure, then sigmoid converts it to breakdown_probability",
        },
    }

def _apply_personality_scaled_delta(
    prior_value: float,
    updated_value: float,
    multiplier: float,
) -> float:
    delta = updated_value - prior_value
    return clamp01(prior_value + (delta * multiplier))

def _update_pad_state(
    prior_pad_state: Dict[str, float],
    case_data: Optional[Dict[str, Any]],
    player_intent: str,
    pressure_delta: float,
    new_evidence_count: int,
    new_contradiction_count: int,
    soft_dialogue_contradiction: bool,
    repeated_question: bool,
) -> Dict[str, float]:
    personality = _case_personality(case_data)
    factors = _calculate_personality_response_factors(case_data, player_intent)
    neuroticism = personality["neuroticism"]
    conscientiousness = personality["conscientiousness"]
    agreeableness = personality["agreeableness"]

    pleasure = prior_pad_state.get("pleasure", 0.5)
    arousal = prior_pad_state.get("arousal", 0.5)
    dominance = prior_pad_state.get("dominance", 0.5)

    contradiction_impact = 0.05 * float(new_contradiction_count)
    evidence_impact = 0.02 * float(new_evidence_count)
    soft_impact = 0.02 if soft_dialogue_contradiction else 0.0

    pleasure_delta = -pressure_delta * (0.16 + (0.12 * neuroticism))
    pleasure_delta -= contradiction_impact * (0.8 - (0.3 * agreeableness))
    pleasure_delta -= evidence_impact * 0.5
    if player_intent == "Rapport":
        pleasure_delta += 0.025 * factors["rapport_affinity"]
    elif player_intent == "Intimidate":
        pleasure_delta -= 0.02
    if repeated_question:
        pleasure_delta -= 0.015

    arousal_delta = pressure_delta * (0.30 + (0.35 * factors["arousal_sensitivity"]))
    arousal_delta += contradiction_impact + evidence_impact + soft_impact
    if player_intent == "Rapport":
        arousal_delta -= 0.02
    elif player_intent == "Intimidate":
        arousal_delta += 0.015

    dominance_delta = -pressure_delta * (0.22 + (0.18 * agreeableness))
    dominance_delta -= (0.045 * float(new_contradiction_count))
    dominance_delta += 0.02 * (conscientiousness - 0.5)
    if player_intent == "Rapport":
        dominance_delta += 0.01
    if repeated_question:
        dominance_delta += 0.005

    next_pad = {
        "pleasure": clamp01(pleasure + pleasure_delta),
        "arousal": clamp01(arousal + arousal_delta),
        "dominance": clamp01(dominance + (dominance_delta * factors["dominance_resistance"])),
    }
    return next_pad

def _calculate_statement_collapse_stage(
    prior_stage: int,
    cumulative_pressure: float,
    breakdown_probability: float,
    cumulative_contradictions_count: int,
    current_hard_contradictions_count: int,
    soft_dialogue_contradiction: bool,
    pad_state: Dict[str, float],
    case_data: Optional[Dict[str, Any]],
) -> int:
    personality = _case_personality(case_data)
    factors = _calculate_personality_response_factors(case_data, "Neutral")
    collapse_signal = (
        (cumulative_pressure * 4.8)
        + (breakdown_probability * 1.8)
        + (0.45 * float(cumulative_contradictions_count))
        + (0.75 * float(current_hard_contradictions_count))
        + (0.25 if soft_dialogue_contradiction else 0.0)
        + (max(0.0, pad_state.get("arousal", 0.5) - 0.55) * 1.8)
        + (max(0.0, 0.45 - pad_state.get("dominance", 0.5)) * 1.6)
        + (max(0.0, 0.45 - pad_state.get("pleasure", 0.5)) * 1.2)
        + (personality["neuroticism"] * 0.3)
        + (personality["agreeableness"] * 0.1)
        - (0.75 * factors["collapse_resistance"])
    )

    if breakdown_probability >= BREAKDOWN_EXPOSURE_THRESHOLD:
        stage = 5
    elif collapse_signal >= 4.5:
        stage = 4
    elif collapse_signal >= 3.4:
        stage = 3
    elif collapse_signal >= 2.2:
        stage = 2
    elif collapse_signal >= 1.1:
        stage = 1
    else:
        stage = 0

    return max(max(0, min(5, int(prior_stage))), stage)

def _soft_dialogue_stage_floor(
    dialogue_contradiction_signal: Optional[Dict[str, Any]],
    cumulative_pressure: float,
    pad_state: Dict[str, float],
    repeated_question: bool,
) -> int:
    signal = dialogue_contradiction_signal or {}
    severity = str(signal.get("severity", "none")).strip().lower()
    if severity not in {"low", "medium", "high"}:
        return 0

    floor = {
        "low": 1,
        "medium": 2,
        "high": 2,
    }.get(severity, 0)

    if signal.get("suspect_self_contradicted"):
        floor = max(floor, 2 if severity in {"medium", "high"} else 1)
    if signal.get("detective_highlighted") and cumulative_pressure >= 0.14:
        floor = max(floor, 1)

    arousal = clamp01(pad_state.get("arousal", 0.5))
    dominance = clamp01(pad_state.get("dominance", 0.5))

    if cumulative_pressure >= 0.24 or arousal >= 0.66:
        floor = max(floor, 2)
    if (
        severity == "high"
        and cumulative_pressure >= 0.42
        and arousal >= 0.78
        and dominance <= 0.42
        and not repeated_question
    ):
        floor = max(floor, 3)

    if repeated_question:
        floor = min(floor, 2)

    return max(0, min(5, floor))

def _generate_final_psychological_reaction(
    case_data: Optional[Dict[str, Any]],
    statement_collapse_stage: int,
    pad_state: Dict[str, float],
    core_fact_exposed: bool,
) -> str:
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data, dict) else {}
    role = norm(suspect_profile.get("case_role", "용의자")) or "용의자"
    personality = _case_personality(case_data)
    neuroticism = personality["neuroticism"]
    conscientiousness = personality["conscientiousness"]
    agreeableness = personality["agreeableness"]
    arousal = pad_state.get("arousal", 0.5)
    dominance = pad_state.get("dominance", 0.5)
    pleasure = pad_state.get("pleasure", 0.5)

    if core_fact_exposed or statement_collapse_stage >= 5:
        if neuroticism >= 0.65:
            return "용의자는 끝내 시선을 피한 채 호흡이 거칠어지고, 더는 기존 진술을 유지하지 못한 채 사실을 털어놓으려는 상태로 무너져 있다."
        if agreeableness >= 0.6:
            return "용의자는 더 버티기 어렵다는 듯 목소리를 낮추고, 일부가 아니라 핵심 사실까지 인정해야 한다는 표정을 보인다."
        return "용의자는 표정이 굳은 채로도 이미 주도권을 잃었고, 핵심 사실을 숨기기보다 받아들이는 쪽으로 기울어 있다."
    if statement_collapse_stage >= 4:
        if dominance <= 0.35:
            return "용의자는 질문을 정면으로 받지 못하고 답변 사이가 자주 끊기며, 기존 진술 구조가 거의 붕괴된 상태다."
        return "용의자는 기존 진술을 유지하려 하지만 세부 설명을 이어가지 못하고 부분적으로 사실을 수정하려는 흔들림이 크다."
    if statement_collapse_stage >= 3:
        if conscientiousness >= 0.7:
            return "용의자는 여전히 진술 틀을 붙잡고 있지만, 세부를 맞추려 할수록 앞선 설명과 어긋나는 모습이 드러난다."
        return "용의자는 이전보다 말을 고르는 시간이 길어지고, 일부 표현을 정정하며 방어선이 눈에 띄게 얇아진다."
    if statement_collapse_stage >= 2:
        if arousal >= 0.65:
            return "용의자는 표정과 호흡에서 긴장이 크게 올라와 있고, 사소한 질문에도 즉답 대신 망설임이 섞이기 시작했다."
        return "용의자는 아직 부인 기조를 유지하지만, 질문이 깊어질수록 말끝이 흔들리고 설명이 짧아지고 있다."
    if statement_collapse_stage >= 1:
        if pleasure <= 0.35:
            return "용의자는 겉으로는 차분하려 하지만 표정이 굳어 있고, 질문 의도를 경계하며 최소한의 말로 버티고 있다."
        return f"이 {role}은 아직 큰 붕괴는 없지만, 방어적으로 진술을 반복하며 심문 흐름을 조심스럽게 살피고 있다."
    return f"이 {role}은 현재까지는 진술 구조를 비교적 안정적으로 유지하고 있으며, 질문 주도권을 쉽게 넘기지 않으려 한다."

def build_final_reaction_speech(
    case_data: Optional[Dict[str, Any]],
    statement_collapse_stage: int,
    pad_state: Dict[str, float],
    core_fact_exposed: bool,
) -> str:
    personality = _case_personality(case_data)
    arousal = clamp01(pad_state.get("arousal", 0.5))
    dominance = clamp01(pad_state.get("dominance", 0.5))
    agreeableness = personality["agreeableness"]
    conscientiousness = personality["conscientiousness"]

    if core_fact_exposed or statement_collapse_stage >= 5:
        if arousal >= 0.9:
            return "더는 같은 말을 계속 반복하기 어렵습니다. 지금은 조금만 정리할 시간을 주십시오."
        if dominance <= 0.32:
            return "계속 그렇게 몰아붙이시면 제가 정리가 안 됩니다. 지금은 더 말을 보태기 어렵습니다."
        if agreeableness >= 0.6:
            return "지금은 차분히 정리해서 다시 말씀드리는 게 맞겠습니다. 섭불리 단정해서 말씀드리기 어렵습니다."
        return "지금은 더 버티기 어렵습니다. 다만 여기서 제가 다 말씀드릴 수는 없습니다."

    if statement_collapse_stage >= 4:
        if conscientiousness >= 0.7:
            return "제 말이 조금 흔들릴 수는 있어도, 지금 바로 단정해서 말씀드리긴 어렵습니다."
        return "지금은 제가 말씀을 정리해야 할 것 같습니다. 같은 말만 반복하기도 어렵습니다."

    if statement_collapse_stage >= 2:
        return "제가 바로 답을 못 드리는 건 맞지만, 지금 드린 말 이상으로 보태기는 어렵습니다."

    return "제가 드린 말씀에서 크게 달라진 건 없습니다. 지금은 그 정도로만 말씀드리겠습니다."

def _normalize_documents(
    raw_documents: Any,
    overview: Dict[str, Any],
    suspect: Dict[str, Any],
) -> Dict[str, Any]:
    documents = raw_documents if isinstance(raw_documents, dict) else {}
    case_overview = documents.get("case_overview", {}) if isinstance(documents.get("case_overview"), dict) else {}
    scene_report = documents.get("scene_report", {}) if isinstance(documents.get("scene_report"), dict) else {}
    character_info = documents.get("character_info", {}) if isinstance(documents.get("character_info"), dict) else {}

    return {
        "case_overview": {
            "incident_time": norm(case_overview.get("incident_time", "")) or norm(overview.get("time", "")),
            "incident_place": norm(case_overview.get("incident_place", "")) or norm(overview.get("place", "")),
            "incident_type": norm(case_overview.get("incident_type", "")) or norm(overview.get("type", "")),
            "victim_status": norm(case_overview.get("victim_status", "")),
            "summary": norm(case_overview.get("summary", "")),
        },
        "scene_report": {
            "summary": norm(scene_report.get("summary", "")),
            "bullet_points": _normalize_string_list(scene_report.get("bullet_points", [])),
        },
        "character_info": {
            "suspect_summary": norm(character_info.get("suspect_summary", "")),
            "victim_relation": norm(character_info.get("victim_relation", "")) or norm(suspect.get("relation", "")),
            "conflict_points": _normalize_string_list(character_info.get("conflict_points", [])),
        },
        "reference_statements": _normalize_string_list(documents.get("reference_statements", [])),
        "timeline": _normalize_string_list(documents.get("timeline", [])),
        "detective_memo": _normalize_string_list(documents.get("detective_memo", [])),
    }

def coerce_case_payload(case_blob: Any, fallback_case_id: str = "") -> Optional[Dict[str, Any]]:
    """
    Normalize one case file into the server's runtime structure.

    Important:
    - documents / suspect_profile are retained for briefing, UI, and later extensions
    - the existing interrogation engine still runs on truth_slots / evidences / contradictions
    """
    if not isinstance(case_blob, dict):
        return None

    overview = case_blob.get("overview", {}) if isinstance(case_blob.get("overview"), dict) else {}
    suspect = case_blob.get("suspect", {}) if isinstance(case_blob.get("suspect"), dict) else {}
    selection_card = case_blob.get("selection_card", {}) if isinstance(case_blob.get("selection_card"), dict) else {}
    suspect_profile = case_blob.get("suspect_profile", {}) if isinstance(case_blob.get("suspect_profile"), dict) else {}
    default_personality = suspect_profile.get("default_personality", {}) if isinstance(suspect_profile.get("default_personality"), dict) else {}
    legacy_personality = suspect_profile.get("personality", {}) if isinstance(suspect_profile.get("personality"), dict) else {}
    mental_state = suspect_profile.get("mental_state", {}) if isinstance(suspect_profile.get("mental_state"), dict) else {}
    truth_slots = case_blob.get("truth_slots", {}) if isinstance(case_blob.get("truth_slots"), dict) else {}
    evidences = case_blob.get("evidences", []) if isinstance(case_blob.get("evidences"), list) else []
    contradictions = case_blob.get("contradictions", []) if isinstance(case_blob.get("contradictions"), list) else []

    def to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    normalized_evidences: List[Dict[str, Any]] = []
    for evidence in evidences:
        if not isinstance(evidence, dict):
            continue
        aliases = evidence.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        normalized_evidences.append(
            {
                "id": norm(evidence.get("id", "")),
                "name": norm(evidence.get("name", "")),
                "description": norm(evidence.get("description", "")),
                "aliases": uniq_strings([norm(alias) for alias in aliases if norm(alias)]),
            }
        )

    normalized_contradictions: List[Dict[str, Any]] = []
    for contradiction in contradictions:
        if not isinstance(contradiction, dict):
            continue
        related_evidence = contradiction.get("related_evidence", [])
        if not isinstance(related_evidence, list):
            related_evidence = []
        slot_name = norm(contradiction.get("slot", ""))
        if slot_name not in TRUTH_SLOT_NAMES:
            slot_name = ""
        truth_value = norm(contradiction.get("truth_value", ""))
        if slot_name == "met_victim_that_day":
            truth_value = _normalize_slot_value_for_slot(truth_value, slot_name)
        contradiction_type = norm(contradiction.get("contradiction_type", ""))
        if contradiction_type not in VALID_CONTRADICTION_TYPES:
            contradiction_type = "claim_vs_evidence"
        normalized_contradictions.append(
            {
                "id": norm(contradiction.get("id", "")),
                "description": norm(contradiction.get("description", "")),
                "related_evidence": uniq_strings([str(item).strip() for item in related_evidence]),
                "slot": slot_name,
                "truth_value": truth_value,
                "contradiction_type": contradiction_type,
            }
        )

    normalized_truth_slots = _default_truth_slots()
    for slot_name in TRUTH_SLOT_NAMES:
        raw_slot_value = norm(truth_slots.get(slot_name, ""))
        if slot_name == "met_victim_that_day":
            normalized_truth_slots[slot_name] = _normalize_slot_value_for_slot(raw_slot_value, slot_name)
        else:
            normalized_truth_slots[slot_name] = raw_slot_value

    case_id = norm(case_blob.get("case_id", "")) or norm(fallback_case_id)
    if not case_id:
        return None

    normalized_personality = _default_personality()
    raw_default_personality = default_personality if default_personality else legacy_personality
    for trait in BIG_FIVE_TRAITS:
        normalized_personality[trait] = _to_clamped_float(raw_default_personality.get(trait, normalized_personality[trait]))

    normalized_mental_state = _default_mental_state()
    for field in PAD_STATE_FIELDS:
        normalized_mental_state[field] = _to_clamped_float(mental_state.get(field, normalized_mental_state[field]))

    return {
        "case_id": case_id,
        "selection_card": {
            "title": norm(selection_card.get("title", "")),
            "subtitle": norm(selection_card.get("subtitle", "")),
            "brief": norm(selection_card.get("brief", "")),
        },
        "overview": {
            "time": norm(overview.get("time", "")),
            "place": norm(overview.get("place", "")),
            "type": norm(overview.get("type", "")),
        },
        "documents": _normalize_documents(case_blob.get("documents", {}), overview, suspect),
        "motive": norm(case_blob.get("motive", "")),
        "crime_flow": norm(case_blob.get("crime_flow", "")),
        "suspect": {
            "name": norm(suspect.get("name", "")),
            "age": to_int(suspect.get("age", 0)),
            "job": norm(suspect.get("job", "")),
            "relation": norm(suspect.get("relation", "")),
        },
        "suspect_profile": {
            "case_role": norm(suspect_profile.get("case_role", "")),
            "default_personality": normalized_personality,
            "selected_personality": {},
            "mental_state": normalized_mental_state,
        },
        "false_statement": norm(case_blob.get("false_statement", "")),
        "truth_slots": normalized_truth_slots,
        "evidences": normalized_evidences,
        "contradictions": normalized_contradictions,
    }

def load_prebuilt_case_library() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for path in sorted(PREBUILT_CASE_DIR.glob("*.json")):
        case_blob = read_json_file(path, None)
        case_data = coerce_case_payload(case_blob)
        if not case_data:
            continue
        cases.append(case_data)
    return cases

def pick_prebuilt_case_choices(num_choices: int = 3) -> List[Dict[str, Any]]:
    library = load_prebuilt_case_library()
    if not library:
        return []
    if len(library) <= num_choices:
        shuffled = list(library)
        random.shuffle(shuffled)
        return shuffled
    return random.sample(library, num_choices)

def _public_case_briefing(case_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(case_data, dict):
        return {}
    suspect = case_data.get("suspect", {}) if isinstance(case_data.get("suspect"), dict) else {}
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data.get("suspect_profile"), dict) else {}
    return {
        "case_id": norm(case_data.get("case_id", "")),
        "selection_card": case_data.get("selection_card", {}),
        "overview": case_data.get("overview", {}),
        "documents": case_data.get("documents", {}),
        "suspect": {
            "name": norm(suspect.get("name", "")),
            "age": int(safe_float(suspect.get("age", 0), 0.0)),
            "job": norm(suspect.get("job", "")),
            "relation": norm(suspect.get("relation", "")),
        },
        "suspect_profile": {
            "default_personality": _case_default_personality(case_data),
            "selected_personality": _normalize_selected_personality_blob(
                suspect_profile.get("selected_personality", {})
            ),
            "mental_state": _normalize_pad_state_blob(suspect_profile.get("mental_state", {})),
        },
    }


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
            aliases = e.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    if alias:
                        words.append(str(alias))
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

    pressure_level = str(interrogation_signal.get("pressure_level", "none")).strip().lower()
    current_evidence_ids = interrogation_signal.get("mentioned_evidence_ids", []) or []
    intent = str(interrogation_signal.get("intent", "irrelevant")).strip().lower()
    target_slot = norm(interrogation_signal.get("target_slot", ""))

    evidence_map = {
        str(e.get("id", "")).strip(): e
        for e in _case_evidences(case_data)
        if e.get("id")
    }

    evidence_lines = []
    for evidence_id in current_evidence_ids:
        evidence = evidence_map.get(str(evidence_id).strip())
        if evidence:
            evidence_lines.append(
                f"- {evidence.get('name', evidence_id)}: {evidence.get('description', '')}".strip()
            )

    directives = []
    if intent == "point_contradiction":
        directives.append(
            "- The detective is explicitly pointing out an inconsistency. Answer that inconsistency directly first in polite Korean."
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
            "- Let the story wobble a little. You may partially correct yourself, but do not reveal every core fact at once."
        )
    if target_slot:
        directives.append(
            f"- Focus on the slot under pressure: {SLOT_LABELS.get(target_slot, target_slot)}."
        )

    if not evidence_lines and not directives:
        return ""

    return (
        f"\n[Pressure level this turn] {pressure_level}\n"
        "[Evidence raised this turn]\n"
        + ("\n".join(evidence_lines) if evidence_lines else "(none)")
        + "\n[Response guidance]\n"
        + ("\n".join(directives) if directives else "(none)")
    )

def _build_suspect_answer_system_prompt() -> str:
    return (
        "You are a suspect being interrogated by a detective.\n"
        "Reply in natural spoken Korean using polite speech (존댓말) every turn.\n"
        "Use 1 to 3 spoken sentences.\n"
        "Let the selected personality visibly change brevity, warmth, hesitation, rigidity, and willingness to elaborate.\n"
        "Do not flatten all suspects into the same generic voice.\n"
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
        "Do not volunteer hidden truth-slot information unless the detective directly asks about that specific point.\n"
        "Do not jump straight to a full confession. If pressure peaks, reveal only the minimum core fact that leaks through.\n"
        "If this is the first turn, do not imply that you already explained it before.\n"
        "Do not say phrases like '말씀드렸습니다', '아까 말했다', '전에 말했다', or '이미 말했듯이' unless such dialogue actually exists in history.\n"
        "When history is empty, answer as if this is the first time you are responding.\n"
    )

def llm_suspect_answer(
    case_context: str,
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    breakdown_probability: float,
    interrogation_signal: Optional[Dict[str, Any]] = None,
    behavior_state: str = DEFAULT_FSM_STATE,
    statement_collapse_stage: int = 0,
    pad_state: Optional[Dict[str, float]] = None,
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
    pressure_level = str((interrogation_signal or {}).get("pressure_level", "none")).strip().lower()
    has_current_contradiction = str((interrogation_signal or {}).get("intent", "")).strip().lower() == "point_contradiction"
    current_pad_state = _normalize_pad_state_blob(pad_state or {})
    collapse_label = _statement_collapse_label(statement_collapse_stage)
    personality = _case_personality(case_data)
    reply_personality_factors = _calculate_personality_response_factors(
        case_data,
        _infer_player_intent(interrogation_signal),
    )
    reply_length_bias = reply_personality_factors.get("reply_length_bias", 1.0)
    personality_speaking_directives = _build_personality_speaking_directives(case_data)

    extra_guard = ""
    if turn_pressure_context:
        extra_guard += turn_pressure_context
    if has_current_contradiction:
        extra_guard += "\n- Answer the contradiction directly and first."
        extra_guard += "\n- Keep the reply polite, short, and focused on that inconsistency."
    elif pressure_level in {"medium", "high"}:
        extra_guard += "\n- Respond to the pressure point directly instead of broadening the story."
        extra_guard += "\n- Stay polite and give only one concrete detail, but let the personality still shape how terse or talkative the surrounding wording feels."
    if behavior_state == "Angry / Uncooperative":
        extra_guard += "\n- Sound colder and more resistant. You may push back and cooperate less."
    elif behavior_state == "Pressured / Shaken":
        extra_guard += "\n- Let slight hesitation or a small wobble show in the wording."
    if reply_length_bias <= 0.82:
        extra_guard += "\n- Keep the reply very short: usually exactly 1 compact sentence, or 2 only if absolutely unavoidable."
    elif reply_length_bias <= 0.95:
        extra_guard += "\n- Keep the reply especially short and compact. Do not volunteer a second sentence unless needed."
    elif reply_length_bias >= 1.35:
        extra_guard += "\n- Use fuller spoken wording and prefer 2 or 3 short sentences with a visible follow-up explanation or reaction."
    elif reply_length_bias >= 1.10:
        extra_guard += "\n- You may use fuller wording and prefer 2 short sentences rather than 1."
    if statement_collapse_stage >= 4:
        extra_guard += "\n- The statement structure is close to collapsing. Let short corrections and unstable wording appear naturally."
    elif statement_collapse_stage >= 3:
        extra_guard += "\n- You are partly revising your story. Small corrections are allowed, but avoid revealing everything at once."
    elif statement_collapse_stage >= 2:
        extra_guard += "\n- Keep the denial, but add visible hesitation or caution."
    elif statement_collapse_stage <= 1:
        extra_guard += "\n- Maintain a relatively stable denial structure."
    extra_guard += "\n- Answer only the detective's latest question or accusation."
    extra_guard += "\n- Put the direct answer in the first sentence."
    extra_guard += "\n- If the detective quotes a line or message, explain that exact line first."
    extra_guard += "\n- Do not widen into unrelated scenes, meetings, background stories, or prior events unless they are strictly needed to answer."
    if is_too_ambiguous(user_text):
        extra_guard += "\n- If the question is vague, ask what point the detective wants clarified."
    if detect_repeat(history, user_text):
        extra_guard += "\n- If the same question is repeated, answer that same point plainly."
    if not hist_lines:
        extra_guard += "\n- This is the first reply in the interrogation."
        extra_guard += "\n- Do not imply any prior explanation or prior statement."
        extra_guard += "\n- Do not use phrases like '말씀드렸습니다', '아까 말했다', '전에 말했다', or '이미 말했듯이'."

    if personality_speaking_directives:
        extra_guard += "\n[Personality speaking style]\n" + "\n".join(personality_speaking_directives)

    user = (
        case_context
        + f"\n[Current behavioral state] {behavior_state}\n"
        + f"[Current statement collapse stage] {statement_collapse_stage} ({collapse_label})\n"
        + (
            f"[Selected Big Five] openness:{personality['openness']:.2f} "
            f"conscientiousness:{personality['conscientiousness']:.2f} "
            f"extraversion:{personality['extraversion']:.2f} "
            f"agreeableness:{personality['agreeableness']:.2f} "
            f"neuroticism:{personality['neuroticism']:.2f}\n"
        )
        + (
            f"[Current PAD state] pleasure:{current_pad_state['pleasure']:.2f} "
            f"arousal:{current_pad_state['arousal']:.2f} "
            f"dominance:{current_pad_state['dominance']:.2f}\n"
        )
        + f"\n[Current breakdown probability reference] {breakdown_probability:.2f}\n"
        + "[Recent dialogue]\n"
        + ("\n".join(hist_lines) if hist_lines else "(none)")
        + f"\n[Latest detective question]\n{user_text}\n"
        + f"\n[Evidence words already raised by the detective; do not go beyond these unless strictly needed] {allowed_evidence_words}\n"
        + f"\n[Response guardrails]\n{extra_guard}\n"
        + "Output only the suspect's reply."
    )

    def _generate_suspect_reply(prompt_text: str, max_output_tokens: int = 220) -> str:
        adjusted_tokens = max(
            90,
            min(380, int(round(max_output_tokens * max(0.55, min(1.6, reply_length_bias))))),
        )
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_text},
            ],
            max_output_tokens=adjusted_tokens,
        )
        return trim_to_1_3_sentences((resp.output_text or "").strip())

    out = _generate_suspect_reply(user, 220)

    if _contains_banned_evidence(out, all_evidence_words, allowed_evidence_words):
        retry_user = (
            user
            + "\n\n[Correction] The previous reply mentioned evidence or facts the detective did not bring up. Remove that and answer with only a general denial or limited explanation."
        )
        out = _generate_suspect_reply(retry_user, 200)

    out = _postprocess_reply_by_personality(
        out,
        case_data,
        pressure_level,
        has_current_contradiction,
    )

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

    if _contains_banned_evidence(out, all_evidence_words, allowed_evidence_words):
        out = "그 부분은 제가 지금 드릴 말씀이 없습니다."

    if not hist_lines:
        replacements = {
            "아까 말씀드렸듯이": "",
            "전에 말씀드렸듯이": "",
            "이미 말씀드렸지만": "",
            "계속 말씀드리지만": "",
            "말씀드렸습니다": "말씀드립니다",
            "말씀드린 대로": "제 말씀은",
        }
        for old, new in replacements.items():
            out = out.replace(old, new)
        out = norm(out)

    return out or "죄송하지만 그 질문에는 바로 답드리기 어렵습니다."

_CONTRADICTION_CUE_PATTERNS = (
    "아까", "방금", "전에", "처음엔", "그런데", "근데", "모순", "말이다르",
    "말이다르잖", "했잖", "했다며", "했는데", "찍혔던데", "라면서", "라고했", "아니라며", "왜이제", "왜지금",
)
_HOME_REST_PATTERNS = ("자고있", "잤", "집에있", "집에만있", "안나갔", "밖에안나갔")
_PRESENCE_DENY_PATTERNS = ("안갔", "간적없", "간일없", "안왔", "안들렀", "방문안")
_PRESENCE_ADMIT_PATTERNS = ("갔", "가서", "왔", "들렀", "방문", "찾아갔", "확인하러", "나갔", "올라갔", "올라간", "올라가")
_MEET_DENY_PATTERNS = ("안만났", "만난적없", "본적없", "마주친적없", "연락안했")
_MEET_ADMIT_PATTERNS = ("만났", "봤", "마주쳤", "연락했", "통화했")
_ALONE_PATTERNS = ("혼자있", "혼자였", "저혼자")
_WITH_OTHER_PATTERNS = ("같이있", "함께있", "둘이있", "누구랑있", "누구와있")
_EXCLUSIVE_LOCATION_PATTERNS = ("에만 있었", "거기에만 있었", "밖에 없었", "줄곧 있었")
_CORRECTION_PATTERNS = ("정확하지 않았", "잘못 말했", "아까 말은", "정정하겠습니다", "맞지만")
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

_DIALOGUE_PLACE_KEYWORDS = (
    "공장 복도",
    "병원 쪽",
    "창고 근처",
    "편의점",
    "공장",
    "복도",
    "병원",
    "집",
    "회사",
    "사무실",
    "창고",
    "주차장",
    "골목",
    "항만",
    "공원",
)

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
    normalized = norm(text)
    target = _squash_korean_text(text)
    presence_denied = any(pattern in target for pattern in _PRESENCE_DENY_PATTERNS)
    meet_denied = any(pattern in target for pattern in _MEET_DENY_PATTERNS)
    exclusive_location = any(pattern in normalized for pattern in _EXCLUSIVE_LOCATION_PATTERNS)
    return {
        "home_rest": any(pattern in target for pattern in _HOME_REST_PATTERNS),
        "presence_denied": presence_denied,
        "presence_admitted": (not presence_denied) and (
            any(pattern in target for pattern in _PRESENCE_ADMIT_PATTERNS) or exclusive_location
        ),
        "meet_denied": meet_denied,
        "meet_admitted": (not meet_denied) and any(
            pattern in target for pattern in _MEET_ADMIT_PATTERNS
        ),
        "alone": any(pattern in target for pattern in _ALONE_PATTERNS),
        "with_other": any(pattern in target for pattern in _WITH_OTHER_PATTERNS),
        "exclusive_location": exclusive_location,
        "correction": any(pattern in normalized for pattern in _CORRECTION_PATTERNS),
    }

def _extract_dialogue_place(text: str) -> str:
    normalized = norm(text)
    if not normalized:
        return ""

    for place in _DIALOGUE_PLACE_KEYWORDS:
        if place in normalized:
            return place

    squashed = _normalize_place_text(normalized)
    for place in _DIALOGUE_PLACE_KEYWORDS:
        place_squashed = _normalize_place_text(place)
        if place_squashed and place_squashed in squashed:
            return place
    return ""

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

def detect_dialogue_contradiction_local(
    history: List[Dict[str, Any]],
    user_text: str,
    suspect_text: str,
) -> Dict[str, Any]:
    current_suspect_text = norm(suspect_text)
    if not current_suspect_text:
        return _empty_dialogue_contradiction_signal("empty suspect reply")

    current_flags = _extract_dialogue_flags(current_suspect_text)
    current_place = _extract_dialogue_place(current_suspect_text)
    if not any(current_flags.values()) and not current_place:
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
        prior_place = _extract_dialogue_place(prior_text)
        severity = "none"

        if (
            current_place
            and prior_place
            and not _slot_values_match(current_place, prior_place, "crime_place")
        ):
            severity = "high" if (
                current_flags.get("correction")
                or current_flags.get("exclusive_location")
                or prior_flags.get("exclusive_location")
            ) else "medium"
            return {
                "detective_highlighted": detective_highlighted,
                "suspect_self_contradicted": True,
                "severity": severity,
                "prior_claim": prior_text,
                "current_claim": current_suspect_text,
                "reason": f"place changed from {prior_place} to {current_place}",
            }

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

def _resolve_case_from_request(
    case_id: str,
    case_json: str = "",
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    resolved_case_id = norm(case_id)
    raw_case_json = case_json or ""
    if not resolved_case_id and not raw_case_json.strip():
        return "", None, {"error": "missing_case_context"}, 400

    case_data = None
    client_case_payload = safe_json_loads(raw_case_json, None) if raw_case_json.strip() else None
    client_case_data = coerce_case_payload(client_case_payload, resolved_case_id)
    if client_case_data:
        payload_case_id = norm(client_case_data.get("case_id", ""))
        if resolved_case_id and payload_case_id != resolved_case_id:
            return resolved_case_id, None, {
                "error": "case_id_mismatch",
                "message": "case_id and case_json.case_id do not match.",
                "case_id": resolved_case_id,
                "case_json_case_id": payload_case_id,
            }, 400
        resolved_case_id = payload_case_id
        case_data = client_case_data
        CASE_CACHE[resolved_case_id] = case_data
        persist_case(case_data)
    elif resolved_case_id:
        case_data = load_case(resolved_case_id)

    if resolved_case_id and not case_data:
        return resolved_case_id, None, {
            "error": "case_not_found",
            "message": "Unknown case_id.",
            "case_id": resolved_case_id,
        }, 404
    if not case_data:
        return resolved_case_id, None, {
            "error": "invalid_case_context",
            "message": "Provide a valid case_id or case_json for interrogation.",
        }, 400
    return resolved_case_id, case_data, None, 200

# =========================================================
# ENDPOINTS
# =========================================================
@app.post("/case/generate")
async def case_generate():
    """
    Prebuilt case selection endpoint.

    This no longer generates a new case with the LLM.
    It returns 3 prebuilt case files and keeps "case" alongside "cases"
    for backwards compatibility with older clients.

    The interrogation engine used by /interrogation/qna is unchanged.
    """
    try:
        case_choices = pick_prebuilt_case_choices(3)
        if not case_choices:
            raise RuntimeError("No prebuilt cases found in cases/prebuilt")

        for case_data in case_choices:
            cid = norm(case_data.get("case_id", ""))
            if not cid:
                continue
            CASE_CACHE[cid] = case_data
            persist_case(case_data)

        primary_case = case_choices[0]
        public_case_choices = [_public_case_briefing(case_data) for case_data in case_choices]
        return JSONResponse(
            status_code=200,
            content={
                "source": "prebuilt",
                "cases": case_choices,
                "case": primary_case,
                "briefing_cases": public_case_choices,
                "briefing_case": public_case_choices[0] if public_case_choices else {},
            },
        )

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb[-2000:]})

@app.post("/interrogation/setup")
async def interrogation_setup(
    case_id: str = Form(""),
    case_json: str = Form(""),
    personality_json: str = Form(""),
    reset_progress: str = Form(""),
):
    try:
        resolved_case_id, case_data, error_content, status_code = _resolve_case_from_request(case_id, case_json)
        if error_content:
            return JSONResponse(status_code=status_code, content=error_content)

        if resolved_case_id and is_truthy_string(reset_progress):
            INTERROGATION_PROGRESS_CACHE.pop(resolved_case_id, None)

        progress_state = _get_progress_state(resolved_case_id)
        if not personality_json.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing_selected_personality",
                    "message": "Select NPC personality before starting interrogation.",
                },
            )
        personality_payload = safe_json_loads(personality_json, None)
        if not isinstance(personality_payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_selected_personality",
                    "message": "personality_json must be a JSON object with all five Big Five values.",
                },
            )
        selected_personality, missing_traits = _validate_selected_personality_payload(
            personality_payload
        )
        if missing_traits:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_selected_personality",
                    "message": "personality_json must include all five Big Five values.",
                    "missing_traits": missing_traits,
                },
            )
        effective_case = _effective_case_from_progress(
            case_data,
            progress_state,
            selected_personality,
        )
        progress_snapshot = dict(progress_state)
        progress_snapshot["selected_personality"] = selected_personality
        progress_snapshot["final_psychological_report"] = _build_final_psychological_report(
            effective_case,
            progress_snapshot,
        )
        stored_progress = _persist_progress_snapshot(resolved_case_id, progress_snapshot)
        return JSONResponse(
            status_code=200,
            content={
                "case_id": resolved_case_id,
                "default_personality": _case_default_personality(case_data),
                "selected_personality": selected_personality,
                "baseline_pad_state": _case_baseline_pad_state(effective_case),
                "statement_collapse_stage": int(stored_progress.get("statement_collapse_stage", 0)),
                "turn_count": int(stored_progress.get("turn_count", 0)),
                "final_psychological_report": stored_progress.get("final_psychological_report", {}),
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb[-2000:]},
        )

@app.post("/interrogation/judgment/submit")
async def interrogation_judgment_submit(
    case_id: str = Form(""),
    case_json: str = Form(""),
    judgment: str = Form(""),
    notes: str = Form(""),
):
    try:
        resolved_case_id, case_data, error_content, status_code = _resolve_case_from_request(case_id, case_json)
        if error_content:
            return JSONResponse(status_code=status_code, content=error_content)

        choice_key = _normalize_judgment_choice(judgment)
        if not choice_key:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_judgment",
                    "allowed": JUDGMENT_CHOICE_LABELS,
                },
            )

        progress_state = _get_progress_state(resolved_case_id)
        effective_case = _effective_case_from_progress(case_data, progress_state)
        submitted_judgment = _normalize_submitted_judgment(
            {
                "choice_key": choice_key,
                "notes": notes,
                "turn_count": progress_state.get("turn_count", 0),
            }
        )
        progress_snapshot = dict(progress_state)
        progress_snapshot["submitted_judgment"] = submitted_judgment
        progress_snapshot["final_psychological_report"] = _build_final_psychological_report(
            effective_case,
            progress_snapshot,
        )
        stored_progress = _persist_progress_snapshot(resolved_case_id, progress_snapshot)
        return JSONResponse(
            status_code=200,
            content={
                "case_id": resolved_case_id,
                "submitted_judgment": stored_progress.get("submitted_judgment", {}),
                "result_available": True,
                "final_psychological_report": stored_progress.get("final_psychological_report", {}),
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb[-2000:]},
        )

@app.post("/interrogation/report")
async def interrogation_report(
    case_id: str = Form(""),
    case_json: str = Form(""),
):
    try:
        resolved_case_id, case_data, error_content, status_code = _resolve_case_from_request(case_id, case_json)
        if error_content:
            return JSONResponse(status_code=status_code, content=error_content)

        progress_state = _get_progress_state(resolved_case_id)
        effective_case = _effective_case_from_progress(case_data, progress_state)
        final_report = _build_final_psychological_report(effective_case, progress_state)
        progress_snapshot = dict(progress_state)
        progress_snapshot["final_psychological_report"] = final_report
        stored_progress = _persist_progress_snapshot(resolved_case_id, progress_snapshot)
        return JSONResponse(
            status_code=200,
            content={
                "case_id": resolved_case_id,
                "final_psychological_report": final_report,
                "statement_records": stored_progress.get("statement_records", []),
                "submitted_judgment": stored_progress.get("submitted_judgment", {}),
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb[-2000:]},
        )

@app.post("/interrogation/result/reveal")
async def interrogation_result_reveal(
    case_id: str = Form(""),
    case_json: str = Form(""),
):
    try:
        resolved_case_id, case_data, error_content, status_code = _resolve_case_from_request(case_id, case_json)
        if error_content:
            return JSONResponse(status_code=status_code, content=error_content)

        progress_state = _get_progress_state(resolved_case_id)
        effective_case = _effective_case_from_progress(case_data, progress_state)
        final_report = _build_final_psychological_report(effective_case, progress_state)
        actual_choice_key = _case_result_choice_key(effective_case)
        submitted_judgment = _normalize_submitted_judgment(progress_state.get("submitted_judgment", {}))
        judgment_matches = bool(
            submitted_judgment.get("choice_key")
            and submitted_judgment.get("choice_key") == actual_choice_key
        )
        progress_snapshot = dict(progress_state)
        progress_snapshot["final_psychological_report"] = final_report
        stored_progress = _persist_progress_snapshot(resolved_case_id, progress_snapshot)
        return JSONResponse(
            status_code=200,
            content={
                "case_id": resolved_case_id,
                "actual_result": {
                    "choice_key": actual_choice_key,
                    "choice_label": JUDGMENT_CHOICE_LABELS.get(actual_choice_key, ""),
                    "case_role": norm(((effective_case or {}).get("suspect_profile", {}) or {}).get("case_role", "")),
                    "crime_flow_summary": norm((effective_case or {}).get("crime_flow", "")),
                },
                "submitted_judgment": submitted_judgment,
                "judgment_matches": judgment_matches,
                "final_psychological_report": final_report,
                "statement_records": stored_progress.get("statement_records", []),
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb[-2000:]},
        )

@app.post("/interrogation/debug_confess")
async def interrogation_debug_confess(
    case_id: str = Form(""),
    case_json: str = Form(""),
    history_json: str = Form("[]"),
    user_text: str = Form(""),
):
    try:
        history = safe_json_loads(history_json, [])
        if not isinstance(history, list):
            history = []
        history = history[-20:]

        case_id = norm(case_id)
        if not case_id and not case_json.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "missing_case_context"},
            )
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
        elif case_id:
            case_data = load_case(case_id)
        if case_id and not case_data:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "case_not_found",
                    "message": "Unknown case_id.",
                    "case_id": case_id,
                },
            )
        if not case_data:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_case_context",
                    "message": "Provide a valid case_id or case_json for interrogation.",
                },
            )

        final_user_text = norm(user_text)
        if not final_user_text:
            for entry in reversed(history):
                if not isinstance(entry, dict):
                    continue
                candidate = norm(entry.get("user_text", ""))
                if candidate:
                    final_user_text = candidate
                    break
        if not final_user_text:
            final_user_text = "더는 숨길 수 없으니 사실대로 전부 인정하세요."

        prior_progress = _get_progress_state(case_id)
        effective_case_data = _effective_case_from_progress(case_data, prior_progress)
        debug_pad_state = {
            "pleasure": 0.08,
            "arousal": 0.94,
            "dominance": 0.12,
        }
        suspect_text = trim_to_1_3_sentences(
            build_final_reaction_speech(effective_case_data, 5, debug_pad_state, True)
        )
        wav_b64 = await tts_to_b64(suspect_text)
        final_psychological_reaction = _generate_final_psychological_reaction(
            effective_case_data,
            5,
            debug_pad_state,
            True,
        )
        statement_records = _normalize_statement_records(prior_progress.get("statement_records", []))
        statement_records.append(
            _build_statement_record(
                MAX_GAME_TURNS,
                final_user_text,
                {"intent": "point_contradiction", "target_slot": "", "pressure_level": "high"},
                suspect_text,
                {
                    "claimed_value": "",
                    "hard_contradiction_ids": [],
                    "soft_dialogue_contradiction": True,
                },
                {
                    "statement_collapse_stage": 5,
                    "pad_state": debug_pad_state,
                    "fsm_state": EXPOSURE_FSM_STATE,
                    "breakdown_probability": 1.0,
                    "core_fact_exposed": True,
                },
            )
        )
        debug_progress_snapshot = {
            "breakdown_probability": 1.0,
            "referenced_evidence_ids": prior_progress.get("referenced_evidence_ids", []),
            "established_contradiction_ids": prior_progress.get("established_contradiction_ids", []),
            "stress_score": 1.0,
            "cooperation_score": 0.0,
            "cumulative_pressure": 1.0,
            "fsm_state": EXPOSURE_FSM_STATE,
            "last_raw_odds": 8.0,
            "last_sue_impact": 3.0,
            "statement_collapse_stage": 5,
            "core_fact_exposed": True,
            "pad_state": debug_pad_state,
            "final_psychological_reaction": final_psychological_reaction,
            "selected_personality": prior_progress.get("selected_personality", {}),
            "statement_records": statement_records[-MAX_STATEMENT_RECORDS:],
            "submitted_judgment": prior_progress.get("submitted_judgment", {}),
            "turn_count": MAX_GAME_TURNS,
        }
        final_psychological_report = _build_final_psychological_report(
            effective_case_data,
            debug_progress_snapshot,
        )
        if case_id:
            _store_progress_state(
                case_id,
                1.0,
                prior_progress.get("referenced_evidence_ids", []),
                prior_progress.get("established_contradiction_ids", []),
                1.0,
                0.0,
                1.0,
                EXPOSURE_FSM_STATE,
                8.0,
                3.0,
                MAX_GAME_TURNS,
                5,
                debug_pad_state,
                final_psychological_reaction,
                prior_progress.get("selected_personality", {}),
                statement_records[-MAX_STATEMENT_RECORDS:],
                prior_progress.get("submitted_judgment", {}),
                final_psychological_report,
            )
        return JSONResponse(
            status_code=200,
            content={
                "user_text": final_user_text,
                "suspect_text": suspect_text,
                "pressure_delta": 0.0,
                "breakdown_probability": 1.0,
                "core_fact_exposed": True,
                "fsm_state": EXPOSURE_FSM_STATE,
                "stress_score": 1.0,
                "cumulative_pressure": 1.0,
                "raw_odds": 8.0,
                "latest_sue_impact": 3.0,
                "statement_collapse_stage": 5,
                "statement_collapse_label": _statement_collapse_label(5),
                "pad_state": debug_pad_state,
                "final_psychological_reaction": final_psychological_reaction,
                "final_psychological_report": final_psychological_report,
                "judgment_ready": True,
                "turn_count": MAX_GAME_TURNS,
                "debug_force_breakdown_state": True,
                "audio_wav_b64": wav_b64,
            },
        )
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb[-2000:]},
        )

@app.post("/interrogation/qna")
async def interrogation_qna(
    file: Optional[UploadFile] = File(None),
    user_text: str = Form(""),
    case_id: str = Form(""),
    case_json: str = Form(""),
    personality_json: str = Form(""),
    history_json: str = Form("[]"),
    debug: str = Form(""),
    reset_progress: str = Form(""),
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
      - breakdown_probability
      - core_fact_exposed
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
                try:
                    final_user_text = stt_transcribe(audio_bytes)
                except Exception as stt_error:
                    print(f"[STT] Failed transcription: {stt_error}")
                    final_user_text = final_user_text or ""
        final_user_text = norm(final_user_text)
        debug_enabled = is_truthy_string(debug) or is_truthy_string(os.getenv("INTERROGATION_DEBUG_RESPONSES", ""))

        case_id = norm(case_id)
        if not case_id and not case_json.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "missing_case_context"},
            )
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
        if not case_data:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_case_context",
                    "message": "Provide a valid case_id or case_json for interrogation.",
                },
            )
        if case_id and is_truthy_string(reset_progress):
            INTERROGATION_PROGRESS_CACHE.pop(case_id, None)
        prior_progress = _get_progress_state(case_id)
        prior_breakdown_probability = float(
            prior_progress.get("breakdown_probability", 0.0)
        )
        prior_turn_count = max(0, int(prior_progress.get("turn_count", 0) or 0))
        prior_cumulative_pressure = clamp01(
            prior_progress.get("cumulative_pressure", 0.0)
        )
        prior_statement_collapse_stage = max(
            0,
            min(5, int(safe_float(prior_progress.get("statement_collapse_stage", 0), 0.0))),
        )
        prior_pad_state = _normalize_pad_state_blob(
            prior_progress.get("pad_state", _case_baseline_pad_state(case_data))
        )
        prior_final_psychological_reaction = norm(
            prior_progress.get("final_psychological_reaction", "")
        )
        prior_selected_personality = _selected_personality_from_progress(prior_progress)
        prior_statement_records = _normalize_statement_records(
            prior_progress.get("statement_records", [])
        )
        prior_submitted_judgment = _normalize_submitted_judgment(
            prior_progress.get("submitted_judgment", {})
        )
        prior_final_psychological_report = _normalize_final_report_blob(
            prior_progress.get("final_psychological_report", {})
        )
        client_personality_payload = safe_json_loads(personality_json, None) if personality_json.strip() else None
        if personality_json.strip() and not isinstance(client_personality_payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_selected_personality",
                    "message": "personality_json must be a JSON object with all five Big Five values.",
                },
            )
        validated_selected_personality: Dict[str, float] = {}
        if isinstance(client_personality_payload, dict):
            validated_selected_personality, missing_traits = _validate_selected_personality_payload(
                client_personality_payload
            )
            if missing_traits:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_selected_personality",
                        "message": "personality_json must include all five Big Five values.",
                        "missing_traits": missing_traits,
                    },
                )
        active_selected_personality = (
            _resolve_selected_personality(
                case_data,
                prior_progress,
                validated_selected_personality,
            )
            if isinstance(client_personality_payload, dict)
            else prior_selected_personality
        )
        if not active_selected_personality:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing_selected_personality",
                    "message": "Select NPC personality before starting interrogation. Send personality_json to /interrogation/setup or include it in /interrogation/qna.",
                },
            )
        effective_case_data = _effective_case_from_progress(
            case_data,
            prior_progress,
            active_selected_personality,
        )
        suspect_case_context = build_case_context(effective_case_data, include_hidden_truth=False)
        prior_pad_state = _normalize_pad_state_blob(
            prior_progress.get("pad_state", _case_baseline_pad_state(effective_case_data))
        )
        prior_core_fact_exposed = bool(
            prior_progress.get("core_fact_exposed", False)
            or prior_statement_collapse_stage >= 5
            or prior_breakdown_probability >= BREAKDOWN_EXPOSURE_THRESHOLD
        )

        if prior_core_fact_exposed:
            msg = "이미 진술 붕괴가 발생했습니다. 이번 심문은 종료됐습니다."
            return JSONResponse(
                status_code=200,
                content={
                    "user_text": final_user_text,
                    "suspect_text": msg,
                    "pressure_delta": 0.0,
                    "breakdown_probability": prior_breakdown_probability,
                    "core_fact_exposed": True,
                    "fsm_state": EXPOSURE_FSM_STATE,
                    "stress_score": float(prior_progress.get("stress_score", 0.0)),
                    "cumulative_pressure": prior_cumulative_pressure,
                    "raw_odds": float(prior_progress.get("last_raw_odds", 0.0)),
                    "latest_sue_impact": float(prior_progress.get("last_sue_impact", 0.0)),
                    "statement_collapse_stage": prior_statement_collapse_stage,
                    "statement_collapse_label": _statement_collapse_label(prior_statement_collapse_stage),
                    "pad_state": prior_pad_state,
                    "final_psychological_reaction": prior_final_psychological_reaction,
                    "final_psychological_report": prior_final_psychological_report,
                    "judgment_ready": True,
                    "turn_count": prior_turn_count,
                    "audio_wav_b64": await tts_to_b64(msg),
                },
            )

        if prior_turn_count >= MAX_GAME_TURNS:
            msg = "이미 이번 심문은 종료됐습니다."
            return JSONResponse(
                status_code=200,
                content={
                    "user_text": final_user_text,
                    "suspect_text": msg,
                    "pressure_delta": 0.0,
                    "breakdown_probability": prior_breakdown_probability,
                    "core_fact_exposed": bool(prior_core_fact_exposed),
                    "fsm_state": norm(prior_progress.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE,
                    "stress_score": float(prior_progress.get("stress_score", 0.0)),
                    "cumulative_pressure": prior_cumulative_pressure,
                    "raw_odds": float(prior_progress.get("last_raw_odds", 0.0)),
                    "latest_sue_impact": float(prior_progress.get("last_sue_impact", 0.0)),
                    "statement_collapse_stage": prior_statement_collapse_stage,
                    "statement_collapse_label": _statement_collapse_label(prior_statement_collapse_stage),
                    "pad_state": prior_pad_state,
                    "final_psychological_reaction": prior_final_psychological_reaction,
                    "final_psychological_report": prior_final_psychological_report,
                    "judgment_ready": True,
                    "turn_count": prior_turn_count,
                    "audio_wav_b64": await tts_to_b64(msg),
                },
            )

        if not final_user_text:
            msg = "…잘 안 들립니다. 다시 말씀해 주세요."
            return JSONResponse(
                status_code=200,
                content={
                    "user_text": "",
                    "suspect_text": msg,
                    "pressure_delta": 0.0,
                    "breakdown_probability": prior_breakdown_probability,
                    "core_fact_exposed": bool(prior_core_fact_exposed),
                    "fsm_state": norm(prior_progress.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE,
                    "stress_score": float(prior_progress.get("stress_score", 0.0)),
                    "cumulative_pressure": prior_cumulative_pressure,
                    "raw_odds": float(prior_progress.get("last_raw_odds", 0.0)),
                    "latest_sue_impact": float(prior_progress.get("last_sue_impact", 0.0)),
                    "statement_collapse_stage": prior_statement_collapse_stage,
                    "statement_collapse_label": _statement_collapse_label(prior_statement_collapse_stage),
                    "pad_state": prior_pad_state,
                    "final_psychological_reaction": prior_final_psychological_reaction,
                    "final_psychological_report": prior_final_psychological_report,
                    "judgment_ready": False,
                    "turn_count": prior_turn_count,
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
                    "breakdown_probability": prior_breakdown_probability,
                    "core_fact_exposed": bool(prior_core_fact_exposed),
                    "fsm_state": norm(prior_progress.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE,
                    "stress_score": float(prior_progress.get("stress_score", 0.0)),
                    "cumulative_pressure": prior_cumulative_pressure,
                    "raw_odds": float(prior_progress.get("last_raw_odds", 0.0)),
                    "latest_sue_impact": float(prior_progress.get("last_sue_impact", 0.0)),
                    "statement_collapse_stage": prior_statement_collapse_stage,
                    "statement_collapse_label": _statement_collapse_label(prior_statement_collapse_stage),
                    "pad_state": prior_pad_state,
                    "final_psychological_reaction": prior_final_psychological_reaction,
                    "final_psychological_report": prior_final_psychological_report,
                    "judgment_ready": False,
                    "turn_count": prior_turn_count,
                    "audio_wav_b64": await tts_to_b64(msg),
                },
            )

        # 2) case load (cache 우선)
        # 3) calc pressure/prob
        question_analysis = llm_evaluate_interrogation(effective_case_data, history, final_user_text)
        repeated_question = detect_repeat(history, final_user_text)
        question_category = _classify_question_category(
            final_user_text,
            question_analysis,
            repeated_question,
        )
        core_fact_exposed = bool(prior_core_fact_exposed)
        current_behavior_state = norm(prior_progress.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE

        # 4) LLM answer
        suspect_text = llm_suspect_answer(
            suspect_case_context,
            effective_case_data,
            history,
            final_user_text,
            prior_breakdown_probability,
            question_analysis,
            current_behavior_state,
            prior_statement_collapse_stage,
            prior_pad_state,
        )

        suspect_text = trim_to_1_3_sentences(suspect_text)
        rule_based_turn = analyze_interrogation_turn_rule_based(
            effective_case_data,
            history,
            final_user_text,
            suspect_text,
            question_analysis,
        )
        dialogue_contradiction_signal = _empty_dialogue_contradiction_signal("not evaluated")
        if not core_fact_exposed:
            dialogue_contradiction_signal = detect_dialogue_contradiction_local(
                history,
                final_user_text,
                suspect_text,
            )
        rule_based_turn["soft_dialogue_contradiction"] = bool(
            dialogue_contradiction_signal.get("detective_highlighted")
            or dialogue_contradiction_signal.get("suspect_self_contradicted")
        ) and str(dialogue_contradiction_signal.get("severity", "none")).strip().lower() != "none"
        rule_based_turn["soft_dialogue_contradiction_signal"] = dialogue_contradiction_signal
        progress_eval = evaluate_interrogation_progress_v3(
            effective_case_data,
            history,
            final_user_text,
            question_analysis,
            prior_progress,
            rule_based_turn["hard_contradiction_ids"],
            dialogue_contradiction_signal,
        )
        if core_fact_exposed:
            progress_eval["breakdown_probability"] = max(
                float(progress_eval.get("breakdown_probability", 0.0)),
                prior_breakdown_probability,
            )
            progress_eval["core_fact_exposed"] = True
            progress_eval["fsm_state"] = EXPOSURE_FSM_STATE
            progress_eval["statement_collapse_stage"] = 5
            progress_eval["statement_collapse_label"] = _statement_collapse_label(5)
        pressure_delta = float(progress_eval["pressure_delta"])
        breakdown_probability = float(
            progress_eval.get("breakdown_probability", 0.0)
        )
        cumulative_evidence_ids = list(progress_eval["cumulative_evidence_ids"])
        cumulative_contradiction_ids = list(progress_eval["cumulative_contradiction_ids"])
        if not core_fact_exposed:
            if bool(progress_eval.get("core_fact_exposed", False)) or breakdown_probability >= BREAKDOWN_EXPOSURE_THRESHOLD:
                core_fact_exposed = True
                progress_eval["fsm_state"] = EXPOSURE_FSM_STATE
                progress_eval["breakdown_probability"] = breakdown_probability
                progress_eval["core_fact_exposed"] = True
                progress_eval["statement_collapse_stage"] = 5
                progress_eval["statement_collapse_label"] = _statement_collapse_label(5)
        else:
            progress_eval["pressure_delta"] = pressure_delta
            progress_eval["breakdown_probability"] = breakdown_probability
            progress_eval["core_fact_exposed"] = True

        # 5) TTS
        wav_b64 = await tts_to_b64(suspect_text)
        next_turn_count = prior_turn_count + 1
        progress_eval["pad_state"] = _normalize_pad_state_blob(
            progress_eval.get("pad_state", prior_pad_state)
        )
        statement_collapse_stage = max(
            0,
            min(5, int(safe_float(progress_eval.get("statement_collapse_stage", 0), 0.0))),
        )
        progress_eval["statement_collapse_stage"] = statement_collapse_stage
        progress_eval["statement_collapse_label"] = _statement_collapse_label(statement_collapse_stage)
        final_psychological_reaction = norm(progress_eval.get("final_psychological_reaction", ""))
        judgment_ready = bool(
            core_fact_exposed
            or next_turn_count >= MAX_GAME_TURNS
            or statement_collapse_stage >= 4
        )
        if core_fact_exposed or next_turn_count >= MAX_GAME_TURNS:
            final_psychological_reaction = _generate_final_psychological_reaction(
                effective_case_data,
                statement_collapse_stage,
                progress_eval["pad_state"],
                bool(core_fact_exposed),
            )
        if core_fact_exposed:
            suspect_text = trim_to_1_3_sentences(
                build_final_reaction_speech(
                    effective_case_data,
                    statement_collapse_stage,
                    progress_eval["pad_state"],
                    True,
                )
            )
        progress_eval["final_psychological_reaction"] = final_psychological_reaction
        statement_record = _build_statement_record(
            next_turn_count,
            final_user_text,
            question_analysis,
            suspect_text,
            rule_based_turn,
            progress_eval,
            repeated_question,
        )
        statement_records = _normalize_statement_records(
            prior_statement_records + [statement_record]
        )
        post_progress_state = {
            "breakdown_probability": breakdown_probability,
            "referenced_evidence_ids": cumulative_evidence_ids,
            "established_contradiction_ids": cumulative_contradiction_ids,
            "stress_score": progress_eval["stress_score"],
            "cooperation_score": progress_eval["cooperation_score"],
            "cumulative_pressure": progress_eval["cumulative_pressure"],
            "fsm_state": progress_eval["fsm_state"],
            "last_raw_odds": progress_eval["raw_odds"],
            "last_sue_impact": progress_eval["latest_sue_impact"],
            "statement_collapse_stage": statement_collapse_stage,
            "pad_state": progress_eval["pad_state"],
            "final_psychological_reaction": final_psychological_reaction,
            "selected_personality": active_selected_personality,
            "statement_records": statement_records,
            "submitted_judgment": prior_submitted_judgment,
            "turn_count": next_turn_count,
        }
        final_psychological_report = _build_final_psychological_report(
            effective_case_data,
            post_progress_state,
        )
        _store_progress_state(
            case_id,
            breakdown_probability,
            cumulative_evidence_ids,
            cumulative_contradiction_ids,
            progress_eval["stress_score"],
            progress_eval["cooperation_score"],
            progress_eval["cumulative_pressure"],
            progress_eval["fsm_state"],
            progress_eval["raw_odds"],
            progress_eval["latest_sue_impact"],
            next_turn_count,
            statement_collapse_stage,
            progress_eval["pad_state"],
            final_psychological_reaction,
            active_selected_personality,
            statement_records,
            prior_submitted_judgment,
            final_psychological_report,
        )

        response_content = {
            "user_text": final_user_text,
            "question_category": question_category["key"],
            "question_category_label": question_category["label"],
            "suspect_text": suspect_text,
            "pressure_delta": float(pressure_delta),
            "breakdown_probability": float(breakdown_probability),
            "core_fact_exposed": bool(core_fact_exposed),
            "fsm_state": progress_eval["fsm_state"],
            "stress_score": float(progress_eval["stress_score"]),
            "cumulative_pressure": float(progress_eval["cumulative_pressure"]),
            "raw_odds": float(progress_eval["raw_odds"]),
            "latest_sue_impact": float(progress_eval["latest_sue_impact"]),
            "statement_collapse_stage": statement_collapse_stage,
            "statement_collapse_label": progress_eval["statement_collapse_label"],
            "pad_state": progress_eval["pad_state"],
            "final_psychological_reaction": final_psychological_reaction,
            "final_psychological_report": final_psychological_report,
            "judgment_ready": judgment_ready,
            "turn_count": next_turn_count,
            "audio_wav_b64": wav_b64,
        }
        if debug_enabled:
            response_content["debug"] = {
                "signal_guide": {
                    "hard": "Case-data contradiction confirmed by server rules and evidence linkage.",
                    "soft": "Dialogue-flow wobble detected from suspect statement changes.",
                    "question_categories": QUESTION_CATEGORY_LABELS,
                },
                "question_analysis": {
                    **question_analysis,
                    "question_category": question_category["key"],
                    "question_category_label": question_category["label"],
                    "question_category_reason": question_category["reason"],
                },
                "rule_based_turn": rule_based_turn,
                "scoring_model": {
                    "player_intent": progress_eval["player_intent"],
                    "pressure_delta": float(progress_eval["pressure_delta"]),
                    "pressure_components": progress_eval.get("pressure_components", {}),
                    "hard_contradiction_ids": progress_eval.get("hard_contradiction_ids", []),
                    "soft_dialogue_contradiction": bool(progress_eval.get("soft_dialogue_contradiction", False)),
                    "soft_dialogue_severity": progress_eval.get("soft_dialogue_severity", "none"),
                    "stress_score": float(progress_eval["stress_score"]),
                    "cooperation_score": float(progress_eval["cooperation_score"]),
                    "cumulative_pressure": float(progress_eval["cumulative_pressure"]),
                    "turn_pressure_gain": float(progress_eval.get("turn_pressure_gain", 0.0)),
                    "defense_intelligence": float(progress_eval["defense_intelligence"]),
                    "latest_sue_impact": float(progress_eval["latest_sue_impact"]),
                    "raw_odds": float(progress_eval["raw_odds"]),
                    "breakdown_probability": float(
                        progress_eval.get("breakdown_probability", 0.0)
                    ),
                    "core_fact_exposed": bool(progress_eval.get("core_fact_exposed", False)),
                    "statement_collapse_stage": statement_collapse_stage,
                    "statement_collapse_label": progress_eval["statement_collapse_label"],
                    "pad_state": progress_eval["pad_state"],
                    "final_psychological_reaction": final_psychological_reaction,
                    "final_psychological_report": final_psychological_report,
                    "personality_response_factors": progress_eval.get("personality_response_factors", {}),
                    "personality_response_breakdown": progress_eval.get("personality_response_breakdown", {}),
                    "latest_statement_record": statement_record,
                    "judgment_ready": judgment_ready,
                    "fsm_state": progress_eval["fsm_state"],
                    "turn_count": next_turn_count,
                },
            }

        return JSONResponse(status_code=200, content=response_content)

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
