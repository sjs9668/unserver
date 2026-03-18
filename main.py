import os
import json
import base64
import traceback
import re
import hashlib
import random
import math
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

for _store_dir in (CASE_STORE_DIR,):
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
        "confession_probability": clamp01(blob.get("confession_probability", 0.0)),
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
        "fsm_state": norm(blob.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE,
        "last_raw_odds": safe_float(blob.get("last_raw_odds", 0.0), 0.0),
        "last_sue_impact": safe_float(blob.get("last_sue_impact", 0.0), 0.0),
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
MAX_MODEL_CONFESSION_GAIN_PER_TURN = 0.18
MAX_GAME_TURNS = 10
STRESS_IDLE_DECAY = 0.01
STRESS_WEAK_TURN_DECAY = 0.003
STRESS_REPEAT_DECAY = 0.01

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
        "low": 0.005,
        "medium": 0.01,
        "high": 0.02,
    },
    "suspect_self_contradicted": {
        "none": 0.0,
        "low": 0.01,
        "medium": 0.02,
        "high": 0.03,
    },
}

class InterrogationCore:
    def __init__(
        self,
        steepness: float = 1.15,
        offset: float = -3.4,
        alpha: float = 2.1,
        beta: float = 1.1,
        gamma: float = 0.9,
    ):
        self.w = steepness
        self.b = offset
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
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
        current_stress: float,
        defense_intelligence: float,
        latest_sue_impact: float,
        caught_contradictions: int,
    ) -> float:
        impact_multiplier = latest_sue_impact if caught_contradictions > 0 else 0.0
        return (
            (self.alpha * current_stress)
            - (self.beta * defense_intelligence)
            + (self.gamma * impact_multiplier)
        )

    def calculate_confession_probability(
        self,
        current_stress: float,
        defense_intelligence: float,
        caught_contradictions: int,
        latest_sue_impact: float,
    ) -> Tuple[float, float]:
        raw_odds = self.calculate_raw_odds(
            current_stress,
            defense_intelligence,
            latest_sue_impact,
            caught_contradictions,
        )
        sigmoid_input = (self.w * raw_odds) + self.b
        sigmoid_input = max(-60.0, min(60.0, sigmoid_input))
        p_confession = 1.0 / (1.0 + math.exp(-sigmoid_input))
        return raw_odds, clamp01(p_confession)

    def evaluate_fsm_state(
        self,
        p_confession: float,
        contradictions: int,
        player_intent: str,
        cooperation: float,
    ) -> str:
        if player_intent == "Intimidate" and cooperation < 0.2:
            return "Angry / Uncooperative"
        if p_confession >= 0.8 or (p_confession >= 0.6 and contradictions >= 2):
            return "Confession / Breakdown"
        if 0.4 <= p_confession < 0.8:
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

def _cap_turn_confession_probability(
    prior_confession_probability: float,
    model_probability: float,
    max_gain: float = MAX_MODEL_CONFESSION_GAIN_PER_TURN,
) -> float:
    capped_probability = min(
        clamp01(model_probability),
        clamp01(prior_confession_probability + max_gain),
    )
    return max(clamp01(prior_confession_probability), capped_probability)

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

    candidates = [evidence_id, name, description]
    lowered = norm(f"{name} {description}").lower()
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
    "crime_place": ("어디", "장소", "현장", "place", "location"),
    "weapon": ("무기", "흉기", "칼", "weapon"),
    "victim_relation": ("피해자", "관계", "사이", "relation"),
    "alibi_claim": ("알리바이", "뭐 했", "어디 있었", "집에", "alibi"),
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
    for slot_name, patterns in QUESTION_SLOT_HINTS.items():
        if any(pattern in normalized for pattern in patterns):
            return slot_name
    return None

def _backfill_question_analysis(
    case_data: Optional[Dict[str, Any]],
    user_text: str,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    backfilled = dict(analysis or {})
    lexical_slot = _detect_slot_from_text(user_text)
    lexical_evidence_ids = _lexical_evidence_hits(case_data, user_text)

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

def apply_dialogue_contradiction_bonus(
    pressure_delta: float,
    confession_probability: float,
    dialogue_signal: Optional[Dict[str, Any]],
    confession_triggered: bool,
) -> Tuple[float, float]:
    if confession_triggered:
        return clamp01(pressure_delta), clamp01(confession_probability)
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

    boosted_pressure = clamp01(min(MAX_TURN_PRESSURE_DELTA, pressure_delta + pressure_bonus))
    boosted_probability = clamp01(confession_probability + confession_bonus)

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

def _empty_progress_state() -> Dict[str, Any]:
    return {
        "confession_probability": 0.0,
        "referenced_evidence_ids": [],
        "established_contradiction_ids": [],
        "stress_score": 0.0,
        "cooperation_score": DEFAULT_COOPERATION_SCORE,
        "fsm_state": DEFAULT_FSM_STATE,
        "last_raw_odds": 0.0,
        "last_sue_impact": 0.0,
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
    confession_probability: float,
    evidence_ids: List[str],
    contradiction_ids: List[str],
    stress_score: float = 0.0,
    cooperation_score: float = DEFAULT_COOPERATION_SCORE,
    fsm_state: str = DEFAULT_FSM_STATE,
    last_raw_odds: float = 0.0,
    last_sue_impact: float = 0.0,
    turn_count: int = 0,
) -> None:
    if not case_id:
        return

    INTERROGATION_PROGRESS_CACHE[case_id] = normalize_progress_state(
        {
            "confession_probability": clamp01(confession_probability),
            "referenced_evidence_ids": sorted(
                {str(eid).strip() for eid in (evidence_ids or []) if str(eid).strip()}
            ),
            "established_contradiction_ids": sorted(
                {str(cid).strip() for cid in (contradiction_ids or []) if str(cid).strip()}
            ),
            "stress_score": clamp01(stress_score),
            "cooperation_score": clamp01(cooperation_score),
            "fsm_state": norm(fsm_state) or DEFAULT_FSM_STATE,
            "last_raw_odds": float(last_raw_odds),
            "last_sue_impact": float(last_sue_impact),
            "turn_count": max(0, int(turn_count or 0)),
        }
    )

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
    if contradiction_type == "claim_vs_truth":
        return False
    if contradiction_type == "claim_vs_evidence":
        return not related_evidence or not (mentioned_evidence_set & related_evidence)
    if contradiction_type == "timeline_mismatch":
        if target_slot not in {"crime_time", "crime_place", "last_seen_place"}:
            return True
        return not related_evidence or not (mentioned_evidence_set & related_evidence)
    if contradiction_type == "alibi_mismatch":
        if target_slot != "alibi_claim":
            return True
        return not related_evidence or not (mentioned_evidence_set & related_evidence)
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
        allowed_slots |= {"crime_place", "crime_time", "last_seen_place", "met_victim_that_day"}
    elif target_slot == "last_seen_place":
        allowed_slots |= {"crime_place"}

    contradiction_ids: List[str] = []
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

        contradiction_ids.append(contradiction_id)

    if slot_truth_matches and not contradiction_ids:
        return []
    return uniq_strings(contradiction_ids)

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
    return {
        "question_analysis": analysis,
        "target_slot": target_slot,
        "claimed_value": claimed_value,
        "claimed_value_confidence": round(claimed_value_confidence, 3),
        "truth_value": truth_slots.get(target_slot or "", ""),
        "mentioned_evidence_ids": mentioned_evidence_ids,
        "detected_contradiction_ids": contradiction_ids,
    }

def build_case_context(
    case_data: Optional[Dict[str, Any]],
    include_hidden_truth: bool = False,
) -> str:
    if not case_data:
        return "No case data.\n"

    overview = case_data.get("overview", {}) or {}
    suspect = case_data.get("suspect", {}) or {}
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
        + "- When pressured hard, the story may wobble slightly, but do not fully confess unless triggered.\n"
        + "\n======================\n"
    )

def evaluate_interrogation_progress_v3(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    interrogation_signal: Optional[Dict[str, Any]] = None,
    prior_progress: Optional[Dict[str, Any]] = None,
    contradiction_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not case_data:
        return {
            "pressure_delta": 0.0,
            "confession_probability": 0.0,
            "cumulative_evidence_ids": [],
            "cumulative_contradiction_ids": [],
            "stress_score": 0.0,
            "cooperation_score": DEFAULT_COOPERATION_SCORE,
            "defense_intelligence": 0.65,
            "latest_sue_impact": 0.0,
            "raw_odds": 0.0,
            "player_intent": "Neutral",
            "fsm_state": DEFAULT_FSM_STATE,
        }

    signal = interrogation_signal or llm_evaluate_interrogation(case_data, history, user_text)
    prior_progress = prior_progress or _empty_progress_state()

    prior_confession_probability = clamp01(prior_progress.get("confession_probability", 0.0))
    prior_stress_score = clamp01(prior_progress.get("stress_score", 0.0))
    prior_cooperation_score = clamp01(
        prior_progress.get("cooperation_score", DEFAULT_COOPERATION_SCORE)
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

    pressure_delta = (
        NEW_EVIDENCE_PRESSURE_DELTA * len(new_evidence_ids)
        + REPEATED_EVIDENCE_PRESSURE_DELTA * len(repeated_evidence_ids)
        + NEW_CONTRADICTION_PRESSURE_DELTA * len(new_contradiction_ids)
        + REPEATED_CONTRADICTION_PRESSURE_DELTA * len(repeated_contradiction_ids)
        + PRESSURE_LEVEL_BONUS.get(pressure_level, 0.0)
    )

    if not user_text or is_too_ambiguous(user_text):
        pressure_delta = 0.0
    elif detect_repeat(history, user_text):
        pressure_delta *= 0.35

    pressure_delta = clamp01(min(MAX_TURN_PRESSURE_DELTA, pressure_delta))
    stress_score = _update_stress_score(
        prior_stress_score,
        pressure_delta,
        pressure_level,
        current_evidence_ids,
        current_contradiction_ids,
        history,
        user_text,
    )

    player_intent = _infer_player_intent(signal)
    cooperation_score = _update_cooperation_score(
        prior_cooperation_score,
        player_intent,
        pressure_delta,
        len(new_evidence_ids),
        len(new_contradiction_ids),
        history,
        user_text,
    )
    defense_intelligence = _infer_defense_intelligence(case_data)
    core = _build_interrogation_core(case_data)
    latest_sue_impact = _calculate_latest_sue_impact(
        case_data,
        list(current_evidence_ids),
        list(current_contradiction_ids),
        core,
    )
    raw_odds, confession_probability = core.calculate_confession_probability(
        stress_score,
        defense_intelligence,
        len(cumulative_contradiction_ids),
        latest_sue_impact,
    )
    confession_probability = _cap_turn_confession_probability(
        prior_confession_probability,
        confession_probability,
    )

    if detect_repeat(history, user_text):
        confession_probability = min(
            confession_probability,
            prior_confession_probability + 0.02,
        )

    confession_probability = clamp01(confession_probability)
    fsm_state = core.evaluate_fsm_state(
        confession_probability,
        len(cumulative_contradiction_ids),
        player_intent,
        cooperation_score,
    )

    return {
        "pressure_delta": pressure_delta,
        "confession_probability": confession_probability,
        "cumulative_evidence_ids": sorted(cumulative_evidence_ids),
        "cumulative_contradiction_ids": sorted(cumulative_contradiction_ids),
        "stress_score": stress_score,
        "cooperation_score": cooperation_score,
        "defense_intelligence": defense_intelligence,
        "latest_sue_impact": latest_sue_impact,
        "raw_odds": raw_odds,
        "player_intent": player_intent,
        "fsm_state": fsm_state,
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
        "truth_slots": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "crime_time": {"type": "string"},
                "crime_place": {"type": "string"},
                "weapon": {"type": "string"},
                "victim_relation": {"type": "string"},
                "alibi_claim": {"type": "string"},
                "actual_action": {"type": "string"},
                "last_seen_place": {"type": "string"},
                "met_victim_that_day": {"type": "string"},
            },
            "required": TRUTH_SLOT_NAMES,
        },
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
            "minItems": 4,
            "maxItems": 5,
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
                    "slot": {
                        "type": "string",
                        "enum": TRUTH_SLOT_NAMES,
                    },
                    "truth_value": {"type": "string"},
                    "contradiction_type": {"type": "string"},
                },
                "required": [
                    "id",
                    "description",
                    "related_evidence",
                    "slot",
                    "truth_value",
                    "contradiction_type",
                ],
            },
            "minItems": 3,
            "maxItems": 4,
        },
    },
    "required": [
        "case_id",
        "overview",
        "motive",
        "crime_flow",
        "suspect",
        "false_statement",
        "truth_slots",
        "evidences",
        "contradictions",
    ],
}

def coerce_case_payload(case_blob: Any, fallback_case_id: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(case_blob, dict):
        return None

    overview = case_blob.get("overview", {}) if isinstance(case_blob.get("overview"), dict) else {}
    suspect = case_blob.get("suspect", {}) if isinstance(case_blob.get("suspect"), dict) else {}
    truth_slots = case_blob.get("truth_slots", {}) if isinstance(case_blob.get("truth_slots"), dict) else {}
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
        slot_name = norm(contradiction.get("slot", ""))
        if slot_name not in TRUTH_SLOT_NAMES:
            slot_name = ""
        truth_value = norm(contradiction.get("truth_value", ""))
        if slot_name == "met_victim_that_day":
            truth_value = _normalize_slot_value_for_slot(truth_value, slot_name)
        normalized_contradictions.append(
            {
                "id": norm(contradiction.get("id", "")),
                "description": norm(contradiction.get("description", "")),
                "related_evidence": uniq_strings([str(item).strip() for item in related_evidence]),
                "slot": slot_name,
                "truth_value": truth_value,
                "contradiction_type": norm(contradiction.get("contradiction_type", "")),
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
        "truth_slots": normalized_truth_slots,
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
    Generate a structured interrogation case while actively varying crime type.
    """
    system = (
        "You generate structured interrogation-game cases.\n"
        "Output must be valid JSON matching the provided schema.\n"
        "The example JSON is only a schema/style reference, not a crime-type template.\n"
        "Vary the crime type, setting, suspect relationship, and motive across generations.\n"
        "Do not keep generating arson or fire cases unless the target profile explicitly asks for one.\n"
        "truth_slots must always be present and must contain short canonical comparison values.\n"
        "truth_slots.met_victim_that_day must be either 'yes' or 'no'.\n"
        "evidences must contain 4 to 5 items.\n"
        "contradictions must contain 3 to 4 items.\n"
        "Each contradiction must include slot, truth_value, and contradiction_type.\n"
        "Each contradiction.related_evidence should reference at least one evidence id when possible.\n"
        "Each contradiction.slot should match the main truth slot being challenged.\n"
        "Each contradiction.truth_value should be the actual value the server can compare against.\n"
        "Use contradiction_type values such as claim_vs_evidence, claim_vs_truth, timeline_mismatch, or alibi_mismatch.\n"
        "Include at least one contradiction tied to the suspect's alibi or claimed whereabouts.\n"
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
        case_data = coerce_case_payload(json.loads(text))
        if not case_data:
            raise RuntimeError("Generated case payload could not be normalized")
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
            "- Let the story wobble a little. You may partially correct yourself, but do not fully confess."
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
        "Do not volunteer hidden truth-slot information unless the detective directly asks about that specific point.\n"
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
    behavior_state: str = DEFAULT_FSM_STATE,
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

    extra_guard = ""
    if turn_pressure_context:
        extra_guard += turn_pressure_context
    if has_current_contradiction:
        extra_guard += "\n- Answer the contradiction directly and first."
        extra_guard += "\n- Keep the reply polite, short, and focused on that inconsistency."
    elif pressure_level in {"medium", "high"}:
        extra_guard += "\n- Respond to the pressure point directly instead of broadening the story."
        extra_guard += "\n- Stay polite and give only one concrete detail."
    if behavior_state == "Angry / Uncooperative":
        extra_guard += "\n- Sound colder and more resistant. You may push back and cooperate less."
    elif behavior_state == "Pressured / Shaken":
        extra_guard += "\n- Let slight hesitation or a small wobble show in the wording."
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
        + f"\n[Current behavioral state] {behavior_state}\n"
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

    return out or "죄송하지만 그 질문에는 바로 답드리기 어렵습니다."

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

        case_context = build_case_context(case_data, include_hidden_truth=True)
        suspect_text = trim_to_1_3_sentences(
            llm_confession(case_context, history, final_user_text)
        )
        wav_b64 = await tts_to_b64(suspect_text)
        if case_id:
            _store_progress_state(
                case_id,
                1.0,
                [],
                [],
                1.0,
                0.0,
                "Confession / Breakdown",
                8.0,
                3.0,
                MAX_GAME_TURNS,
            )
        return JSONResponse(
            status_code=200,
            content={
                "user_text": final_user_text,
                "suspect_text": suspect_text,
                "pressure_delta": 0.0,
                "confession_probability": 1.0,
                "confession_triggered": True,
                "fsm_state": "Confession / Breakdown",
                "stress_score": 1.0,
                "raw_odds": 8.0,
                "latest_sue_impact": 3.0,
                "turn_count": MAX_GAME_TURNS,
                "debug_force_confession": True,
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
        suspect_case_context = build_case_context(case_data, include_hidden_truth=False)
        confession_case_context = build_case_context(case_data, include_hidden_truth=True)
        prior_progress = _get_progress_state(case_id)
        prior_confession_probability = float(prior_progress["confession_probability"])
        prior_turn_count = max(0, int(prior_progress.get("turn_count", 0) or 0))
        prior_confession_triggered = prior_confession_probability >= 0.85

        if prior_confession_triggered:
            msg = "이미 피의자가 자백했습니다. 이번 심문은 종료됐습니다."
            return JSONResponse(
                status_code=200,
                content={
                    "user_text": final_user_text,
                    "suspect_text": msg,
                    "pressure_delta": 0.0,
                    "confession_probability": prior_confession_probability,
                    "confession_triggered": True,
                    "fsm_state": "Confession / Breakdown",
                    "stress_score": float(prior_progress.get("stress_score", 0.0)),
                    "raw_odds": float(prior_progress.get("last_raw_odds", 0.0)),
                    "latest_sue_impact": float(prior_progress.get("last_sue_impact", 0.0)),
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
                    "confession_probability": prior_confession_probability,
                    "confession_triggered": bool(prior_confession_triggered),
                    "fsm_state": norm(prior_progress.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE,
                    "stress_score": float(prior_progress.get("stress_score", 0.0)),
                    "raw_odds": float(prior_progress.get("last_raw_odds", 0.0)),
                    "latest_sue_impact": float(prior_progress.get("last_sue_impact", 0.0)),
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
                    "confession_probability": prior_confession_probability,
                    "confession_triggered": bool(prior_confession_triggered),
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
                    "confession_probability": prior_confession_probability,
                    "confession_triggered": bool(prior_confession_triggered),
                    "turn_count": prior_turn_count,
                    "audio_wav_b64": await tts_to_b64(msg),
                },
            )

        # 2) case load (cache 우선)
        # 3) calc pressure/prob
        question_analysis = llm_evaluate_interrogation(case_data, history, final_user_text)
        confession_triggered = prior_confession_probability >= 0.85
        current_behavior_state = norm(prior_progress.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE

        # 4) LLM answer
        if confession_triggered:
            suspect_text = llm_confession(confession_case_context, history, final_user_text)
        else:
            suspect_text = llm_suspect_answer(
                suspect_case_context,
                case_data,
                history,
                final_user_text,
                prior_confession_probability,
                question_analysis,
                current_behavior_state,
            )

        suspect_text = trim_to_1_3_sentences(suspect_text)
        rule_based_turn = analyze_interrogation_turn_rule_based(
            case_data,
            history,
            final_user_text,
            suspect_text,
            question_analysis,
        )
        progress_eval = evaluate_interrogation_progress_v3(
            case_data,
            history,
            final_user_text,
            question_analysis,
            prior_progress,
            rule_based_turn["detected_contradiction_ids"],
        )
        if confession_triggered:
            progress_eval["confession_probability"] = max(
                float(progress_eval["confession_probability"]),
                prior_confession_probability,
            )
            progress_eval["fsm_state"] = "Confession / Breakdown"
        pressure_delta = float(progress_eval["pressure_delta"])
        confession_probability = float(progress_eval["confession_probability"])
        cumulative_evidence_ids = list(progress_eval["cumulative_evidence_ids"])
        cumulative_contradiction_ids = list(progress_eval["cumulative_contradiction_ids"])
        if not confession_triggered:
            dialogue_contradiction_signal = llm_evaluate_dialogue_contradiction(
                history,
                final_user_text,
                suspect_text,
            )
            base_pressure_delta = pressure_delta
            pressure_delta, confession_probability = apply_dialogue_contradiction_bonus(
                pressure_delta,
                confession_probability,
                dialogue_contradiction_signal,
                confession_triggered,
            )
            stress_bonus = max(0.0, pressure_delta - base_pressure_delta)
            progress_eval["pressure_delta"] = float(pressure_delta)
            progress_eval["stress_score"] = clamp01(progress_eval["stress_score"] + stress_bonus)
            core = _build_interrogation_core(case_data)
            raw_odds, model_probability = core.calculate_confession_probability(
                progress_eval["stress_score"],
                progress_eval["defense_intelligence"],
                len(progress_eval["cumulative_contradiction_ids"]),
                progress_eval["latest_sue_impact"],
            )
            progress_eval["raw_odds"] = raw_odds
            progress_eval["confession_probability"] = _cap_turn_confession_probability(
                prior_confession_probability,
                max(float(confession_probability), model_probability),
            )
            progress_eval["fsm_state"] = core.evaluate_fsm_state(
                progress_eval["confession_probability"],
                len(progress_eval["cumulative_contradiction_ids"]),
                progress_eval["player_intent"],
                progress_eval["cooperation_score"],
            )
            confession_probability = float(progress_eval["confession_probability"])
            if confession_probability >= 0.85:
                confession_triggered = True
                progress_eval["fsm_state"] = "Confession / Breakdown"
                progress_eval["confession_probability"] = confession_probability
                suspect_text = trim_to_1_3_sentences(
                    llm_confession(confession_case_context, history, final_user_text)
                )
        else:
            progress_eval["pressure_delta"] = pressure_delta
            progress_eval["confession_probability"] = confession_probability

        # 5) TTS
        wav_b64 = await tts_to_b64(suspect_text)
        next_turn_count = prior_turn_count + 1
        _store_progress_state(
            case_id,
            confession_probability,
            cumulative_evidence_ids,
            cumulative_contradiction_ids,
            progress_eval["stress_score"],
            progress_eval["cooperation_score"],
            progress_eval["fsm_state"],
            progress_eval["raw_odds"],
            progress_eval["latest_sue_impact"],
            next_turn_count,
        )

        response_content = {
            "user_text": final_user_text,
            "suspect_text": suspect_text,
            "pressure_delta": float(pressure_delta),
            "confession_probability": float(confession_probability),
            "confession_triggered": bool(confession_triggered),
            "fsm_state": progress_eval["fsm_state"],
            "stress_score": float(progress_eval["stress_score"]),
            "raw_odds": float(progress_eval["raw_odds"]),
            "latest_sue_impact": float(progress_eval["latest_sue_impact"]),
            "turn_count": next_turn_count,
            "audio_wav_b64": wav_b64,
        }
        if debug_enabled:
            response_content["debug"] = {
                "question_analysis": question_analysis,
                "rule_based_turn": rule_based_turn,
                "scoring_model": {
                    "player_intent": progress_eval["player_intent"],
                    "pressure_delta": float(progress_eval["pressure_delta"]),
                    "stress_score": float(progress_eval["stress_score"]),
                    "cooperation_score": float(progress_eval["cooperation_score"]),
                    "defense_intelligence": float(progress_eval["defense_intelligence"]),
                    "latest_sue_impact": float(progress_eval["latest_sue_impact"]),
                    "raw_odds": float(progress_eval["raw_odds"]),
                    "confession_probability": float(progress_eval["confession_probability"]),
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
