"""심문 API 전체 흐름을 조율하는 오케스트레이션 서비스.

한 턴의 처리 순서:
1. 음성 입력이 있으면 STT로 질문 텍스트 변환
2. 사건 데이터와 이전 심문 진행 상태 로드
3. 질문 의도/대상 슬롯/증거 언급/압박 수준 분석
4. NPC 답변 생성
5. 하드 모순(사건 데이터 충돌)과 소프트 모순(대화 중 진술 변화) 판정
6. 압박, PAD 감정 상태, 진술 붕괴 단계, 최종 심리 반응 갱신
7. 진술 기록을 저장하고 TTS 음성을 포함해 클라이언트에 반환
"""

import os
import traceback
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse

# =========================================================
from app.storage.case_store import (
    CASE_CACHE,
    load_case,
    persist_case,
)
from app.utils.json import safe_json_loads
from app.utils.text import (
    clamp01,
    detect_repeat,
    is_too_ambiguous,
    is_truthy_string,
    norm,
    norm_for_match,
    safe_float,
    trim_to_1_3_sentences,
    uniq_strings,
)

# =========================================================
# IN-MEMORY STORE (서버 재시작하면 초기화)
# =========================================================
from app.storage.progress_store import INTERROGATION_PROGRESS_CACHE

# 최종 서버 구조:
# - 사건은 LLM 즉석 생성이 아니라 cases/prebuilt의 사전 제작 JSON을 사용한다.
# - 심문 엔진은 질문 분석 -> 답변 생성 -> 모순 판정 -> 압박/붕괴 계산 순서로 동작한다.
# - documents/personality/PAD는 브리핑 UI와 심리 모델 계산에 연결되는 메타데이터다.

def normalize_progress_state(blob: Any) -> Dict[str, Any]:
    from app.schemas.interrogation import normalize_progress_state as _impl
    return _impl(blob)

# 질문 자체의 압박 강도. 증거/모순보다 작은 보정값으로 두어,
# 발표 내용처럼 "모순을 동반한 압박"이 더 크게 작동하게 한다.
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

# 누적 압박을 Sigmoid에 넣을 때 사용하는 기울기와 임계점.
# midpoint 근처를 넘으면 붕괴 확률이 빠르게 올라간다.
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

# 소프트 모순은 사건 데이터와 직접 충돌하지 않아도 진술이 흔들린 신호이므로
# 압박과 붕괴 진행에 별도 보너스를 준다.
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

def InterrogationCore(*args, **kwargs):
    from app.services.scoring_service import InterrogationCore as _InterrogationCore
    return _InterrogationCore(*args, **kwargs)


def _build_interrogation_core(case_data: Optional[Dict[str, Any]]):
    from app.services.scoring_service import _build_interrogation_core as _impl
    return _impl(case_data)


def _infer_defense_intelligence(case_data: Optional[Dict[str, Any]]) -> float:
    from app.services.scoring_service import _infer_defense_intelligence as _impl
    return _impl(case_data)


def _infer_player_intent(interrogation_signal: Optional[Dict[str, Any]]) -> str:
    from app.services.scoring_service import _infer_player_intent as _impl
    return _impl(interrogation_signal)


def _infer_evidence_specificity(evidence: Dict[str, Any]) -> str:
    from app.services.scoring_service import _infer_evidence_specificity as _impl
    return _impl(evidence)


def _infer_evidence_source(evidence: Dict[str, Any]) -> str:
    from app.services.scoring_service import _infer_evidence_source as _impl
    return _impl(evidence)


def _calculate_latest_sue_impact(
    case_data: Optional[Dict[str, Any]],
    mentioned_evidence_ids: List[str],
    contradiction_ids: List[str],
    core: Any,
) -> float:
    from app.services.scoring_service import _calculate_latest_sue_impact as _impl
    return _impl(case_data, mentioned_evidence_ids, contradiction_ids, core)


def _update_cooperation_score(
    prior_cooperation: float,
    player_intent: str,
    pressure_delta: float,
    new_evidence_count: int,
    new_contradiction_count: int,
    history: List[Dict[str, Any]],
    user_text: str,
) -> float:
    from app.services.scoring_service import _update_cooperation_score as _impl
    return _impl(
        prior_cooperation,
        player_intent,
        pressure_delta,
        new_evidence_count,
        new_contradiction_count,
        history,
        user_text,
    )


def _update_stress_score(
    prior_stress_score: float,
    pressure_delta: float,
    pressure_level: str,
    current_evidence_ids: set,
    current_contradiction_ids: set,
    history: List[Dict[str, Any]],
    user_text: str,
) -> float:
    from app.services.scoring_service import _update_stress_score as _impl
    return _impl(
        prior_stress_score,
        pressure_delta,
        pressure_level,
        current_evidence_ids,
        current_contradiction_ids,
        history,
        user_text,
    )

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
    """LLM 질문 분석 결과를 발표 자료의 질문 유형 라벨로 변환한다."""
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
    """LLM 구조화 결과가 서버 허용 범위를 벗어나지 않도록 보정한다."""
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
    """LLM이 놓친 슬롯/증거/압박 단서를 로컬 키워드 규칙으로 보강한다."""
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
    """LLM 질문 분석 실패 시에도 게임이 멈추지 않도록 로컬 규칙으로 분석한다."""
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
    from app.services.openai_service import llm_evaluate_interrogation as _impl
    return _impl(case_data, history, user_text)

def _dialogue_contradiction_bonus_values(
    dialogue_signal: Optional[Dict[str, Any]],
) -> Tuple[float, float, bool]:
    from app.services.scoring_service import _dialogue_contradiction_bonus_values as _impl
    return _impl(dialogue_signal)

def _default_suspect_reply_review(
    reason: str = "",
    answers_question_directly: bool = True,
) -> Dict[str, Any]:
    from app.services.contradiction_service import _default_suspect_reply_review as _impl
    return _impl(reason, answers_question_directly)

def _empty_progress_state() -> Dict[str, Any]:
    from app.storage.progress_store import _empty_progress_state as _impl
    return _impl()


def _get_progress_state(case_id: str) -> Dict[str, Any]:
    from app.storage.progress_store import _get_progress_state as _impl
    return _impl(case_id)


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
    from app.storage.progress_store import _store_progress_state as _impl
    return _impl(
        case_id,
        breakdown_probability,
        evidence_ids,
        contradiction_ids,
        stress_score,
        cooperation_score,
        cumulative_pressure,
        fsm_state,
        last_raw_odds,
        last_sue_impact,
        turn_count,
        statement_collapse_stage,
        pad_state,
        final_psychological_reaction,
        selected_personality,
        statement_records,
        submitted_judgment,
        final_psychological_report,
    )


def _persist_progress_snapshot(case_id: str, progress_state: Dict[str, Any]) -> Dict[str, Any]:
    from app.storage.progress_store import _persist_progress_snapshot as _impl
    return _impl(case_id, progress_state)

def normalize_slot_value(value: str) -> str:
    from app.services.contradiction_service import normalize_slot_value as _impl
    return _impl(value)


def _normalize_slot_value_for_slot(value: str, target_slot: str = "") -> str:
    from app.services.contradiction_service import _normalize_slot_value_for_slot as _impl
    return _impl(value, target_slot)


def extract_claimed_slot_value(answer: str, target_slot: str, case_data: Optional[Dict[str, Any]]) -> str:
    from app.services.contradiction_service import extract_claimed_slot_value as _impl
    return _impl(answer, target_slot, case_data)


def detect_contradictions_from_slot(
    target_slot: str,
    claimed_value: str,
    case_data: Optional[Dict[str, Any]],
    mentioned_evidence_ids: List[str],
) -> List[str]:
    from app.services.contradiction_service import detect_contradictions_from_slot as _impl
    return _impl(target_slot, claimed_value, case_data, mentioned_evidence_ids)


def analyze_interrogation_turn_rule_based(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    suspect_text: str,
    question_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from app.services.contradiction_service import analyze_interrogation_turn_rule_based as _impl
    return _impl(case_data, history, user_text, suspect_text, question_analysis)

def build_case_context(
    case_data: Optional[Dict[str, Any]],
    include_hidden_truth: bool = False,
) -> str:
    """NPC 답변 생성 프롬프트에 넣을 사건 브리핑 문자열을 만든다."""
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
                for trait in HEXACO_TRAITS
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
    from app.services.scoring_service import evaluate_interrogation_progress_v3 as _impl
    return _impl(
        case_data,
        history,
        user_text,
        interrogation_signal,
        prior_progress,
        contradiction_ids,
        dialogue_contradiction_signal,
    )

def stt_transcribe(audio_bytes: bytes) -> str:
    from app.services.openai_service import stt_transcribe as _impl
    return _impl(audio_bytes)


async def tts_to_b64(text: str) -> str:
    from app.services.openai_service import tts_to_b64 as _impl
    return await _impl(text)

HEXACO_TRAITS = (
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness_to_experience",
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
    from app.schemas.interrogation import _to_clamped_float as _impl
    return _impl(value, default)


def _normalize_string_list(values: Any) -> List[str]:
    from app.schemas.interrogation import _normalize_string_list as _impl
    return _impl(values)


def _default_personality() -> Dict[str, float]:
    from app.schemas.interrogation import _default_personality as _impl
    return _impl()


def _default_mental_state() -> Dict[str, float]:
    from app.schemas.interrogation import _default_mental_state as _impl
    return _impl()


def _normalize_pad_state_blob(blob: Any) -> Dict[str, float]:
    from app.schemas.interrogation import _normalize_pad_state_blob as _impl
    return _impl(blob)


def _normalize_personality_blob(blob: Any) -> Dict[str, float]:
    from app.schemas.interrogation import _normalize_personality_blob as _impl
    return _impl(blob)


def _missing_hexaco_traits(blob: Any) -> List[str]:
    from app.schemas.interrogation import _missing_hexaco_traits as _impl
    return _impl(blob)


def _validate_selected_personality_payload(blob: Any) -> Tuple[Dict[str, float], List[str]]:
    from app.schemas.interrogation import _validate_selected_personality_payload as _impl
    return _impl(blob)


def _normalize_selected_personality_blob(blob: Any) -> Dict[str, float]:
    from app.schemas.interrogation import _normalize_selected_personality_blob as _impl
    return _impl(blob)


def _selected_personality_from_progress(progress_state: Optional[Dict[str, Any]]) -> Dict[str, float]:
    from app.schemas.interrogation import _selected_personality_from_progress as _impl
    return _impl(progress_state)


def _resolve_selected_personality(
    case_data: Optional[Dict[str, Any]],
    progress_state: Optional[Dict[str, Any]] = None,
    personality_blob: Any = None,
) -> Dict[str, float]:
    from app.schemas.interrogation import _resolve_selected_personality as _impl
    return _impl(case_data, progress_state, personality_blob)


def _normalize_statement_record(record: Any) -> Dict[str, Any]:
    from app.schemas.interrogation import _normalize_statement_record as _impl
    return _impl(record, _classify_question_category)


def _normalize_statement_records(values: Any) -> List[Dict[str, Any]]:
    from app.schemas.interrogation import _normalize_statement_records as _impl
    return _impl(values, _classify_question_category)


def _normalize_judgment_choice(value: Any) -> str:
    from app.schemas.interrogation import _normalize_judgment_choice as _impl
    return _impl(value)


def _normalize_submitted_judgment(blob: Any) -> Dict[str, Any]:
    from app.schemas.interrogation import _normalize_submitted_judgment as _impl
    return _impl(blob)


def _normalize_final_report_blob(blob: Any) -> Dict[str, Any]:
    from app.schemas.interrogation import _normalize_final_report_blob as _impl
    return _impl(blob)


def _case_default_personality(case_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    from app.schemas.case import _case_default_personality as _impl
    return _impl(case_data)


def _case_personality(case_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    from app.schemas.case import _case_personality as _impl
    return _impl(case_data)


def _case_baseline_pad_state(case_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    from app.schemas.case import _case_baseline_pad_state as _impl
    return _impl(case_data)


def _apply_selected_personality_to_case(
    case_data: Optional[Dict[str, Any]],
    selected_personality: Optional[Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    from app.schemas.case import _apply_selected_personality_to_case as _impl
    return _impl(case_data, selected_personality)


def _effective_case_from_progress(
    case_data: Optional[Dict[str, Any]],
    progress_state: Optional[Dict[str, Any]] = None,
    selected_personality: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    from app.schemas.case import _effective_case_from_progress as _impl
    return _impl(case_data, progress_state, selected_personality)


def _extract_core_claim_text(suspect_text: str, claimed_value: str = "") -> str:
    from app.schemas.interrogation import _extract_core_claim_text as _impl
    return _impl(suspect_text, claimed_value)


def _case_result_choice_key(case_data: Optional[Dict[str, Any]]) -> str:
    from app.schemas.case import _case_result_choice_key as _impl
    return _impl(case_data)


def _build_final_psychological_report(
    case_data: Optional[Dict[str, Any]],
    progress_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    from app.schemas.interrogation import _build_final_psychological_report as _impl
    return _impl(case_data, progress_state)


def _build_statement_record(
    turn_index: int,
    user_text: str,
    question_analysis: Dict[str, Any],
    suspect_text: str,
    rule_based_turn: Dict[str, Any],
    progress_eval: Dict[str, Any],
    repeated_question: bool = False,
) -> Dict[str, Any]:
    from app.schemas.interrogation import _build_statement_record as _impl
    return _impl(
        turn_index,
        user_text,
        question_analysis,
        suspect_text,
        rule_based_turn,
        progress_eval,
        repeated_question,
        _classify_question_category,
    )


def _statement_collapse_label(stage: int) -> str:
    from app.schemas.interrogation import _statement_collapse_label as _impl
    return _impl(stage)


def _calculate_personality_response_factors(
    case_data: Optional[Dict[str, Any]],
    player_intent: str,
) -> Dict[str, float]:
    from app.services.scoring_service import _calculate_personality_response_factors as _impl
    return _impl(case_data, player_intent)


def _build_personality_speaking_directives(case_data: Optional[Dict[str, Any]]) -> List[str]:
    from app.services.openai_service import _build_personality_speaking_directives as _impl
    return _impl(case_data)


def _postprocess_reply_by_personality(
    text: str,
    case_data: Optional[Dict[str, Any]],
    pressure_level: str,
    has_current_contradiction: bool,
) -> str:
    from app.services.openai_service import _postprocess_reply_by_personality as _impl
    return _impl(text, case_data, pressure_level, has_current_contradiction)


def _build_personality_response_breakdown(
    case_data: Optional[Dict[str, Any]],
    player_intent: str,
) -> Dict[str, Any]:
    from app.services.scoring_service import _build_personality_response_breakdown as _impl
    return _impl(case_data, player_intent)


def _apply_personality_scaled_delta(
    prior_value: float,
    updated_value: float,
    multiplier: float,
) -> float:
    from app.services.scoring_service import _apply_personality_scaled_delta as _impl
    return _impl(prior_value, updated_value, multiplier)


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
    from app.services.scoring_service import _update_pad_state as _impl
    return _impl(
        prior_pad_state,
        case_data,
        player_intent,
        pressure_delta,
        new_evidence_count,
        new_contradiction_count,
        soft_dialogue_contradiction,
        repeated_question,
    )


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
    from app.services.scoring_service import _calculate_statement_collapse_stage as _impl
    return _impl(
        prior_stage,
        cumulative_pressure,
        breakdown_probability,
        cumulative_contradictions_count,
        current_hard_contradictions_count,
        soft_dialogue_contradiction,
        pad_state,
        case_data,
    )


def _soft_dialogue_stage_floor(
    dialogue_contradiction_signal: Optional[Dict[str, Any]],
    cumulative_pressure: float,
    pad_state: Dict[str, float],
    repeated_question: bool,
) -> int:
    from app.services.scoring_service import _soft_dialogue_stage_floor as _impl
    return _impl(dialogue_contradiction_signal, cumulative_pressure, pad_state, repeated_question)


def _generate_final_psychological_reaction(
    case_data: Optional[Dict[str, Any]],
    statement_collapse_stage: int,
    pad_state: Dict[str, float],
    core_fact_exposed: bool,
) -> str:
    from app.schemas.interrogation import _generate_final_psychological_reaction as _impl
    return _impl(case_data, statement_collapse_stage, pad_state, core_fact_exposed)


def build_final_reaction_speech(
    case_data: Optional[Dict[str, Any]],
    statement_collapse_stage: int,
    pad_state: Dict[str, float],
    core_fact_exposed: bool,
) -> str:
    from app.schemas.interrogation import build_final_reaction_speech as _impl
    return _impl(case_data, statement_collapse_stage, pad_state, core_fact_exposed)

def _normalize_documents(
    raw_documents: Any,
    overview: Dict[str, Any],
    suspect: Dict[str, Any],
) -> Dict[str, Any]:
    from app.services.case_service import _normalize_documents as _impl
    return _impl(raw_documents, overview, suspect)


def coerce_case_payload(case_blob: Any, fallback_case_id: str = "") -> Optional[Dict[str, Any]]:
    from app.services.case_service import coerce_case_payload as _impl
    return _impl(case_blob, fallback_case_id)


def load_prebuilt_case_library() -> List[Dict[str, Any]]:
    from app.services.case_service import load_prebuilt_case_library as _impl
    return _impl()


def pick_prebuilt_case_choices(num_choices: int = 3) -> List[Dict[str, Any]]:
    from app.services.case_service import pick_prebuilt_case_choices as _impl
    return _impl(num_choices)


def _public_case_briefing(case_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from app.services.case_service import _public_case_briefing as _impl
    return _impl(case_data)

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
    from app.services.openai_service import llm_suspect_answer as _impl
    return _impl(
        case_context,
        case_data,
        history,
        user_text,
        breakdown_probability,
        interrogation_signal,
        behavior_state,
        statement_collapse_stage,
        pad_state,
    )

def llm_review_suspect_reply(user_text: str, suspect_text: str) -> Dict[str, Any]:
    from app.services.contradiction_service import llm_review_suspect_reply as _impl
    return _impl(user_text, suspect_text)


def detect_dialogue_contradiction_local(
    history: List[Dict[str, Any]],
    user_text: str,
    suspect_text: str,
) -> Dict[str, Any]:
    from app.services.contradiction_service import detect_dialogue_contradiction_local as _impl
    return _impl(history, user_text, suspect_text)

def _resolve_case_from_request(
    case_id: str,
    case_json: str = "",
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """요청의 case_id 또는 case_json으로 이번 심문에 사용할 사건 데이터를 찾는다."""
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
async def case_generate():
    from app.services.case_service import case_generate as _impl
    return await _impl()

async def interrogation_setup(
    case_id: str = Form(""),
    case_json: str = Form(""),
    personality_json: str = Form(""),
    reset_progress: str = Form(""),
):
    """심문 시작 전 플레이어가 선택한 NPC HEXACO 성향을 진행 상태에 저장한다."""
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
                    "message": "personality_json must be a JSON object with all six HEXACO values.",
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
                    "message": "personality_json must include all six HEXACO values.",
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

async def interrogation_judgment_submit(
    case_id: str = Form(""),
    case_json: str = Form(""),
    judgment: str = Form(""),
    notes: str = Form(""),
):
    """심문 종료 후 플레이어의 최종 판단을 저장하고 실제 정답과 비교한다."""
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
        if int(progress_state.get("turn_count", 0) or 0) < MAX_GAME_TURNS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "judgment_not_ready",
                    "message": "Judgment can be submitted only after the interrogation reaches the turn limit.",
                    "turn_count": int(progress_state.get("turn_count", 0) or 0),
                    "max_turns": MAX_GAME_TURNS,
                },
            )
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

async def interrogation_report(
    case_id: str = Form(""),
    case_json: str = Form(""),
):
    """현재까지의 진술 기록과 최종 심리 리포트를 반환한다."""
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

async def interrogation_result_reveal(
    case_id: str = Form(""),
    case_json: str = Form(""),
):
    """플레이어 판단 이후 사건 파일에 저장된 실제 역할을 공개한다."""
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

async def interrogation_debug_confess(
    case_id: str = Form(""),
    case_json: str = Form(""),
    history_json: str = Form("[]"),
    user_text: str = Form(""),
):
    """시연/디버그용으로 강제 붕괴 상태를 만들어 응답을 확인하는 엔드포인트."""
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
    플레이어의 한 턴 질문을 처리하는 핵심 엔드포인트.

    Request(Form-data):
      - file: wav (선택, 음성 질문)
      - user_text: text (선택, 파일이 없을 때 직접 질문)
      - case_id/case_json: 심문할 사건
      - personality_json: 선택된 HEXACO 성향
      - history_json: 이전 대화 기록

    Response(JSON):
      - user_text / suspect_text
      - question_category
      - pressure_delta / cumulative_pressure / breakdown_probability
      - statement_collapse_stage / pad_state / final_psychological_report
      - audio_wav_b64
    """
    try:
        # 0) 클라이언트가 보내 준 최근 대화 기록을 정리한다.
        history = safe_json_loads(history_json, [])
        if not isinstance(history, list):
            history = []
        history = history[-20:]  # 너무 길면 토큰/비용 폭발 방지

        # 1) 입력 질문 확보: 음성 파일이 있으면 STT, 없으면 user_text를 사용한다.
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
        # 2) 사건 데이터 확보: case_json이 오면 정규화해 저장하고, 아니면 case_id로 로드한다.
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
        # 3) 이전 턴까지 누적된 압박/모순/PAD/진술 기록을 불러온다.
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
        # 4) NPC 성향은 심문 시작 전 고정값이다. 요청값이 있으면 검증하고,
        # 없으면 setup 단계에서 저장된 selected_personality를 사용한다.
        client_personality_payload = safe_json_loads(personality_json, None) if personality_json.strip() else None
        if personality_json.strip() and not isinstance(client_personality_payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_selected_personality",
                    "message": "personality_json must be a JSON object with all six HEXACO values.",
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
                        "message": "personality_json must include all six HEXACO values.",
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
        # 선택된 HEXACO 성향을 사건 복사본에 반영해 이후 모든 계산과 답변 생성에 사용한다.
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

        # 최대 턴 수를 넘으면 추가 심문 대신 판단 단계로 넘어가도록 고정 응답을 보낸다.
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

        # 5) 질문 분석: 자유 발화 질문을 의도, 대상 슬롯, 증거 언급, 압박 수준으로 구조화한다.
        question_analysis = llm_evaluate_interrogation(effective_case_data, history, final_user_text)
        repeated_question = detect_repeat(history, final_user_text)
        question_category = _classify_question_category(
            final_user_text,
            question_analysis,
            repeated_question,
        )
        core_fact_exposed = bool(prior_core_fact_exposed)
        current_behavior_state = norm(prior_progress.get("fsm_state", DEFAULT_FSM_STATE)) or DEFAULT_FSM_STATE

        # 6) NPC 답변 생성: 사건 브리핑, HEXACO, PAD, 붕괴 단계, 질문 분석 결과를 프롬프트에 반영한다.
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

        # 7) 모순 판정: 하드 모순은 사건 데이터 룰 기반, 소프트 모순은 최근 진술 변화 기반이다.
        suspect_text = trim_to_1_3_sentences(suspect_text)
        rule_based_turn = analyze_interrogation_turn_rule_based(
            effective_case_data,
            history,
            final_user_text,
            suspect_text,
            question_analysis,
        )
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
        # 8) 압박/심리/붕괴 계산: 모순과 증거 누적값을 PAD와 붕괴 확률로 변환한다.
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

        # 9) TTS: 클라이언트가 바로 재생할 수 있도록 NPC 답변을 WAV base64로 반환한다.
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
        judgment_ready = bool(next_turn_count >= MAX_GAME_TURNS)
        # 10턴에 도달하면 최종 심리 반응을 계산하고, 플레이어 판단 버튼을 활성화한다.
        if next_turn_count >= MAX_GAME_TURNS:
            final_psychological_reaction = _generate_final_psychological_reaction(
                effective_case_data,
                statement_collapse_stage,
                progress_eval["pad_state"],
                bool(core_fact_exposed),
            )
        if next_turn_count >= MAX_GAME_TURNS:
            suspect_text = trim_to_1_3_sentences(
                build_final_reaction_speech(
                    effective_case_data,
                    statement_collapse_stage,
                    progress_eval["pad_state"],
                    bool(core_fact_exposed),
                )
            )
        progress_eval["final_psychological_reaction"] = final_psychological_reaction
        # 10) 이번 턴 질문/답변/핵심 주장/모순/PAD 상태를 진술 기록에 추가한다.
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
        # 11) 최종 판단 화면에서 보여 줄 심리 리포트를 최신 진행 상태 기준으로 갱신한다.
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

        # 12) Unreal 클라이언트가 UI와 음성을 갱신할 수 있도록 한 번에 응답한다.
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
        # 디버그 응답은 통계 분석과 시연 검증용으로 압박 세부 항목까지 포함한다.
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
