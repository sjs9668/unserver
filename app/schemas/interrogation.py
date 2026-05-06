import re
from typing import Any, Dict, List, Optional, Tuple

from app.utils.text import clamp01, norm, safe_float, trim_to_1_3_sentences, uniq_strings

DEFAULT_COOPERATION_SCORE = 0.55
DEFAULT_FSM_STATE = "Idle / Evasion"
BREAKDOWN_EXPOSURE_THRESHOLD = 0.85
MAX_GAME_TURNS = 10

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

InterrogationProgress = Dict[str, Any]
StatementRecords = List[Dict[str, Any]]

QUESTION_CATEGORY_LABELS = {
    "basic_fact": "기본 사실 질문",
    "statement_lock": "진술 고정 질문",
    "pressure": "압박 질문",
    "confirmation": "확인 질문",
    "repeat": "반복 질문",
    "misc": "기타 질문",
}

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
    return {trait: 0.5 for trait in HEXACO_TRAITS}

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
    for trait in HEXACO_TRAITS:
        normalized[trait] = _to_clamped_float(raw.get(trait, normalized[trait]), normalized[trait])
    return normalized

def _missing_hexaco_traits(blob: Any) -> List[str]:
    raw = blob if isinstance(blob, dict) else {}
    return [trait for trait in HEXACO_TRAITS if trait not in raw]

def _validate_selected_personality_payload(blob: Any) -> Tuple[Dict[str, float], List[str]]:
    missing_traits = _missing_hexaco_traits(blob)
    if missing_traits:
        return {}, missing_traits
    return _normalize_personality_blob(blob), []

def _normalize_selected_personality_blob(blob: Any) -> Dict[str, float]:
    raw = blob if isinstance(blob, dict) else {}
    if _missing_hexaco_traits(raw):
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
    if _missing_hexaco_traits(raw):
        return {}
    return _normalize_personality_blob(raw)

def _normalize_statement_record(record: Any, classify_question_category: Any = None) -> Dict[str, Any]:
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
        if callable(classify_question_category):
            question_category = classify_question_category(
                norm(raw.get("question_text", "")),
                {
                    "intent": raw.get("question_intent", ""),
                    "target_slot": raw.get("target_slot", ""),
                    "pressure_level": raw.get("pressure_level", ""),
                },
                False,
            )["key"]
        else:
            question_category = "misc"

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

def _normalize_statement_records(values: Any, classify_question_category: Any = None) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    records = [_normalize_statement_record(value, classify_question_category) for value in values]
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
    from app.schemas.case import _effective_case_from_progress, _case_personality

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
    classify_question_category: Any = None,
) -> Dict[str, Any]:
    stage = max(0, min(5, int(safe_float(progress_eval.get("statement_collapse_stage", 0), 0.0))))
    if callable(classify_question_category):
        question_category = classify_question_category(
            user_text,
            question_analysis,
            repeated_question,
        )
    else:
        key = norm(question_analysis.get("question_category", "")) or "misc"
        if key not in QUESTION_CATEGORY_LABELS:
            key = "misc"
        question_category = {"key": key, "label": QUESTION_CATEGORY_LABELS.get(key, QUESTION_CATEGORY_LABELS["misc"])}
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
        },
        classify_question_category,
    )

def _statement_collapse_label(stage: int) -> str:
    return STATEMENT_COLLAPSE_LABELS.get(max(0, min(5, int(stage))), STATEMENT_COLLAPSE_LABELS[0])

def _generate_final_psychological_reaction(
    case_data: Optional[Dict[str, Any]],
    statement_collapse_stage: int,
    pad_state: Dict[str, float],
    core_fact_exposed: bool,
) -> str:
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data, dict) else {}
    role = norm(suspect_profile.get("case_role", "용의자")) or "용의자"
    from app.schemas.case import _case_personality

    personality = _case_personality(case_data)
    emotionality = personality["emotionality"]
    conscientiousness = personality["conscientiousness"]
    agreeableness = personality["agreeableness"]
    arousal = pad_state.get("arousal", 0.5)
    dominance = pad_state.get("dominance", 0.5)
    pleasure = pad_state.get("pleasure", 0.5)

    if core_fact_exposed or statement_collapse_stage >= 5:
        if emotionality >= 0.65:
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
    from app.schemas.case import _case_personality

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
