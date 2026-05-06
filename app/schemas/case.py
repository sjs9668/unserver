import copy
from typing import Any, Dict, Optional

from app.utils.text import norm
from app.schemas.interrogation import (
    HEXACO_TRAITS,
    _default_personality,
    _normalize_pad_state_blob,
    _normalize_personality_blob,
    _normalize_selected_personality_blob,
    _selected_personality_from_progress,
    _to_clamped_float,
)

CasePayload = Dict[str, Any]

def _case_default_personality(case_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data, dict) else {}
    personality = {}
    if isinstance(suspect_profile, dict):
        if isinstance(suspect_profile.get("default_personality"), dict):
            personality = suspect_profile.get("default_personality", {})
    normalized = _default_personality()
    for trait in HEXACO_TRAITS:
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


def _case_result_choice_key(case_data: Optional[Dict[str, Any]]) -> str:
    suspect_profile = case_data.get("suspect_profile", {}) if isinstance(case_data, dict) else {}
    role = norm(suspect_profile.get("case_role", ""))
    if any(keyword in role for keyword in ("주범", "실행범", "범인")):
        return "principal"
    if any(keyword in role for keyword in ("공범", "은폐", "교사", "방조")):
        return "accomplice_or_coverup"
    return "not_directly_involved"
