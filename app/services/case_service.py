import random
import traceback
from typing import Any, Dict, List, Optional

from fastapi.responses import JSONResponse

from app.config import PREBUILT_CASE_DIR
from app.storage.case_store import CASE_CACHE, persist_case
from app.utils.json import read_json_file
from app.utils.text import norm, safe_float, uniq_strings
from app.schemas.case import _case_default_personality
from app.schemas.interrogation import (
    HEXACO_TRAITS,
    PAD_STATE_FIELDS,
    _default_mental_state,
    _default_personality,
    _normalize_pad_state_blob,
    _normalize_selected_personality_blob,
    _normalize_string_list,
    _to_clamped_float,
)
from app.services.contradiction_service import _normalize_slot_value_for_slot

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

VALID_CONTRADICTION_TYPES = {
    "claim_vs_evidence",
    "claim_vs_truth",
    "timeline_mismatch",
    "alibi_mismatch",
}


def _default_truth_slots() -> Dict[str, str]:
    return {slot_name: "" for slot_name in TRUTH_SLOT_NAMES}

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
    for trait in HEXACO_TRAITS:
        normalized_personality[trait] = _to_clamped_float(default_personality.get(trait, normalized_personality[trait]))

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


async def case_generate():
    """
    Prebuilt case selection endpoint.

    This no longer generates a new case with the LLM.
    It returns 3 prebuilt case files and keeps "case" alongside "cases"
    for backwards compatibility with older clients.
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
