"""심문 진행 상태 저장소.

턴별 압박도, 누적 증거/모순, PAD 상태, 진술 기록을 case_id별로 보관한다.
최종 제출 버전에서는 메모리 저장소를 사용하므로 서버 재시작 시 진행 상태는 초기화된다.
"""

from typing import Any, Dict, List, Optional

from app.schemas.interrogation import (
    BREAKDOWN_EXPOSURE_THRESHOLD,
    DEFAULT_COOPERATION_SCORE,
    DEFAULT_FSM_STATE,
    _default_mental_state,
    _normalize_final_report_blob,
    _normalize_pad_state_blob,
    _normalize_selected_personality_blob,
    _normalize_statement_records,
    _normalize_submitted_judgment,
    normalize_progress_state,
)
from app.utils.text import clamp01, norm

INTERROGATION_PROGRESS_CACHE: Dict[str, Dict[str, Any]] = {}


def _empty_progress_state() -> Dict[str, Any]:
    """새 심문을 시작할 때 사용하는 기본 진행 상태."""
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
    """case_id에 연결된 진행 상태를 읽고, 깨진 값은 정규화한다."""
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
    """한 턴 처리 후의 심문 진행 상태를 표준 형태로 저장한다."""
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
    """외부에서 만든 진행 상태 스냅샷도 같은 정규화 경로로 저장한다."""
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
