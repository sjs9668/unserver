"""심문 압박, 붕괴 확률, PAD 감정 상태를 계산하는 서비스.

이 모듈은 발표의 핵심 로직인 "증거/모순/질문 강도 -> 누적 압박 ->
Sigmoid 붕괴 확률 -> 진술 붕괴 단계/PAD 변화"를 담당한다.
HEXACO 성향은 고정값으로 읽어 압박 민감도, 협조도, 말 길이, 붕괴 저항에 반영한다.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from app.utils.text import clamp01, detect_repeat, is_too_ambiguous, norm, safe_float
from app.services.interrogation_service import (
    BREAKDOWN_EXPOSURE_THRESHOLD,
    CORE_QUESTION_PRESSURE_BONUS,
    DEFAULT_COOPERATION_SCORE,
    DEFAULT_FSM_STATE,
    DIALOGUE_CONTRADICTION_CONFESSION_BONUS,
    DIALOGUE_CONTRADICTION_PRESSURE_BONUS,
    EXPOSURE_FSM_STATE,
    HIGH_IMPACT_PRESSURED_MIN_BREAKDOWN,
    HIGH_IMPACT_SUE_THRESHOLD,
    HIGH_DEFENSE_JOB_HINTS,
    HIGH_SOURCE_EVIDENCE_HINTS,
    HIGH_SPECIFICITY_EVIDENCE_HINTS,
    LOW_DEFENSE_JOB_HINTS,
    MAX_TURN_CUMULATIVE_PRESSURE_GAIN,
    MAX_TURN_PRESSURE_DELTA,
    NEW_CONTRADICTION_PRESSURE_DELTA,
    NEW_CONTRADICTION_PRESSURE_PROGRESS_BONUS,
    NEW_EVIDENCE_PRESSURE_DELTA,
    NEW_EVIDENCE_PRESSURE_PROGRESS_BONUS,
    PRESSURE_LEVEL_BONUS,
    PRESSURE_SIGMOID_MIDPOINT,
    PRESSURE_SIGMOID_STEEPNESS,
    REPEATED_CONTRADICTION_PRESSURE_DELTA,
    REPEATED_EVIDENCE_PRESSURE_DELTA,
    SOFT_DIALOGUE_PRESSURE_PROGRESS_BONUS,
    STRESS_IDLE_DECAY,
    STRESS_REPEAT_DECAY,
    STRESS_WEAK_TURN_DECAY,
    SUE_PRESSURE_BONUS_SCALE,
    _case_baseline_pad_state,
    _case_contradictions,
    _case_evidences,
    _case_personality,
    _default_mental_state,
    _empty_progress_state,
    _generate_final_psychological_reaction,
    _normalize_pad_state_blob,
    _statement_collapse_label,
)

class InterrogationCore:
    """심문 진행의 핵심 수학 모델을 묶은 클래스."""

    def __init__(
        self,
        pressure_steepness: float = PRESSURE_SIGMOID_STEEPNESS,
        pressure_midpoint: float = PRESSURE_SIGMOID_MIDPOINT,
    ):
        self.w = pressure_steepness
        self.midpoint = pressure_midpoint
        # SUE(Strategic Use of Evidence) 효과: 구체적이고 출처가 강한 증거일수록
        # 모순을 지적했을 때 심리적 압박이 크게 작동하도록 가중치를 둔다.
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
        """누적 압박을 Sigmoid에 넣기 전 원시 점수로 변환한다."""
        return self.w * (clamp01(cumulative_pressure) - self.midpoint)

    def calculate_breakdown_probability(
        self,
        cumulative_pressure: float,
    ) -> Tuple[float, float]:
        """누적 압박을 0~1 사이의 진술 붕괴 확률로 변환한다."""
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
        """붕괴 확률과 협조도를 바탕으로 NPC의 현재 행동 상태를 정한다."""
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
    """이번 턴에 언급된 증거와 모순의 SUE 압박 효과를 계산한다."""
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
    """플레이어 질문 태도와 모순 제시 결과에 따라 NPC 협조도를 갱신한다."""
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
    """이번 턴 압박이 NPC 스트레스에 얼마나 누적되는지 계산한다."""
    stress_delta = pressure_delta

    if not current_contradiction_ids and not current_evidence_ids:
        if pressure_level == "none":
            stress_delta = -STRESS_IDLE_DECAY
        elif pressure_level == "low":
            stress_delta = -STRESS_WEAK_TURN_DECAY

    if detect_repeat(history, user_text):
        stress_delta -= STRESS_REPEAT_DECAY

    return clamp01(prior_stress_score + stress_delta)

def _dialogue_contradiction_bonus_values(
    dialogue_signal: Optional[Dict[str, Any]],
) -> Tuple[float, float, bool]:
    """소프트 모순 신호를 압박 보너스와 붕괴 보너스로 변환한다."""
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

def evaluate_interrogation_progress_v3(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    interrogation_signal: Optional[Dict[str, Any]] = None,
    prior_progress: Optional[Dict[str, Any]] = None,
    contradiction_ids: Optional[List[str]] = None,
    dialogue_contradiction_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """한 턴의 질문/답변/모순 결과를 종합해 심문 진행 상태를 갱신한다."""
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

    if interrogation_signal is None:
        from app.services.openai_service import llm_evaluate_interrogation
        signal = llm_evaluate_interrogation(case_data, history, user_text)
    else:
        signal = interrogation_signal
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

    # 현재 턴에서 새로 제시된 증거/모순과 이미 누적된 증거/모순을 분리한다.
    # 새 모순은 반복 증거보다 더 큰 압박을 주도록 별도 가중치를 적용한다.
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
    # 압박은 증거, 하드 모순, 소프트 모순, 질문 강도로 나누어 계산한다.
    # 이후 debug 응답에서 각 항목을 보여 주면 통계/발표용 분석에 활용할 수 있다.
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
    # PAD는 실시간 감정 상태다. HEXACO는 고정 성향이고,
    # PAD는 이번 턴의 압박과 모순으로 계속 변한다.
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
    # 누적 압박은 Sigmoid를 통과해 붕괴 확률이 된다.
    # 초반엔 변화가 작고 임계점 이후 급격히 흔들리는 패턴을 표현한다.
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

def _calculate_personality_response_factors(
    case_data: Optional[Dict[str, Any]],
    player_intent: str,
) -> Dict[str, float]:
    """HEXACO 성향을 게임 계산에 쓰는 배율/보정값으로 변환한다."""
    personality = _case_personality(case_data)
    honesty_humility = personality["honesty_humility"]
    emotionality = personality["emotionality"]
    extraversion = personality["extraversion"]
    agreeableness = personality["agreeableness"]
    conscientiousness = personality["conscientiousness"]
    openness_to_experience = personality["openness_to_experience"]

    pressure_multiplier = (
        0.96
        + (0.30 * emotionality)
        - (0.16 * conscientiousness)
        - (0.08 * agreeableness)
    )
    if player_intent == "Rapport":
        pressure_multiplier -= 0.12 * max(honesty_humility, agreeableness)
    elif player_intent == "Intimidate":
        pressure_multiplier += 0.08 + (0.12 * emotionality) - (0.05 * extraversion)
    elif player_intent == "Confront":
        pressure_multiplier += 0.05 + (0.08 * honesty_humility)

    return {
        "pressure_multiplier": max(0.65, min(1.65, pressure_multiplier)),
        "stress_multiplier": max(
            0.6,
            min(
                1.9,
                0.82
                + (0.82 * emotionality)
                - (0.25 * conscientiousness)
                - (0.10 * extraversion),
            ),
        ),
        "direct_bonus_multiplier": max(
            0.72,
            min(
                1.65,
                0.88
                + (0.30 * honesty_humility)
                + (0.24 * emotionality)
                - (0.18 * conscientiousness),
            ),
        ),
        "cooperation_shift": max(
            -0.18,
            min(
                0.20,
                ((honesty_humility - 0.5) * 0.15)
                + ((agreeableness - 0.5) * 0.16)
                + ((extraversion - 0.5) * 0.05)
                - ((emotionality - 0.5) * 0.06),
            ),
        ),
        "arousal_sensitivity": max(
            0.72,
            min(
                1.95,
                0.86
                + (0.98 * emotionality)
                + (0.14 * openness_to_experience),
            ),
        ),
        "dominance_resistance": max(
            0.62,
            min(
                1.55,
                0.84
                + (0.38 * conscientiousness)
                + (0.32 * extraversion)
                - (0.26 * emotionality)
                - (0.10 * agreeableness),
            ),
        ),
        "collapse_resistance": max(
            0.58,
            min(
                1.75,
                0.88
                + (0.92 * conscientiousness)
                - (0.34 * emotionality)
                - (0.18 * honesty_humility),
            ),
        ),
        "rapport_affinity": max(
            0.72,
            min(
                1.5,
                0.82
                + (0.50 * honesty_humility)
                + (0.42 * agreeableness),
            ),
        ),
        "reply_length_bias": max(
            0.55,
            min(
                1.9,
                0.60
                + (0.92 * extraversion)
                + (0.18 * openness_to_experience),
            ),
        ),
        "rigidity_bias": max(
            0.62,
            min(
                1.6,
                0.96
                + (0.54 * conscientiousness)
                + (0.24 * (1.0 - openness_to_experience)),
            ),
        ),
        "friendliness_bias": max(
            0.58,
            min(
                1.6,
                0.72
                + (0.78 * agreeableness)
                + (0.34 * honesty_humility)
                - (0.16 * emotionality),
            ),
        ),
        "volatility_bias": max(
            0.58,
            min(
                1.7,
                0.76
                + (0.72 * emotionality)
                - (0.24 * conscientiousness)
                - (0.12 * agreeableness),
            ),
        ),
    }

def _build_personality_response_breakdown(
    case_data: Optional[Dict[str, Any]],
    player_intent: str,
) -> Dict[str, Any]:
    personality = _case_personality(case_data)
    factors = _calculate_personality_response_factors(case_data, player_intent)
    intent_shift = {
        "Rapport": "honesty-humility and agreeableness make rapport land better",
        "Probe": "neutral factual pressure",
        "Confront": "honesty-humility and openness make contradiction pressure bite differently",
        "Intimidate": "emotionality raises pressure and arousal sensitivity",
        "Neutral": "baseline interpretation",
    }.get(player_intent, "baseline interpretation")
    return {
        "traits": {key: round(value, 3) for key, value in personality.items()},
        "intent": player_intent,
        "intent_effect": intent_shift,
        "computed_factors": {key: round(value, 3) for key, value in factors.items()},
        "readout": {
            "pressure": "higher emotionality increases felt pressure; conscientiousness and agreeableness stabilize it",
            "cooperation": "honesty-humility and agreeableness soften cooperation loss; low honesty-humility increases self-protective resistance",
            "collapse": "conscientiousness delays collapse, emotionality accelerates wobble, honesty-humility makes concessions leak sooner",
            "pad": "emotionality pushes arousal up faster, extraversion and conscientiousness help preserve dominance, agreeableness lowers combative dominance",
            "speech": "extraversion changes response length, agreeableness changes warmth, honesty-humility changes sincerity, emotionality changes shakiness, conscientiousness changes consistency, openness changes reframing",
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
    """압박과 모순이 PAD 세 축(안정도, 긴장도, 주도권)에 미치는 변화를 계산한다."""
    personality = _case_personality(case_data)
    factors = _calculate_personality_response_factors(case_data, player_intent)
    honesty_humility = personality["honesty_humility"]
    emotionality = personality["emotionality"]
    extraversion = personality["extraversion"]
    agreeableness = personality["agreeableness"]
    conscientiousness = personality["conscientiousness"]

    pleasure = prior_pad_state.get("pleasure", 0.5)
    arousal = prior_pad_state.get("arousal", 0.5)
    dominance = prior_pad_state.get("dominance", 0.5)

    contradiction_impact = 0.05 * float(new_contradiction_count)
    evidence_impact = 0.02 * float(new_evidence_count)
    soft_impact = 0.02 if soft_dialogue_contradiction else 0.0

    pleasure_delta = -pressure_delta * (0.13 + (0.14 * emotionality))
    pleasure_delta -= contradiction_impact * (0.74 - (0.22 * agreeableness))
    pleasure_delta -= evidence_impact * 0.5
    if player_intent == "Rapport":
        pleasure_delta += 0.025 * factors["rapport_affinity"]
    elif player_intent == "Intimidate":
        pleasure_delta -= 0.02
    elif player_intent == "Confront":
        pleasure_delta -= 0.012 * honesty_humility
    if repeated_question:
        pleasure_delta -= 0.015

    arousal_delta = pressure_delta * (0.30 + (0.35 * factors["arousal_sensitivity"]))
    arousal_delta += contradiction_impact + evidence_impact + soft_impact
    if player_intent == "Rapport":
        arousal_delta -= 0.02
    elif player_intent == "Intimidate":
        arousal_delta += 0.015

    dominance_delta = -pressure_delta * (0.18 + (0.18 * emotionality) + (0.10 * agreeableness))
    dominance_delta -= (0.045 * float(new_contradiction_count))
    dominance_delta += 0.03 * (extraversion - 0.5)
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
    """누적 압박, 모순 수, PAD 상태, HEXACO를 합산해 0~5 붕괴 단계를 산출한다."""
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
        + (personality["emotionality"] * 0.32)
        + (personality["honesty_humility"] * 0.14)
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
    """소프트 모순이 잡혔을 때 최소 붕괴 단계를 보장한다."""
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
