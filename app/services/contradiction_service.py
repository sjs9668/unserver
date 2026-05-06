import re
from typing import Any, Dict, List, Optional, Tuple

from app.utils.text import norm, norm_for_match, uniq_strings
from app.services.interrogation_service import (
    _case_contradictions,
    _case_truth_slots,
    _empty_dialogue_contradiction_signal,
)

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
    if question_analysis is None:
        from app.services.openai_service import llm_evaluate_interrogation
        analysis = llm_evaluate_interrogation(case_data, history, user_text)
    else:
        analysis = question_analysis
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
