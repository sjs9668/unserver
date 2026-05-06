import base64
import json
from io import BytesIO
from typing import Any, Dict, List, Optional

from openai import OpenAI, BadRequestError

from app.config import LLM_MODEL, OPENAI_API_KEY, STT_FALLBACK, STT_PRIMARY, TTS_MODEL, TTS_VOICE
from app.utils.json import safe_json_loads
from app.utils.text import detect_repeat, is_too_ambiguous, norm, norm_for_match, trim_to_1_3_sentences
from app.services.contradiction_service import (
    _is_overly_evasive_answer,
    _split_short_sentences,
    llm_review_suspect_reply,
)
from app.services.scoring_service import _calculate_personality_response_factors, _infer_player_intent
from app.services.interrogation_service import (
    DEFAULT_FSM_STATE,
    QUESTION_ANALYSIS_SCHEMA,
    QUESTION_SLOT_GUIDE,
    SLOT_LABELS,
    TRUTH_SLOT_NAMES,
    _backfill_question_analysis,
    _case_evidences,
    _case_personality,
    _dialogue_lines,
    _empty_question_analysis,
    _fallback_question_analysis,
    _normalize_pad_state_blob,
    _sanitize_question_analysis,
    _statement_collapse_label,
)

client = OpenAI(api_key=OPENAI_API_KEY)

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

def _build_personality_speaking_directives(case_data: Optional[Dict[str, Any]]) -> List[str]:
    personality = _case_personality(case_data)
    honesty_humility = personality["honesty_humility"]
    emotionality = personality["emotionality"]
    extraversion = personality["extraversion"]
    agreeableness = personality["agreeableness"]
    conscientiousness = personality["conscientiousness"]
    openness_to_experience = personality["openness_to_experience"]

    directives: List[str] = [
        "- Make the personality difference obvious on the surface of the reply, not only in hidden state.",
    ]

    if honesty_humility >= 0.7:
        directives.append("- High honesty-humility: sound less manipulative and less entitled. Let moral discomfort, shame, or reluctant candor leak through under pressure.")
    elif honesty_humility <= 0.3:
        directives.append("- Low honesty-humility: sound more self-serving, calculating, and blame-shifting. Protect yourself first and minimize fault.")

    if emotionality >= 0.7:
        directives.append("- High emotionality: sound nervous, tense, and easily unsettled. Let fear, strain, and protective hedging show clearly.")
    elif emotionality <= 0.3:
        directives.append("- Low emotionality: sound unusually cold, flat, and unsentimental. Avoid nervous hedging and do not sound easily rattled.")

    if conscientiousness >= 0.7:
        directives.append("- High conscientiousness: sound careful, controlled, and precise. Keep the wording organized and internally consistent.")
    elif conscientiousness <= 0.3:
        directives.append("- Low conscientiousness: allow looser wording, slight messiness, and small mid-sentence self-corrections.")

    if extraversion >= 0.75:
        directives.append("- High extraversion: be visibly more talkative. Usually give 2 or 3 spoken sentences, and add one short follow-up explanation after the direct answer.")
    elif extraversion <= 0.3:
        directives.append("- Low extraversion: be visibly clipped. Usually stop after 1 short sentence and avoid voluntary follow-up explanation.")

    if agreeableness >= 0.7:
        directives.append("- High agreeableness: sound patient, soft, and non-combative. Even when resisting, avoid snapping and keep the tone civil.")
    elif agreeableness <= 0.3:
        directives.append("- Low agreeableness: sound sharp, argumentative, and easily irritated. Challenge the detective more openly and avoid conciliatory wording.")

    if openness_to_experience >= 0.7:
        directives.append("- High openness to experience: reach for alternative framings, unusual angles, or reinterpretations of what happened.")
    elif openness_to_experience <= 0.3:
        directives.append("- Low openness to experience: stay literal, narrow, and repetitive. Reuse the same frame rather than inventing new angles.")

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
            f"[Selected HEXACO] honesty_humility:{personality['honesty_humility']:.2f} "
            f"emotionality:{personality['emotionality']:.2f} "
            f"extraversion:{personality['extraversion']:.2f} "
            f"agreeableness:{personality['agreeableness']:.2f} "
            f"conscientiousness:{personality['conscientiousness']:.2f} "
            f"openness_to_experience:{personality['openness_to_experience']:.2f}\n"
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
