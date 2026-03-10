import os
import json
import base64
import traceback
import re
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

_openai_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    global _openai_client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    if _openai_client is None:
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# =========================================================
# MODEL CONFIG
# =========================================================
STT_PRIMARY = "gpt-4o-mini-transcribe"
STT_FALLBACK = "whisper-1"
LLM_MODEL = "gpt-5.2"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "verse"

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True, "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip())}

# =========================================================
# IN-MEMORY STORE (서버 재시작하면 초기화)
# =========================================================
CASE_CACHE: Dict[str, Dict[str, Any]] = {}

# =========================================================
# EXAMPLE CASE (템플릿 예시로만 사용)
# =========================================================
EXAMPLE_CASE_PATH = Path(__file__).parent / "cases" / "case_interrogation_01.json"
EXAMPLE_CASE_TEXT = ""

if EXAMPLE_CASE_PATH.exists():
    EXAMPLE_CASE_TEXT = EXAMPLE_CASE_PATH.read_text(encoding="utf-8").strip()
else:
    EXAMPLE_CASE_TEXT = ""

# =========================================================
# UTILS
# =========================================================
def safe_json_loads(s: str, default: Any):
    try:
        return json.loads(s)
    except Exception:
        return default

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_for_match(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())

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

def extract_case_keywords(case_data: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    압박도 계산용 키워드:
    - evidences: id/name
    - contradictions: id/description
    """
    if not case_data:
        return [], []

    ev_keywords: List[str] = []
    cd_keywords: List[str] = []

    evidences = case_data.get("evidences", []) or []
    for e in evidences:
        if isinstance(e, dict):
            if e.get("id"):
                ev_keywords.append(str(e["id"]))
            if e.get("name"):
                ev_keywords.append(str(e["name"]))

    contradictions = case_data.get("contradictions", []) or []
    for c in contradictions:
        if isinstance(c, dict):
            if c.get("id"):
                cd_keywords.append(str(c["id"]))
            if c.get("description"):
                cd_keywords.append(str(c["description"]))

    def uniq(xs: List[str]) -> List[str]:
        out = []
        seen = set()
        for x in xs:
            x = (x or "").strip()
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return uniq(ev_keywords), uniq(cd_keywords)

def build_case_context(case_data: Optional[Dict[str, Any]]) -> str:
    """
    LLM에게 줄 컨텍스트(심문 게임용 내부 사건 정보)
    """
    if not case_data:
        return "사건 정보 없음.\n"

    ov = case_data.get("overview", {}) or {}
    suspect = case_data.get("suspect", {}) or {}

    evidences = case_data.get("evidences", []) or []
    contradictions = case_data.get("contradictions", []) or []

    ev_lines = []
    for e in evidences:
        if isinstance(e, dict):
            ev_lines.append(f"- {e.get('name','')} : {e.get('description','')}".strip())
        else:
            ev_lines.append(f"- {str(e)}")

    cd_lines = []
    for c in contradictions:
        if isinstance(c, dict):
            rel = c.get("related_evidence", [])
            rel_s = f" (related:{rel})" if rel else ""
            cd_lines.append(f"- {c.get('description','')}{rel_s}".strip())
        else:
            cd_lines.append(f"- {str(c)}")

    return (
        "=== 사건 정보(내부) ===\n"
        f"[개요] 시간:{ov.get('time','')} 장소:{ov.get('place','')} 유형:{ov.get('type','')}\n"
        f"[동기] {case_data.get('motive','')}\n"
        f"[범행 흐름(사실)] {case_data.get('crime_flow','')}\n"
        f"[용의자] 이름:{suspect.get('name','')} 나이:{suspect.get('age','')} 직업:{suspect.get('job','')} 관계:{suspect.get('relation','')}\n"
        f"[기본 거짓 진술] {case_data.get('false_statement','')}\n"
        "[증거 목록]\n" + ("\n".join(ev_lines) if ev_lines else "- 없음") + "\n"
        "[모순점 목록]\n" + ("\n".join(cd_lines) if cd_lines else "- 없음") + "\n"
        "======================\n"
    )

def calc_pressure_and_prob(
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str
) -> Tuple[float, float]:
    """
    2주차 초기 버전 + 3주차 안정화(반복 질문 반감)
    - 증거 언급: +0.20 * hits
    - 모순 찌름: +0.35 * hits
    - 기본 확률: 0.12 + min(0.30, 0.03 * 질문수)
    - 최종 확률 = base + pressure, 0~1 클램프
    """
    ev_kws, cd_kws = extract_case_keywords(case_data)

    all_text = user_text + " " + " ".join(
        [(h.get("user_text", "") or "") for h in history if isinstance(h, dict)]
    )
    all_text_match = norm_for_match(all_text)

    ev_hits = 0
    cd_hits = 0

    for kw in ev_kws:
        kw_match = norm_for_match(kw)
        if kw_match and kw_match in all_text_match:
            ev_hits += 1
    for kw in cd_kws:
        kw_match = norm_for_match(kw)
        if kw_match and kw_match in all_text_match:
            cd_hits += 1

    pressure = 0.0
    pressure += 0.20 * ev_hits
    pressure += 0.35 * cd_hits
    pressure = clamp01(pressure)

    base = 0.12 + min(0.30, 0.03 * len(history))
    prob = clamp01(base + pressure)

    # 반복 질문은 압박 반감(치팅 방지)
    if detect_repeat(history, user_text):
        pressure = clamp01(pressure * 0.5)
        prob = clamp01(base + pressure)

    return pressure, prob

# =========================================================
# STT
# =========================================================
def stt_transcribe(audio_bytes: bytes) -> str:
    """
    1) gpt-4o-mini-transcribe 시도
    2) 포맷/호환 에러면 whisper-1로 폴백
    """
    def _call(model_name: str) -> str:
        client = get_openai_client()
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

    client = get_openai_client()
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
            "minItems": 2,
            "maxItems": 6,
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
                },
                # ✅ 여기만 수정: related_evidence를 required에 포함
                "required": ["id", "description", "related_evidence"],
            },
            "minItems": 2,
            "maxItems": 5,
        },
    },
    "required": [
        "case_id",
        "overview",
        "motive",
        "crime_flow",
        "suspect",
        "false_statement",
        "evidences",
        "contradictions",
    ],
}

def llm_generate_case() -> Dict[str, Any]:
    """
    seed 없이, case_interrogation_01.json 예시를 보고 자동 생성.
    - 구조/톤/필드 사용방식은 예시를 따르되
    - 내용(사건/인물/증거/모순)은 반드시 새로 만들기
    """
    system = (
        "너는 '음성 기반 심문 게임'의 사건 생성기다.\n"
        "현실적인 한국 수사/심문 톤의 사건을 만든다.\n"
        "규칙:\n"
        "- 출력은 반드시 JSON이며, 주어진 스키마를 엄격히 준수한다.\n"
        "- 아래 제공되는 '예시 사건 JSON'은 형식/톤/밀도(정보량)의 기준이다.\n"
        "- 예시와 '구조'는 비슷하게 유지하되, '내용'은 완전히 다른 새 사건으로 만들 것.\n"
        "- evidences는 2~6개, contradictions는 2~5개.\n"
        "- contradictions.related_evidence에는 연결되는 evidence id를 넣어라(가능하면 1개 이상).\n"
        "- false_statement는 초반엔 그럴듯하지만, 증거/모순 누적으로 무너지게 설계.\n"
        "- crime_flow는 '사실(정답)' 요약이며, 플레이어에게 직접 노출되지 않는 내부 참고용.\n"
        "- case_id는 'case_'로 시작하는 문자열.\n"
    )

    example_block = (
        f"\n[예시 사건 JSON]\n{EXAMPLE_CASE_TEXT}\n"
        if EXAMPLE_CASE_TEXT else
        "\n[예시 사건 JSON]\n(예시 파일이 없어도 생성은 해야 한다.)\n"
    )

    user = (
        "예시를 참고해서, 완전히 새로운 사건 1개를 생성하라.\n"
        "중요: 예시의 문장/표현/사건 디테일을 그대로 복사하지 말고, 다른 사건으로 만들 것.\n"
        "사건 유형/장소/증거/모순 조합이 뻔하지 않게 다양하게 만들 것.\n"
        + example_block +
        "\n이제 새 사건 JSON을 출력하라.\n"
    )

    client = get_openai_client()
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
        #max_output_tokens=1100,
        max_output_tokens=4000,
        store=False,
        temperature=1.0,
    )

    text = (resp.output_text or "").strip()
    return json.loads(text)


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

def llm_suspect_answer(
    case_context: str,
    case_data: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    user_text: str,
    confession_probability: float
) -> str:
    """
    용의자 답변 생성(2~3주차)
    - 형사가 꺼내지 않은 증거를 먼저 말하지 않게 가드
    - 1~3문장
    """
    system = (
        "너는 심문실에 앉아있는 '용의자'다.\n"
        "유저는 담당 형사다.\n"
        "규칙:\n"
        "- 한국어\n"
        "- 1~3문장\n"
        "- 너무 소설처럼 장황하게 말하지 말 것\n"
        "- 형사가 언급하지 않은 증거(예: CCTV, 목격자, 물증)를 네가 먼저 꺼내지 마라.\n"
        "- 증거/모순 압박이 약하면 기본 거짓 진술을 유지하며 버틴다.\n"
        "- 압박이 강하면 변명/흔들림/부분 인정(핵심 범행 자백은 금지)을 한다.\n"
        "- 자백 트리거가 아니면 범행을 완전 자백하지 말 것.\n"
    )

    recent = history[-4:] if isinstance(history, list) else []
    hist_lines = []
    detective_mentions = []
    for h in recent:
        if not isinstance(h, dict):
            continue
        u = (h.get("user_text", "") or "").strip()
        s = (h.get("suspect_text", "") or "").strip()
        if u:
            hist_lines.append(f"형사: {u}")
            detective_mentions.append(u)
        if s:
            hist_lines.append(f"용의자: {s}")

    detective_mentions.append(user_text)

    allowed_evidence_words = _collect_allowed_evidence_words(case_data, detective_mentions)
    all_evidence_words = _all_evidence_words(case_data)

    extra_guard = ""
    if is_too_ambiguous(user_text):
        extra_guard += "\n- 질문이 모호하면 무슨 뜻인지 되묻고 질문을 구체화하게 유도해라."
    if detect_repeat(history, user_text):
        extra_guard += "\n- 같은 질문을 반복하면 이미 답했다고 짧게 말해라."

    user = (
        case_context +
        f"\n[현재 자백 확률(참고용)] {confession_probability:.2f}\n"
        "[최근 대화]\n" + ("\n".join(hist_lines) if hist_lines else "(없음)") + "\n"
        f"\n형사의 질문: {user_text}\n"
        f"\n[형사가 이미 꺼낸 증거 키워드(이것만 언급 가능)] {allowed_evidence_words}\n"
        f"\n[추가 가드]\n{extra_guard}\n"
        "용의자의 답변만 출력하라."
    )

    client = get_openai_client()
    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_output_tokens=220,
    )
    out = trim_to_1_3_sentences((resp.output_text or "").strip())

    # 금지 증거 스포가 나오면 1회 재시도
    if _contains_banned_evidence(out, all_evidence_words, allowed_evidence_words):
        retry_user = (
            user +
            "\n\n[경고] 방금 답변에 '형사가 언급하지 않은 증거'가 포함됐다. "
            "그런 단어는 절대 말하지 말고, 일반적인 부인/변명으로만 다시 답해라."
        )
        resp2 = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": retry_user},
            ],
            max_output_tokens=200,
        )
        out = trim_to_1_3_sentences((resp2.output_text or "").strip())

    return out or "…모릅니다. 정말 집에 있었습니다."

def llm_confession(case_context: str, history: List[Dict[str, Any]], user_text: str) -> str:
    client = get_openai_client()
    system = (
        "너는 심문실에 앉아있는 '용의자'다.\n"
        "유저는 담당 형사다.\n"
        "지금은 '자백하는 순간'이다.\n"
        "규칙:\n"
        "- 한국어\n"
        "- 1~3문장\n"
        "- 변명보다 '인정' 중심\n"
        "- 감정(체념/후회/두려움) 중 하나를 살짝 포함\n"
    )

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
            {"role": "system", "content": system},
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
        if not cid or cid in CASE_CACHE:
            cid = f"case_{uuid.uuid4().hex[:8]}"
            case_data["case_id"] = cid

        CASE_CACHE[cid] = case_data
        return JSONResponse(status_code=200, content={"case": case_data})

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb[-2000:]})

@app.post("/interrogation/qna")
async def interrogation_qna(
    file: Optional[UploadFile] = File(None),
    user_text: str = Form(""),
    case_id: str = Form(""),
    history_json: str = Form("[]"),
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

        if not final_user_text:
            msg = "…잘 안 들립니다. 다시 말씀해 주세요."
            return JSONResponse(
                status_code=200,
                content={
                    "user_text": "",
                    "suspect_text": msg,
                    "pressure_delta": 0.0,
                    "confession_probability": 0.12,
                    "confession_triggered": False,
                    "audio_wav_b64": await tts_to_b64(msg),
                },
            )

        if is_too_ambiguous(final_user_text):
            msg = "무슨 뜻인지 잘 모르겠습니다. 시간이나 장소를 구체적으로 말해 보세요."
            base = 0.12 + min(0.30, 0.03 * len(history))
            return JSONResponse(
                status_code=200,
                content={
                    "user_text": final_user_text,
                    "suspect_text": msg,
                    "pressure_delta": 0.0,
                    "confession_probability": float(clamp01(base)),
                    "confession_triggered": False,
                    "audio_wav_b64": await tts_to_b64(msg),
                },
            )

        # 2) case load (cache 우선)
        case_data = None
        if case_id:
            case_data = CASE_CACHE.get(case_id)

        case_context = build_case_context(case_data)

        # 3) calc pressure/prob
        pressure_delta, confession_probability = calc_pressure_and_prob(case_data, history, final_user_text)
        confession_triggered = confession_probability >= 0.85

        # 4) LLM answer
        if confession_triggered:
            suspect_text = llm_confession(case_context, history, final_user_text)
        else:
            suspect_text = llm_suspect_answer(case_context, case_data, history, final_user_text, confession_probability)

        suspect_text = trim_to_1_3_sentences(suspect_text)

        # 5) TTS
        wav_b64 = await tts_to_b64(suspect_text)

        return JSONResponse(
            status_code=200,
            content={
                "user_text": final_user_text,
                "suspect_text": suspect_text,
                "pressure_delta": float(pressure_delta),
                "confession_probability": float(confession_probability),
                "confession_triggered": bool(confession_triggered),
                "audio_wav_b64": wav_b64,
            },
        )

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
