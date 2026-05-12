"""사전 제작 사건 선택 API 라우터."""

from fastapi import APIRouter

from app.services import case_service

router = APIRouter()

# 최종 버전에서는 LLM 즉석 생성 대신 cases/prebuilt의 사건 3개를 반환한다.
router.post("/case/generate")(case_service.case_generate)
