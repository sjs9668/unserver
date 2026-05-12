"""심문 진행 관련 API 라우터.

실제 처리 로직은 services/interrogation_service.py에 두고,
이 파일은 URL과 핸들러를 연결하는 얇은 계층만 담당한다.
"""

from fastapi import APIRouter

from app.services import interrogation_service

router = APIRouter()

router.post("/interrogation/setup")(interrogation_service.interrogation_setup)
router.post("/interrogation/judgment/submit")(interrogation_service.interrogation_judgment_submit)
router.post("/interrogation/report")(interrogation_service.interrogation_report)
router.post("/interrogation/result/reveal")(interrogation_service.interrogation_result_reveal)
router.post("/interrogation/debug_confess")(interrogation_service.interrogation_debug_confess)
router.post("/interrogation/qna")(interrogation_service.interrogation_qna)
