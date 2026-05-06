from fastapi import APIRouter

from app.services import interrogation_service

router = APIRouter()

router.post("/interrogation/setup")(interrogation_service.interrogation_setup)
router.post("/interrogation/judgment/submit")(interrogation_service.interrogation_judgment_submit)
router.post("/interrogation/report")(interrogation_service.interrogation_report)
router.post("/interrogation/result/reveal")(interrogation_service.interrogation_result_reveal)
router.post("/interrogation/debug_confess")(interrogation_service.interrogation_debug_confess)
router.post("/interrogation/qna")(interrogation_service.interrogation_qna)
