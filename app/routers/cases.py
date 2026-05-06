from fastapi import APIRouter

from app.services import case_service

router = APIRouter()

router.post("/case/generate")(case_service.case_generate)
