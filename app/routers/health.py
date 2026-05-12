"""서버 상태 확인 라우터."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """배포 플랫폼과 클라이언트가 서버 생존 여부를 확인하는 엔드포인트."""
    return {"ok": True}
