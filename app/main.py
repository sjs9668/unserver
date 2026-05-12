"""FastAPI 서버 진입점.

Unreal 클라이언트가 호출하는 사건 선택, 심문 진행, 상태 확인 API를
하나의 FastAPI 앱에 등록한다.
"""

from fastapi import FastAPI

from app.routers import cases, health, interrogation


def create_app() -> FastAPI:
    """라우터를 등록해 서버 앱 인스턴스를 생성한다."""
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(cases.router)
    app.include_router(interrogation.router)
    return app


app = create_app()


if __name__ == "__main__":
    import os

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False, log_level="info")
