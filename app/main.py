from fastapi import FastAPI

from app.routers import cases, health, interrogation


def create_app() -> FastAPI:
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
