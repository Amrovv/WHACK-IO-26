"""D: FastAPI + WebSocket server. See coach/server/app.py."""


def __getattr__(name: str):
    # Lazy, so `python -m coach.server.app` doesn't import app.py twice.
    if name in ("create_app", "serve"):
        from coach.server import app

        return getattr(app, name)
    raise AttributeError(name)
