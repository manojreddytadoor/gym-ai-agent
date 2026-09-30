"""Web server: serves the UI and streams each coach's progress as NDJSON."""

import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .coaches import Coach, DietCoach, WorkoutCoach
from .llm import OllamaLLM
from .profile import UserProfile

log = logging.getLogger(__name__)

MAX_SESSIONS = 200  # sessions live in memory; the oldest are dropped first

app = FastAPI(title="Gym AI Coaches")
llm = OllamaLLM()


@dataclass
class Session:
    profile: UserProfile
    coaches: dict[str, Coach]


sessions: OrderedDict[str, Session] = OrderedDict()


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, max_length=2000)


@app.get("/api/model")
def model() -> dict:
    return llm.model_info()


@app.post("/api/sessions")
def create_session(profile: UserProfile) -> dict:
    session_id = uuid.uuid4().hex
    sessions[session_id] = Session(
        profile=profile,
        coaches={"diet": DietCoach(llm, profile), "workout": WorkoutCoach(llm, profile)},
    )
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)
    return {"session_id": session_id}


@app.post("/api/sessions/{session_id}/{coach_name}/chat")
def chat(session_id: str, coach_name: Literal["diet", "workout"], request: ChatRequest) -> StreamingResponse:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Generate your plans again.")
    coach = session.coaches[coach_name]
    if not coach.lock.acquire(blocking=False):
        raise HTTPException(409, "This coach is still replying.")

    def stream():
        try:
            for event in coach.chat(request.message):
                yield json.dumps(event) + "\n"
        except Exception:
            log.exception("Unexpected error in %s coach", coach_name)
            yield json.dumps({"type": "error", "message": "Something went wrong. Check the server log."}) + "\n"
        finally:
            coach.lock.release()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
