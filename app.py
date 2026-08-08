"""Interior Design agent.

Upload photos of a room, describe the look you want, then keep chatting: each
follow-up ("make the walls sage", "add a floor lamp in the corner") edits the most
recent render. State is held in a per-session Gemini chat, so the model remembers
your room and every render it has produced.

Model: gemini-2.5-flash-image ("nano-banana"), which retains generated images in
chat history and can edit them across turns.

Auth uses Application Default Credentials via Vertex AI -- run
`gcloud auth application-default login` if needed. Set your Google Cloud project
via the GOOGLE_CLOUD_PROJECT environment variable (and GOOGLE_CLOUD_LOCATION if
you want a location other than "global").

Run:
    export GOOGLE_CLOUD_PROJECT=your-project-id
    python app.py
then open http://localhost:8000
"""

import asyncio
import base64
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types

INDEX_FILE = Path(__file__).with_name("index.html")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

if not PROJECT_ID:
    raise RuntimeError(
        "Set GOOGLE_CLOUD_PROJECT to your Google Cloud project id/number "
        "(and run `gcloud auth application-default login`)."
    )

# Image-capable models verified available for this project. Switching model
# starts a new session (a chat is bound to one model for its whole history).
MODELS = [
    {
        "id": "gemini-2.5-flash-image",
        "name": "Nano Banana (2.5 Flash)",
        "desc": "Fast & economical. Best for quick iterations and everyday restyles.",
    },
    {
        "id": "gemini-3.1-flash-image",
        "name": "Nano Banana (3.1 Flash)",
        "desc": "Balanced: near-Pro quality at low latency and price. Better detail and aspect handling than 2.5.",
    },
    {
        "id": "gemini-3-pro-image",
        "name": "Nano Banana Pro (3 Pro)",
        "desc": "Highest quality: sharpest detail, best prompt adherence and text. Slower and pricier.",
    },
]
VALID_MODELS = {m["id"] for m in MODELS}
DEFAULT_MODEL = MODELS[0]["id"]

SYSTEM_INSTRUCTION = (
    "You are an expert interior designer working iteratively with a client. "
    "When the client shares room photos, treat them as a real room: preserve the "
    "camera angle, window and door positions, and overall "
    "proportions while restyling. Always respond with photorealistic "
    "interior renders, and offer a few options when you see fit. Treat each new "
    "message as feedback to apply to your most recent render, unless the client "
    "clearly asks to start over. Alongside the image, give some short text "
    "describing what you changed."
)

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# session_id -> genai chat (holds full history incl. generated images)
_sessions: dict[str, object] = {}
# session_id -> model id the chat was created with
_session_models: dict[str, str] = {}

app = FastAPI(title="Interior Design Agent")


def _new_chat(model: str):
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_modalities=["TEXT", "IMAGE"],
        ),
    )


def _extract(response) -> tuple[str | None, str]:
    """Return (image_data_url, text) from a chat response."""
    image_url: str | None = None
    text_chunks: list[str] = []
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data and image_url is None:
                mime = inline.mime_type or "image/png"
                b64 = base64.b64encode(inline.data).decode()
                image_url = f"data:{mime};base64,{b64}"
            elif getattr(part, "text", None):
                text_chunks.append(part.text)
    return image_url, " ".join(text_chunks).strip()


def _send(chat, parts: list[types.Part]):
    return chat.send_message(parts)


@app.post("/message")
async def message(
    message: str = Form(""),
    session_id: str = Form(""),
    model: str = Form(""),
    files: list[UploadFile] | None = None,
):
    """Send a message to the design agent; creates a session if needed.

    A chat is bound to one model, so a change of model always starts a fresh
    session (the caller is told via the returned session_id).
    """
    model = model if model in VALID_MODELS else DEFAULT_MODEL

    image_parts: list[types.Part] = []
    for upload in files or []:
        data = await upload.read()
        if data:
            image_parts.append(
                types.Part.from_bytes(
                    data=data, mime_type=upload.content_type or "image/jpeg"
                )
            )

    if not message.strip() and not image_parts:
        raise HTTPException(status_code=400, detail="Type a message or attach a photo.")

    reuse = (
        session_id in _sessions and _session_models.get(session_id) == model
    )
    if reuse:
        chat = _sessions[session_id]
    else:
        _sessions.pop(session_id, None)
        _session_models.pop(session_id, None)
        session_id = uuid.uuid4().hex
        chat = _new_chat(model)
        _sessions[session_id] = chat
        _session_models[session_id] = model

    parts = image_parts + ([types.Part(text=message.strip())] if message.strip() else [])

    try:
        response = await asyncio.to_thread(_send, chat, parts)
    except Exception as exc:  # surface model/quota errors to the UI
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    image_url, text = _extract(response)
    if image_url is None and not text:
        raise HTTPException(status_code=502, detail="The model returned nothing.")

    return JSONResponse(
        {"session_id": session_id, "model": model, "image": image_url, "text": text}
    )


@app.post("/reset")
async def reset(session_id: str = Form("")):
    """Drop a session's history so the next message starts a fresh room."""
    _sessions.pop(session_id, None)
    _session_models.pop(session_id, None)
    return JSONResponse({"ok": True})


@app.get("/config")
async def config():
    """Expose the available models so the frontend can build its picker."""
    return JSONResponse({"models": MODELS, "default_model": DEFAULT_MODEL})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
