# Interior Design Agent

A small web app that redesigns your room. Upload photos, describe the look you
want, then keep chatting — each follow-up ("make the walls sage green", "add a
floor lamp in the corner") edits the most recent render. Conversation state
(including every image the model has produced) is held in a per-session Gemini
chat, so the model remembers your room across turns.

It uses Google's Gemini image models ("Nano Banana") on Vertex AI, which can take
your photos plus a text brief and return photorealistic interior renders — and
edit those renders conversationally.

## Features

- **Iterative chat** — restyle by conversation, not one-shot prompts.
- **Model picker** — switch between the available image models:
  - `gemini-2.5-flash-image` — fast & economical, great for quick iterations.
  - `gemini-3.1-flash-image` — balanced: near-Pro quality at low latency/price.
  - `gemini-3-pro-image` — highest quality, best prompt adherence. Slower/pricier.

  Switching model starts a new room (a chat is bound to one model).
- **Multi-photo uploads**, drag-and-drop, and mid-conversation attachments.
- **Text replies** — the model describes what it changed alongside each render.
- Click any render to open it full-size in a new window.

## Requirements

- Python 3.10+
- A Google Cloud project with Vertex AI enabled and access to the Gemini image models.
- Application Default Credentials.

## Setup

```bash
pip install -r requirements.txt
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id     # required
export GOOGLE_CLOUD_LOCATION=global             # optional, defaults to "global"
python app.py
```

Then open http://localhost:8000.

## How it works

- FastAPI backend (`app.py`) serves a single-page UI and two endpoints:
  - `POST /message` — sends your text + any uploaded photos to the model and
    returns the render plus a short description.
  - `POST /reset` — clears a session so the next message starts a fresh room.
- Photos are sent as inline image bytes (not filenames) in the same user turn as
  your text.
- Sessions are held in memory, so restarting the server clears history.

## Notes

- Generation typically takes ~10–30s per turn; the UI shows an elapsed-time
  indicator while it works.
- Model availability depends on your Google Cloud project and region.
