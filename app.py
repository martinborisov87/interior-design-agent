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
import json
import os
import uuid

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types

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


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML.replace("__MODELS_JSON__", json.dumps(MODELS)).replace(
        "__DEFAULT_MODEL__", DEFAULT_MODEL
    )


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Interior Design Agent</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 820px; margin: 0 auto; padding: 16px 16px 140px;
  }
  header { display: flex; align-items: baseline; justify-content: space-between; }
  h1 { font-size: 1.4rem; margin: 8px 0; }
  #reset { background: none; border: 1px solid #bbb; color: inherit; border-radius: 8px; padding: 6px 12px; cursor: pointer; font: inherit; }
  .modelbar { margin: 4px 0 2px; display: flex; align-items: center; gap: 8px; }
  .modelbar label { font-weight: 600; font-size: .9rem; }
  #model { padding: 6px 10px; border-radius: 8px; border: 1px solid #ccc; font: inherit; background: transparent; color: inherit; }
  #modelDesc { color: #888; font-size: .85rem; margin: 0 0 4px; }
  #chat { display: flex; flex-direction: column; gap: 14px; }
  .msg { display: flex; }
  .msg.user { justify-content: flex-end; }
  .bubble { max-width: 80%; padding: 10px 14px; border-radius: 14px; line-height: 1.45; }
  .user .bubble { background: #4a90d9; color: #fff; border-bottom-right-radius: 4px; }
  .agent .bubble { background: rgba(127,127,127,.15); border-bottom-left-radius: 4px; }
  .bubble img { display: block; width: 100%; max-width: 460px; border-radius: 10px; margin-top: 6px; }
  .bubble .attached { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
  .bubble .attached img { height: 70px; width: auto; margin: 0; }
  .bubble .caption { margin: 0 0 2px; }
  .bubble .caption:has(+ a) { margin-bottom: 8px; }
  .hint { color: #888; font-size: .9rem; }
  form {
    position: fixed; bottom: 0; left: 0; right: 0; background: var(--bg, rgba(255,255,255,.92));
    backdrop-filter: blur(8px); border-top: 1px solid rgba(127,127,127,.25);
    padding: 12px 16px;
  }
  @media (prefers-color-scheme: dark) { form { --bg: rgba(20,20,22,.92); } }
  .composer {
    max-width: 820px; margin: 0 auto; display: flex; flex-direction: column; gap: 8px;
    border: 1px solid rgba(127,127,127,.4); border-radius: 16px; padding: 10px 12px;
    background: rgba(127,127,127,.06); transition: border-color .15s;
  }
  .composer:focus-within { border-color: #4a90d9; }
  #text {
    width: 100%; border: 0; outline: none; background: transparent; color: inherit;
    font: inherit; font-size: 1.05rem; line-height: 1.5; resize: none;
    min-height: 66px; max-height: 260px; padding: 4px 4px;
  }
  .composer-actions { display: flex; justify-content: space-between; align-items: center; }
  .iconbtn { border: 1px solid rgba(127,127,127,.5); background: transparent; color: inherit; border-radius: 20px; padding: 8px 14px; cursor: pointer; font-size: .95rem; }
  .iconbtn:hover { border-color: #4a90d9; color: #4a90d9; }
  #send { background: #4a90d9; color: #fff; border: 0; border-radius: 20px; padding: 10px 26px; font-size: 1rem; font-weight: 600; cursor: pointer; }
  #send:disabled { opacity: .5; cursor: default; }
  #pending { display: flex; gap: 6px; flex-wrap: wrap; }
  #pending img { height: 52px; border-radius: 8px; cursor: pointer; }
  .error { color: #d9534f; }
  .loading { display: flex; align-items: center; gap: 10px; color: #888; }
  .spinner { width: 16px; height: 16px; flex: none; border: 2px solid rgba(127,127,127,.35); border-top-color: #4a90d9; border-radius: 50%; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .skeleton {
    margin-top: 8px; width: 100%; max-width: 460px; aspect-ratio: 4 / 3; border-radius: 10px;
    background: linear-gradient(100deg, rgba(127,127,127,.10) 30%, rgba(127,127,127,.24) 50%, rgba(127,127,127,.10) 70%);
    background-size: 200% 100%; animation: shimmer 1.3s linear infinite;
  }
  @keyframes shimmer { to { background-position: -200% 0; } }
</style>
</head>
<body>
  <header>
    <h1>Interior Design Agent</h1>
    <button id="reset" type="button">New room</button>
  </header>
  <div class="modelbar">
    <label for="model">Model</label>
    <select id="model"></select>
  </div>
  <p id="modelDesc"></p>
  <p class="hint">Attach photos of your room and describe the look you want. Then keep chatting — “make the walls sage green”, “add a large plant by the window” — and it edits the last render. Switching model starts a new room.</p>

  <div id="chat"></div>

  <form id="form">
    <div class="composer">
      <div id="pending"></div>
      <textarea id="text" rows="3" placeholder="Describe the room you want, or give feedback — e.g. “make the walls sage green and add a large plant by the window”. Press Enter to send, Shift+Enter for a new line."></textarea>
      <div class="composer-actions">
        <button type="button" class="iconbtn" id="attach" title="Attach photos">📎 Add photos</button>
        <button type="submit" id="send">Send</button>
      </div>
    </div>
    <input type="file" id="files" accept="image/*" multiple hidden />
  </form>

<script>
  const chat = document.getElementById('chat');
  const form = document.getElementById('form');
  const text = document.getElementById('text');
  const filesInput = document.getElementById('files');
  const attach = document.getElementById('attach');
  const pending = document.getElementById('pending');
  const send = document.getElementById('send');
  const resetBtn = document.getElementById('reset');
  const modelSel = document.getElementById('model');
  const modelDesc = document.getElementById('modelDesc');
  const MODELS = __MODELS_JSON__;
  let sessionId = '';
  let pendingFiles = [];

  MODELS.forEach(m => {
    const o = document.createElement('option');
    o.value = m.id; o.textContent = m.name; modelSel.appendChild(o);
  });
  modelSel.value = '__DEFAULT_MODEL__';
  function showDesc() {
    const m = MODELS.find(x => x.id === modelSel.value);
    modelDesc.textContent = m ? m.desc : '';
  }
  showDesc();
  modelSel.addEventListener('change', () => {
    showDesc();
    // A chat is bound to one model, so switching starts a fresh room.
    if (sessionId || chat.children.length) {
      if (sessionId) { const fd = new FormData(); fd.append('session_id', sessionId); fetch('/reset', { method: 'POST', body: fd }); }
      sessionId = ''; chat.innerHTML = ''; pendingFiles = []; renderPending();
      addAgent(null, 'Switched to ' + (MODELS.find(x => x.id === modelSel.value)?.name || modelSel.value) + ' — starting a new room.', false);
    }
  });

  attach.addEventListener('click', () => filesInput.click());
  filesInput.addEventListener('change', () => {
    for (const f of filesInput.files) if (f.type.startsWith('image/')) pendingFiles.push(f);
    renderPending();
    filesInput.value = '';
  });
  function renderPending() {
    pending.innerHTML = '';
    pendingFiles.forEach((f, i) => {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(f);
      img.title = 'Click to remove';
      img.onclick = () => { pendingFiles.splice(i, 1); renderPending(); };
      pending.appendChild(img);
    });
  }

  function autogrow() {
    text.style.height = 'auto';
    text.style.height = Math.min(text.scrollHeight, 260) + 'px';
  }
  text.addEventListener('input', autogrow);
  text.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });

  function addUser(msg, files) {
    const wrap = document.createElement('div');
    wrap.className = 'msg user';
    const b = document.createElement('div');
    b.className = 'bubble';
    if (msg) b.appendChild(document.createTextNode(msg));
    if (files.length) {
      const att = document.createElement('div');
      att.className = 'attached';
      files.forEach(f => { const im = document.createElement('img'); im.src = URL.createObjectURL(f); att.appendChild(im); });
      b.appendChild(att);
    }
    wrap.appendChild(b); chat.appendChild(wrap);
    scroll();
  }
  function addAgent(imgSrc, msg, isError) {
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    const b = document.createElement('div');
    b.className = 'bubble';
    if (isError) b.innerHTML = '<span class="error">' + msg + '</span>';
    else {
      if (msg) { const cap = document.createElement('div'); cap.className = 'caption'; cap.textContent = msg; b.appendChild(cap); }
      if (imgSrc) renderImage(b, imgSrc);
    }
    wrap.appendChild(b); chat.appendChild(wrap);
    scroll();
    return b;
  }
  function scroll() { window.scrollTo(0, document.body.scrollHeight); }

  // Open a rendered image in a new window. Browsers block opening data: URLs
  // directly, so convert to a blob URL first.
  function openImage(dataUrl) {
    const [head, b64] = dataUrl.split(',');
    const mime = (head.match(/data:(.*?);base64/) || [])[1] || 'image/png';
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([arr], { type: mime }));
    window.open(url, '_blank');
  }
  // Animated placeholder shown while the model works (generation takes ~10-30s
  // and returns nothing until the image is ready, so we show elapsed time).
  function startLoading() {
    const wrap = document.createElement('div'); wrap.className = 'msg agent';
    const b = document.createElement('div'); b.className = 'bubble';
    b.innerHTML =
      '<div class="loading"><span class="spinner"></span>' +
      '<span>Designing… <span class="secs">0s</span></span></div>' +
      '<div class="skeleton"></div>';
    wrap.appendChild(b); chat.appendChild(wrap); scroll();
    const t0 = Date.now();
    const secs = b.querySelector('.secs');
    const iv = setInterval(() => { secs.textContent = Math.round((Date.now() - t0) / 1000) + 's'; }, 250);
    return { bubble: b, stop: () => clearInterval(iv) };
  }
  function renderImage(container, src) {
    const im = document.createElement('img');
    im.src = src; im.onload = scroll; im.style.cursor = 'zoom-in';
    im.title = 'Open in new window';
    im.onclick = () => openImage(src);
    container.appendChild(im);
  }

  // Keep the last message clear of the fixed composer, which grows as you
  // type or attach photos.
  function syncBottomPadding() {
    document.body.style.paddingBottom = (form.offsetHeight + 24) + 'px';
  }
  new ResizeObserver(syncBottomPadding).observe(form);
  window.addEventListener('resize', syncBottomPadding);
  syncBottomPadding();

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const msg = text.value.trim();
    if (!msg && !pendingFiles.length) return;
    const files = pendingFiles.slice();

    addUser(msg, files);
    text.value = ''; text.style.height = 'auto';
    pendingFiles = []; renderPending();
    send.disabled = true;
    const load = startLoading();
    const thinking = load.bubble;

    const fd = new FormData();
    fd.append('message', msg);
    fd.append('session_id', sessionId);
    fd.append('model', modelSel.value);
    files.forEach(f => fd.append('files', f));

    try {
      const res = await fetch('/message', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      sessionId = data.session_id;
      thinking.innerHTML = '';
      if (data.text) { const cap = document.createElement('div'); cap.className = 'caption'; cap.textContent = data.text; thinking.appendChild(cap); }
      else if (!data.image) { const cap = document.createElement('div'); cap.className = 'caption'; cap.textContent = '(no description returned)'; thinking.appendChild(cap); }
      if (data.image) renderImage(thinking, data.image);
    } catch (err) {
      thinking.innerHTML = '<span class="error">' + err.message + '</span>';
    } finally {
      load.stop();
      send.disabled = false;
      scroll();
    }
  });

  resetBtn.addEventListener('click', async () => {
    if (sessionId) {
      const fd = new FormData(); fd.append('session_id', sessionId);
      await fetch('/reset', { method: 'POST', body: fd });
    }
    sessionId = ''; chat.innerHTML = ''; pendingFiles = []; renderPending();
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
