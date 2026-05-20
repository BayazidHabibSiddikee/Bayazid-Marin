import os
import re
import json
import asyncio
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bayazid import (
    main as bayazid_main,
    teach_topic, generate_quiz, create_study_plan,
    review_code, explain_error,
    handle_timer_command, format_study_context,
    timer, memory
)
from classifier import classify, extract_timer_task, extract_topic, extract_quiz_params
from config import UPLOAD_FOLDER, HOST, PORT

app = FastAPI(title="Bayazid HS-02")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── PAGE ROUTES ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/chat", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse(request=request, name="bayazid_chat.html")

@app.get("/profile", response_class=HTMLResponse)
async def get_profile(request: Request):
    return templates.TemplateResponse(request=request, name="profile.html")


# ── UPLOAD ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if not image.filename:
        return JSONResponse({"error": "No filename"}, status_code=400)
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', image.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buf:
        buf.write(await image.read())
    return {"ok": True, "path": f"/{filepath}"}


# ── MAIN CHAT ENDPOINT ────────────────────────────────────────────────────

@app.post("/message")
async def handle_message(
    message: str = Form(...),
    image: UploadFile = File(None),
    study_context: str = Form(None)
):
    image_path = None
    if image and image.filename:
        filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(image_path, "wb") as buf:
            buf.write(await image.read())
        image_path = os.path.abspath(image_path)

    clf = classify(message)
    intent = clf["intent"]
    sub = clf.get("sub_intent")

    # Route by intent
    if intent == "timer":
        task = extract_timer_task(message)
        result = await handle_timer_command(sub or "status", task)
        async def timer_stream():
            yield result
        return StreamingResponse(timer_stream(), media_type="text/plain")

    elif intent == "teach":
        topic = extract_topic(message)
        depth = sub or "standard"
        return StreamingResponse(teach_topic(topic, depth), media_type="text/plain")

    elif intent == "study_plan":
        topic = extract_topic(message)
        return StreamingResponse(create_study_plan(topic), media_type="text/plain")

    elif intent == "code_review":
        # Extract code block if present
        code_match = re.search(r'```[\w]*\n?([\s\S]+?)```', message)
        code = code_match.group(1) if code_match else message
        return StreamingResponse(review_code(code), media_type="text/plain")

    elif intent == "debug":
        error_match = re.search(r'```[\w]*\n?([\s\S]+?)```', message)
        error_text = error_match.group(1) if error_match else message
        return StreamingResponse(explain_error(error_text), media_type="text/plain")

    else:
        # General chat — pass through to main
        ctx = format_study_context(json.loads(study_context)) if study_context else None
        return StreamingResponse(
            bayazid_main(message, image_path=image_path, study_context=ctx),
            media_type="text/plain"
        )


# ── QUIZ ENDPOINT (streams the quiz as plain text) ───────────────────────

@app.post("/quiz/generate")
async def generate_quiz_endpoint(
    topic: str = Form(...),
    difficulty: str = Form("medium"),
    num_questions: int = Form(5)
):
    return StreamingResponse(
        generate_quiz(topic, difficulty, num_questions),
        media_type="text/plain"
    )


# ── TIMER API ─────────────────────────────────────────────────────────────

@app.post("/timer/{command}")
async def timer_command(command: str, task: str = Form("")):
    result = await handle_timer_command(command, task)
    return JSONResponse({"message": result, "stats": timer.get_stats()})

@app.get("/timer/stats")
async def get_timer_stats():
    return JSONResponse(timer.get_stats())


# ── MEMORY MANAGEMENT ─────────────────────────────────────────────────────

@app.post("/memory/clear")
async def clear_memory():
    memory.clear()
    return JSONResponse({"ok": True, "message": "Memory cleared."})

@app.get("/memory/status")
async def memory_status():
    return JSONResponse({
        "message_count": len(memory.messages),
        "messages": memory.get()
    })


# ── HEALTH ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "operational", "codename": "BAYAZID HS-02"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
