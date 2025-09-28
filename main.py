# backend/main.py
import os
import uuid
import asyncio
import json
import re
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from gtts import gTTS
import google.generativeai as genai

# ------------------ Config ------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

genai.configure(api_key=GEMINI_API_KEY)

GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)

# ------------------ App ------------------
app = FastAPI(title="CogniPath Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Jobs Store ------------------
JOBS = {}

# ------------------ Helpers ------------------
def save_tts_mp3(text: str, filename: str) -> str:
    path = os.path.join(GENERATED_DIR, f"{filename}.mp3")
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(path)
    return path

async def generate_with_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        return "Error generating content"

def extract_json(text: str):
    try:
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"JSON parsing error: {e}")
    return {}

# ------------------ Job Processor ------------------
async def process_job(job_id: str, content: str):
    try:
        JOBS[job_id]["status"] = "processing"

        # ------------------ Gemini prompts ------------------
        summary_prompt = f"Summarize in simple terms:\n\n{content}"

        graph_prompt = f"""
Extract the key concepts from the following text and return a JSON object formatted as a mind map.
Format strictly as:
{{
  "nodes": [{{"id": "n1", "label": "concept1"}}, ...],
  "edges": [{{"from": "n1", "to": "n2"}}, ...]
}}
Text:
{content}
"""

        quiz_prompt = f"""
Generate educational content from the following text. Return a JSON object with exactly:
{{
  "mcq": [
    {{
      "question": "Your question here",
      "options": ["A)", "B)", "C)", "D)"],
      "answer": "Correct option letter"
    }},
    ...
  ],
  "flashcards": [
    {{
      "front": "Question front text",
      "back": "Answer back text"
    }},
    ...
  ]
}}
Make sure all MCQs have 4 options labeled A, B, C, D and include 2 flashcards.
Text:
{content}
"""

        # ------------------ Run tasks concurrently ------------------
        summary_task = asyncio.create_task(generate_with_gemini(summary_prompt))
        graph_task = asyncio.create_task(generate_with_gemini(graph_prompt))
        quiz_task = asyncio.create_task(generate_with_gemini(quiz_prompt))

        summary_res, graph_res, quiz_res = await asyncio.gather(summary_task, graph_task, quiz_task)

        # ------------------ Parse JSON safely ------------------
        graph_json = extract_json(graph_res)
        if not graph_json.get("nodes"):
            graph_json = {"nodes": [{"id": "n1", "label": "Key concepts"}], "edges": []}

        quiz_json = extract_json(quiz_res)
        if not quiz_json.get("mcq"):
            quiz_json["mcq"] = []
        if not quiz_json.get("flashcards"):
            quiz_json["flashcards"] = []

        # ------------------ TTS ------------------
        audio_path = save_tts_mp3(summary_res, job_id)

        # ------------------ Store result ------------------
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "summary": summary_res,
            "graph": graph_json,
            "quiz": quiz_json,
            "audio_path": audio_path,
        }

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["result"] = {"error": str(e)}

# ------------------ Endpoints ------------------
@app.post("/process")
async def process_endpoint(content: str = Form(...)):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content is required")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "result": None}
    asyncio.create_task(process_job(job_id, content))
    return {"job_id": job_id}

@app.get("/hub/{job_id}")
async def get_hub(job_id: str):
    entry = JOBS.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found")

    resp = {"status": entry["status"]}
    if entry["status"] == "done":
        res = entry["result"]
        resp["result"] = {
            "summary": res["summary"],
            "graph": res["graph"],
            "quiz": res["quiz"],
            "audio_url": f"/audio/{res['audio_path'].split('/')[-1]}" if res.get("audio_path") else None
        }
    elif entry["status"] == "error":
        resp["result"] = entry.get("result")
    return JSONResponse(resp)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)
