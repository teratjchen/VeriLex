import base64
import datetime
import html as html_lib
import io
import json
import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="LawLight")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load system prompt from prompt.txt — edit that file to change the AI's behavior.
SYSTEM_PROMPT = Path("prompt.txt").read_text()

# How responses are tailored per persona — added to each request (not the cached prompt).
PERSONA_INSTRUCTIONS = {
    "novice": (
        "USER KNOWLEDGE LEVEL: Complete beginner — no legal background whatsoever. "
        "Write at a 5th-grade reading level. Use zero jargon; if a legal term is unavoidable, "
        "immediately explain it in parentheses using everyday words. "
        "Use relatable analogies (e.g. 'think of this like a bill from a store'). "
        "Be warm, clear, and reassuring in tone — avoid anything that feels intimidating. "
        "In the glossary, define even basic-sounding terms. "
        "In rights_under_law, be concrete about what this means for an everyday person. "
        "Include all available local resources."
    ),
    "intermediate": (
        "USER KNOWLEDGE LEVEL: General public with a basic understanding of how legal systems work. "
        "Write in plain English. You may use common legal terms (like 'eviction', 'defendant', 'statute') "
        "but briefly explain any specialized or technical language. "
        "Balanced depth — thorough but not overwhelming. Standard tone."
    ),
    "expert": (
        "USER KNOWLEDGE LEVEL: Legal professional, law student, or highly experienced person. "
        "You may use full legal terminology without lay explanations. "
        "Provide deeper statutory analysis and more case law citations. "
        "Plain summaries can be concise — prioritize legal precision over simplicity. "
        "Include procedural nuances and any minority or split-jurisdiction considerations where relevant."
    ),
}

# In-memory feedback store (persists until server restarts).
feedback_db: list[dict] = []


class AnalyzeRequest(BaseModel):
    text: str
    language: str = "English"
    persona: str = "intermediate"


class FeedbackRequest(BaseModel):
    name: str = ""
    email: str = ""
    category: str = "general"
    message: str


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: ANTHROPIC_API_KEY is not set."
        )

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No document text provided.")
    if len(text) > 60_000:
        raise HTTPException(
            status_code=400,
            detail="Document too long (max ~40 pages). Try pasting a specific section."
        )

    persona_ctx = PERSONA_INSTRUCTIONS.get(req.persona, PERSONA_INSTRUCTIONS["intermediate"])
    user_msg = (
        f"{persona_ctx}\n\n"
        f"Respond entirely in {req.language}.\n\n"
        f"DOCUMENT:\n{text}"
    )

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        result = json.loads(raw[start:end])
        return JSONResponse(content=result)

    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse response: {e}")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    content = await file.read()

    # ── Step 1: Fast text extraction for digital PDFs ─────────────────────────
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            num_pages = len(pdf.pages)
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n\n".join(pages).strip()
        if text:
            return {"text": text[:60_000], "pages": num_pages, "method": "text"}
    except Exception:
        num_pages = 0

    # ── Step 2: Claude Vision OCR for scanned / handwritten PDFs ─────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text — PDF appears to be scanned or handwritten. Try copying and pasting the text instead."
        )

    try:
        import fitz  # PyMuPDF — no system-level dependencies required

        pdf_doc = fitz.open(stream=content, filetype="pdf")
        num_pages = len(pdf_doc)

        # Convert pages to images (cap at 10 pages to keep costs reasonable)
        vision_content = []
        for i, page in enumerate(pdf_doc):
            if i >= 10:
                break
            mat = fitz.Matrix(2.0, 2.0)  # 2× zoom for better handwriting recognition
            pix = page.get_pixmap(matrix=mat)
            img_b64 = base64.standard_b64encode(pix.tobytes("png")).decode()
            vision_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
            })

        vision_content.append({
            "type": "text",
            "text": (
                "This is a scanned or handwritten document. "
                "Transcribe all visible text exactly as written, preserving the document's structure and layout. "
                "Return only the transcribed text — no commentary, no explanation."
            ),
        })

        client = anthropic.Anthropic(api_key=api_key)
        ocr_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": vision_content}],
        )

        extracted = ocr_response.content[0].text.strip()
        if not extracted:
            raise HTTPException(
                status_code=400,
                detail="Could not read text — image quality may be too low. Try a clearer scan."
            )

        return {"text": extracted[:60_000], "pages": num_pages, "method": "ocr"}

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF vision support unavailable on this server.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    feedback_db.append({
        "id": len(feedback_db) + 1,
        "name": req.name.strip() or "Anonymous",
        "email": req.email.strip(),
        "category": req.category,
        "message": req.message.strip(),
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    })
    return {"ok": True}


@app.get("/admin")
async def admin_view(key: str = ""):
    admin_key = os.environ.get("ADMIN_KEY", "")
    if not admin_key or key != admin_key:
        raise HTTPException(status_code=403, detail="Access denied.")

    CAT_COLORS = {
        "question":   ("#dbeafe", "#1d4ed8"),
        "suggestion": ("#dcfce7", "#15803d"),
        "complaint":  ("#fee2e2", "#b91c1c"),
        "other":      ("#f1f5f9", "#475569"),
    }

    rows = ""
    for item in reversed(feedback_db):
        bg, fg = CAT_COLORS.get(item["category"], CAT_COLORS["other"])
        rows += f"""
        <div style="border:1px solid #e2e8f0;border-radius:10px;padding:18px;margin-bottom:14px;background:#fff">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <strong style="font-size:15px">#{item['id']} &nbsp;{html_lib.escape(item['name'])}</strong>
            <span style="background:{bg};color:{fg};padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">
              {item['category']}
            </span>
          </div>
          <p style="margin:0 0 10px;line-height:1.6">{html_lib.escape(item['message'])}</p>
          <small style="color:#94a3b8">{item['timestamp']}{(' &nbsp;·&nbsp; <a href="mailto:' + html_lib.escape(item['email']) + '" style="color:#3b82f6">' + html_lib.escape(item['email']) + '</a>') if item.get('email') else ''}</small>
        </div>"""

    if not rows:
        rows = "<p style='color:#94a3b8;text-align:center;padding:40px 0'>No feedback yet.</p>"

    page = f"""<!DOCTYPE html>
<html><head><title>LawLight — Feedback Admin</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#f8f7f4;color:#1a1a2e;margin:0;padding:32px 16px}}
  .wrap{{max-width:680px;margin:0 auto}}
  h1{{font-size:22px;font-weight:800;margin-bottom:4px}}
  .meta{{color:#64748b;font-size:14px;margin-bottom:28px}}
</style>
</head><body>
<div class="wrap">
  <h1>⚖️ LawLight — Feedback</h1>
  <p class="meta">{len(feedback_db)} submission(s) total &nbsp;·&nbsp; newest first</p>
  {rows}
</div>
</body></html>"""

    return HTMLResponse(content=page)
