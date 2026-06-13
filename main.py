import base64
import datetime
import html as html_lib
import io
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)  # override shell env so .env always wins

import anthropic
from anthropic import AsyncAnthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="VeriLex")

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

def _extract_json(raw: str) -> dict:
    """
    Robustly extract and parse a JSON object from a Claude response string.
    Handles cases where non-ASCII characters (Chinese full-width quotes, etc.)
    or unescaped ASCII double-quotes inside string values cause json.loads to fail.
    """
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    candidate = raw[start:end]

    # First attempt: standard parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Second attempt: permissive parse (allows control characters in strings)
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        pass

    # Third attempt: replace common full-width / curly quote variants
    # that Claude sometimes emits when generating non-Latin text
    cleaned = (
        candidate
        .replace('“', '\\"').replace('”', '\\"')  # "..." → escaped quotes
        .replace('‘', "\\'").replace('’', "\\'")  # '...' → escaped quotes
        .replace('「', '').replace('」', '')        # 「」 → removed
        .replace('\x00', '')                               # null bytes
    )
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    # Fourth attempt: use json-repair to fix unescaped quotes and other common
    # issues that arise when Claude generates Chinese or other non-Latin output.
    try:
        from json_repair import repair_json
        repaired = repair_json(candidate, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
    except Exception:
        pass

    raise ValueError("Could not parse JSON from model response after all repair attempts")


def _ocr_quality_ok(text: str) -> bool:
    """Return False if OCR output looks like a failed extraction rather than real content."""
    if len(text) < 20:
        return False
    # High '?' ratio is a strong signal of encoding failure / unreadable chars
    if text.count("?") / len(text) > 0.35:
        return False
    # Near-zero character variety means repeated garbage (e.g. "RRRRRRRR")
    payload = text.lower().replace(" ", "").replace("\n", "")
    if payload and len(set(payload)) < 4:
        return False
    return True


# JSON key names in schema order — used to emit phase labels while streaming.
_STREAM_PHASES = [
    (re.compile(r'"document_type"\s*:'),   "Reading document…"),
    (re.compile(r'"plain_summary"\s*:'),   "Summarizing content…"),
    (re.compile(r'"applicable_laws"\s*:'), "Finding applicable laws…"),
    (re.compile(r'"glossary"\s*:'),        "Building glossary…"),
    (re.compile(r'"local_resources"\s*:'), "Locating resources…"),
]

def _detect_phase(buf: str) -> str | None:
    """Return the label of the last JSON phase whose key has appeared in buf."""
    last = None
    for pattern, label in _STREAM_PHASES:
        if pattern.search(buf):
            last = label
    return last


# Domains that VeriLex is allowed to cite — anything else is a hallucination signal.
_ALLOWED_STATUTE_DOMAINS = {"leginfo.legislature.ca.gov", "law.cornell.edu"}


def _should_verify(result: dict) -> bool:
    """Return True when the analysis warrants a second-pass accuracy check."""
    if result.get("case_citations"):          # must always be []
        return True
    if result.get("urgency_level") in ("critical", "high"):
        return True
    for law in result.get("applicable_laws", []):
        if law.get("url"):                    # has statute URL — worth checking
            return True
    return False


async def _verify_result(result: dict, api_key: str) -> tuple[bool, list[str]]:
    """
    Lightweight second-pass accuracy check via Haiku.
    Returns (passed: bool, issues: list[str]).
    Never raises — a verification failure never blocks the user.
    """
    check_payload = {
        "document_type": result.get("document_type"),
        "urgency_level": result.get("urgency_level"),
        "key_dates": result.get("key_dates", []),
        "applicable_laws": result.get("applicable_laws", []),
        "case_citations": result.get("case_citations", []),
    }
    verify_prompt = (
        "Check this legal document analysis for rule violations.\n\n"
        "RULES:\n"
        "1. case_citations must be an empty array. Flag any non-empty value.\n"
        "2. urgency_level 'critical' or 'high' requires at least one entry in key_dates.\n"
        "3. applicable_laws URLs must only use leginfo.legislature.ca.gov or law.cornell.edu. "
        "Flag any other domains. An empty URL string is acceptable.\n"
        "4. urgency_level must be exactly one of: critical, high, medium, low.\n\n"
        f"ANALYSIS:\n{json.dumps(check_payload, ensure_ascii=False)}\n\n"
        'Respond with ONLY valid JSON: {"pass": true, "issues": []} or '
        '{"pass": false, "issues": ["description of issue"]}'
    )
    try:
        client = AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": verify_prompt}],
        )
        raw = response.content[0].text.strip()
        parsed = _extract_json(raw)
        return bool(parsed.get("pass", True)), list(parsed.get("issues", []))
    except Exception:
        return True, []  # verification failure never blocks the user


# In-memory feedback store (persists until server restarts).
feedback_db: list[dict] = []

# In-memory usage log (persists until server restarts).
usage_log: list[dict] = []


class AnalyzeRequest(BaseModel):
    text: str
    language: str = "English"
    persona: str = "intermediate"
    user_location: str = ""  # e.g. "Oakland, California, United States"


class FeedbackRequest(BaseModel):
    name: str = ""
    email: str = ""
    category: str = "general"
    message: str


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class FollowUpRequest(BaseModel):
    question: str
    document_text: str
    prior_analysis: dict
    language: str = "English"
    persona: str = "intermediate"
    conversation: list[ChatMessage] = []


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return FileResponse("index.html")


@app.get("/robots.txt")
async def robots():
    content = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /admin\n"
        "Disallow: /analyze\n"
        "Disallow: /analyze-stream\n"
        "Disallow: /followup\n"
        "Disallow: /extract-pdf\n"
        "Disallow: /extract-image\n"
        "Disallow: /feedback\n"
        "\n"
        "Sitemap: https://verilex.org/sitemap.xml\n"
    )
    return Response(content=content, media_type="text/plain")


@app.get("/sitemap.xml")
async def sitemap():
    content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://verilex.org/</loc>
    <changefreq>monthly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://verilex.org/privacy</loc>
    <changefreq>yearly</changefreq>
    <priority>0.3</priority>
  </url>
  <url>
    <loc>https://verilex.org/terms</loc>
    <changefreq>yearly</changefreq>
    <priority>0.3</priority>
  </url>
  <url>
    <loc>https://verilex.org/transparency</loc>
    <changefreq>monthly</changefreq>
    <priority>0.5</priority>
  </url>
</urlset>"""
    return Response(content=content, media_type="application/xml")


@app.get("/og-image.png")
async def og_image():
    return FileResponse("og-image.png", media_type="image/png")


@app.get("/privacy")
async def privacy():
    return FileResponse("privacy.html")


@app.get("/terms")
async def terms():
    return FileResponse("terms.html")


@app.get("/transparency")
async def transparency():
    return FileResponse("transparency.html")


@app.get("/config")
async def get_config():
    """Returns public client-side config (never secrets like the Anthropic key)."""
    return JSONResponse({
        "google_client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
        "google_api_key":   os.environ.get("GOOGLE_API_KEY", ""),
    })


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


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
    location_ctx = (
        f"USER LOCATION: {req.user_location}. "
        f"Prioritize legal aid organizations, court resources, and statutes specific to this location. "
        f"In local_resources, include real organizations that serve this area, with accurate phone numbers and URLs.\n\n"
        if req.user_location else
        "USER LOCATION: Unknown. Include a mix of national resources and note that local resources may vary by state.\n\n"
    )
    if req.language.lower() in ("auto-detect", "auto"):
        lang_instruction = (
            "Detect the language of the document text and respond entirely in that same language "
            "throughout every field of your JSON response. If the document is in Spanish, respond in Spanish. "
            "If it is in Chinese, respond in Chinese. Match the document's language exactly."
        )
    else:
        lang_instruction = (
            f"The document below may be written in any language. Analyze it fully regardless of the document's language, "
            f"then respond entirely in {req.language}."
        )

    user_msg = (
        f"{persona_ctx}\n\n"
        f"{location_ctx}"
        f"{lang_instruction}\n\n"
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
        result = _extract_json(raw)

        v_triggered = _should_verify(result)
        v_passed, v_issues = True, []
        if v_triggered:
            v_passed, v_issues = await _verify_result(result, api_key)
            if not v_passed:
                result["_warnings"] = [
                    "Some parts of this analysis could not be fully verified. "
                    "Please cross-check specific statute references and deadlines "
                    "with official sources before acting."
                ]

        usage_log.append({
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
            "language": req.language,
            "persona": req.persona,
            "location": req.user_location or "Unknown",
            "doc_length": len(text),
            "document_type": result.get("document_type", "unknown"),
            "success": True,
            "verification_triggered": v_triggered,
            "verification_passed": v_passed,
        })
        return JSONResponse(content=result)

    except (json.JSONDecodeError, ValueError) as e:
        usage_log.append({
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
            "language": req.language,
            "persona": req.persona,
            "location": req.user_location or "Unknown",
            "doc_length": len(text),
            "document_type": "parse_error",
            "success": False,
        })
        raise HTTPException(status_code=500, detail=f"Failed to parse response: {e}")
    except anthropic.APIError as e:
        usage_log.append({
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
            "language": req.language,
            "persona": req.persona,
            "location": req.user_location or "Unknown",
            "doc_length": len(text),
            "document_type": "api_error",
            "success": False,
        })
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")


@app.post("/analyze-stream")
async def analyze_stream(req: AnalyzeRequest):
    """
    Streaming version of /analyze. Returns Server-Sent Events.
    Event types:
      phase  — data: {"phase": "<label>"}     fires up to 5 times as Claude writes each JSON section
      result — data: <full analysis JSON>     fires once when complete
      error  — data: {"detail": "<message>"}  fires on failure
    """
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
    location_ctx = (
        f"USER LOCATION: {req.user_location}. "
        f"Prioritize legal aid organizations, court resources, and statutes specific to this location. "
        f"In local_resources, include real organizations that serve this area, with accurate phone numbers and URLs.\n\n"
        if req.user_location else
        "USER LOCATION: Unknown. Include a mix of national resources and note that local resources may vary by state.\n\n"
    )
    if req.language.lower() in ("auto-detect", "auto"):
        lang_instruction = (
            "Detect the language of the document text and respond entirely in that same language "
            "throughout every field of your JSON response. If the document is in Spanish, respond in Spanish. "
            "If it is in Chinese, respond in Chinese. Match the document's language exactly."
        )
    else:
        lang_instruction = (
            f"The document below may be written in any language. Analyze it fully regardless of the document's language, "
            f"then respond entirely in {req.language}."
        )

    user_msg = (
        f"{persona_ctx}\n\n"
        f"{location_ctx}"
        f"{lang_instruction}\n\n"
        f"DOCUMENT:\n{text}"
    )

    async def event_stream():
        buf = ""
        last_phase = None
        log_entry = {
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
            "language": req.language,
            "persona": req.persona,
            "location": req.user_location or "Unknown",
            "doc_length": len(text),
            "document_type": "unknown",
            "success": False,
        }
        try:
            client = AsyncAnthropic(api_key=api_key)
            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                async for chunk in stream.text_stream:
                    buf += chunk
                    phase = _detect_phase(buf)
                    if phase and phase != last_phase:
                        last_phase = phase
                        yield f"event: phase\ndata: {json.dumps({'phase': phase})}\n\n"

            result = _extract_json(buf)

            v_triggered = _should_verify(result)
            v_passed, v_issues = True, []
            if v_triggered:
                v_passed, v_issues = await _verify_result(result, api_key)
                if not v_passed:
                    result["_warnings"] = [
                        "Some parts of this analysis could not be fully verified. "
                        "Please cross-check specific statute references and deadlines "
                        "with official sources before acting."
                    ]

            log_entry.update(
                document_type=result.get("document_type", "unknown"),
                success=True,
                verification_triggered=v_triggered,
                verification_passed=v_passed,
            )
            usage_log.append(log_entry)
            yield f"event: result\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"

        except (json.JSONDecodeError, ValueError) as e:
            log_entry["document_type"] = "parse_error"
            usage_log.append(log_entry)
            yield f"event: error\ndata: {json.dumps({'detail': f'Failed to parse response: {e}'})}\n\n"
        except anthropic.APIError as e:
            log_entry["document_type"] = "api_error"
            usage_log.append(log_entry)
            yield f"event: error\ndata: {json.dumps({'detail': f'AI service error: {e}'})}\n\n"
        except Exception as e:
            log_entry["document_type"] = "unknown_error"
            usage_log.append(log_entry)
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables nginx proxy buffering on Railway
        },
    )


FOLLOWUP_SYSTEM = """\
You are VeriLex, a legal document accessibility assistant. The user has already received a full analysis of their document. They now have follow-up questions.

STRICT RULES:
1. Only answer questions about the specific document and analysis provided. Do not answer general legal questions unrelated to this document.

2. NEVER give legal advice. If the user asks what they should do, whether to sign, whether to contest, whether they have a strong case, or any other action recommendation: explain what the document says about the relevant issue, then say — "For advice specific to your situation, please consult a licensed attorney or a free legal aid organization."

3. NO FABRICATION: Never invent or guess information that is not explicitly present in the document text or prior analysis. If the answer to a question is not in the document, say clearly: "I don't see that specific information in your document." Do not infer, calculate, or speculate beyond what the document states.

4. QUOTE DIRECTLY: When referencing something from the document, quote or paraphrase from the actual text. Never construct a quote or paraphrase from memory.

5. ACKNOWLEDGE UNCERTAINTY: If you are unsure about something, say so explicitly. Use phrases like "the document appears to say…" or "based on this clause…" rather than stating uncertain things as facts.

6. Be concise. Answer the specific question asked without repeating the full analysis.

7. Respond entirely in the language specified in the request."""


@app.post("/followup")
async def followup(req: FollowUpRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Server configuration error.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question too long (max 2000 characters).")

    client = anthropic.Anthropic(api_key=api_key)

    # Document + analysis as a cached first turn so repeated follow-ups are cheap
    if req.language.lower() in ("auto-detect", "auto"):
        lang_note = "Detect the language of the document and respond in that same language."
    else:
        lang_note = f"Respond entirely in {req.language}."

    doc_context = (
        f"DOCUMENT TEXT:\n{req.document_text}\n\n"
        f"PRIOR ANALYSIS:\n{json.dumps(req.prior_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"{lang_note}"
    )

    messages: list[dict] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": doc_context, "cache_control": {"type": "ephemeral"}}],
        },
        {
            "role": "assistant",
            "content": "I have read the document and analysis. What would you like to know?",
        },
    ]

    for msg in req.conversation:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": question})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[{"type": "text", "text": FOLLOWUP_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        answer = response.content[0].text.strip()
        return {"answer": answer}
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
        # Require a meaningful amount of text to trust the extraction.
        # Low character counts suggest encoding failure (common with some RTL/CJK PDFs).
        if text and len(text) > 30 * num_pages:
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
        # JPEG at 1.5× zoom keeps text sharp while being ~75% smaller than 2× PNG,
        # which dramatically reduces API payload size and processing time.
        vision_content = []
        for i, page in enumerate(pdf_doc):
            if i >= 10:
                break
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            jpeg_bytes = pix.tobytes("jpg", jpg_quality=85)
            # If a page is still very large (e.g. high-DPI scan), compress further
            if len(jpeg_bytes) > 1_500_000:
                jpeg_bytes = pix.tobytes("jpg", jpg_quality=65)
            img_b64 = base64.standard_b64encode(jpeg_bytes).decode()
            vision_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
            })

        vision_content.append({
            "type": "text",
            "text": (
                "This is a scanned or handwritten document. It may be in any language or script, "
                "including Arabic, Chinese, Korean, Hindi, Russian, Hebrew, or other non-Latin scripts. "
                "Transcribe all visible text exactly as it appears in its original language and script. "
                "Do not translate. Preserve the document's structure and layout. "
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
        if not extracted or not _ocr_quality_ok(extracted):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not reliably read this document — the scan quality may be too low, "
                    "the script may be unrecognized, or the page is mostly images. "
                    "Try a higher-resolution scan, better lighting, or paste the text directly."
                )
            )

        return {"text": extracted[:60_000], "pages": num_pages, "method": "ocr"}

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF vision support unavailable on this server.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")


@app.post("/extract-image")
async def extract_image(file: UploadFile = File(...)):
    """Use Claude Vision to read text from a JPEG/PNG photo of a legal document."""
    fname = (file.filename or "").lower()
    if not any(fname.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: ANTHROPIC_API_KEY is not set."
        )

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 20 MB).")

    # Determine media type
    if fname.endswith(".png"):
        media_type = "image/png"
    elif fname.endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"

    img_b64 = base64.standard_b64encode(content).decode()

    vision_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img_b64},
        },
        {
            "type": "text",
            "text": (
                "This is a photo of a legal document. It may be in any language or script, "
                "including Arabic, Chinese, Korean, Hindi, Russian, Hebrew, or other non-Latin scripts. "
                "Transcribe all visible text exactly as it appears in its original language and script. "
                "Do not translate. Preserve the document's structure and layout. "
                "Return only the transcribed text — no commentary, no explanation."
            ),
        },
    ]

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": vision_content}],
        )
        extracted = response.content[0].text.strip()
        if not extracted or not _ocr_quality_ok(extracted):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not reliably read this document — the image quality may be too low, "
                    "the script may be unrecognized, or the page is mostly non-text. "
                    "Try a clearer photo with better lighting, or paste the text directly."
                )
            )
        return {"text": extracted[:60_000], "method": "ocr"}

    except HTTPException:
        raise
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")


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

    # ── Usage stats ──────────────────────────────────────────────────────────
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    week_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

    total_analyses   = len(usage_log)
    today_analyses   = sum(1 for u in usage_log if u["date"] == today)
    week_analyses    = sum(1 for u in usage_log if u["date"] >= week_ago)
    successful       = sum(1 for u in usage_log if u["success"])
    success_rate     = f"{100 * successful // total_analyses}%" if total_analyses else "—"
    v_triggered      = sum(1 for u in usage_log if u.get("verification_triggered"))
    v_flagged        = sum(1 for u in usage_log if u.get("verification_triggered") and not u.get("verification_passed", True))

    # Counts by field
    def top_counts(field: str, n: int = 5) -> list[tuple[str, int]]:
        from collections import Counter
        return Counter(u[field] for u in usage_log).most_common(n)

    lang_rows = "".join(
        f"<tr><td>{html_lib.escape(k)}</td><td style='text-align:right;font-weight:600'>{v}</td></tr>"
        for k, v in top_counts("language")
    ) or "<tr><td colspan=2 style='color:#94a3b8'>No data</td></tr>"

    persona_rows = "".join(
        f"<tr><td>{html_lib.escape(k)}</td><td style='text-align:right;font-weight:600'>{v}</td></tr>"
        for k, v in top_counts("persona")
    ) or "<tr><td colspan=2 style='color:#94a3b8'>No data</td></tr>"

    doctype_rows = "".join(
        f"<tr><td>{html_lib.escape(k)}</td><td style='text-align:right;font-weight:600'>{v}</td></tr>"
        for k, v in top_counts("document_type")
    ) or "<tr><td colspan=2 style='color:#94a3b8'>No data</td></tr>"

    location_rows = "".join(
        f"<tr><td>{html_lib.escape(k)}</td><td style='text-align:right;font-weight:600'>{v}</td></tr>"
        for k, v in top_counts("location")
    ) or "<tr><td colspan=2 style='color:#94a3b8'>No data</td></tr>"

    # Recent analyses (last 10)
    recent_rows = ""
    for u in reversed(usage_log[-10:]):
        status_color = "#15803d" if u["success"] else "#b91c1c"
        status_label = "✓" if u["success"] else "✗"
        recent_rows += f"""
        <tr>
          <td style='color:#64748b;font-size:12px'>{u['timestamp']}</td>
          <td>{html_lib.escape(u['document_type'])}</td>
          <td>{html_lib.escape(u['language'])}</td>
          <td>{html_lib.escape(u['persona'])}</td>
          <td style='color:#64748b;font-size:12px'>{html_lib.escape(u['location'][:30])}</td>
          <td style='text-align:center;color:{status_color};font-weight:700'>{status_label}</td>
        </tr>"""
    if not recent_rows:
        recent_rows = "<tr><td colspan=6 style='color:#94a3b8;text-align:center;padding:20px'>No analyses yet.</td></tr>"

    # ── Feedback rows ─────────────────────────────────────────────────────────
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
<html><head><title>VeriLex — Admin</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#f8f7f4;color:#1a1a2e;margin:0;padding:32px 16px}}
  .wrap{{max-width:760px;margin:0 auto}}
  h1{{font-size:22px;font-weight:800;margin-bottom:4px}}
  h2{{font-size:16px;font-weight:700;margin:32px 0 12px;color:#334155}}
  .meta{{color:#64748b;font-size:14px;margin-bottom:28px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:8px}}
  .card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center}}
  .card .num{{font-size:28px;font-weight:800;color:#2563eb}}
  .card .lbl{{font-size:12px;color:#64748b;margin-top:2px}}
  table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;font-size:14px}}
  th{{background:#f1f5f9;padding:8px 12px;text-align:left;font-size:12px;color:#64748b;font-weight:600}}
  td{{padding:8px 12px;border-top:1px solid #f1f5f9}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  @media(max-width:520px){{.grid2{{grid-template-columns:1fr}}}}
</style>
</head><body>
<div class="wrap">
  <h1>⚖️ VeriLex — Admin</h1>
  <p class="meta">Usage analytics &amp; feedback &nbsp;·&nbsp; data since last deploy</p>

  <h2>📊 Usage Overview</h2>
  <div class="cards">
    <div class="card"><div class="num">{total_analyses}</div><div class="lbl">Total analyses</div></div>
    <div class="card"><div class="num">{today_analyses}</div><div class="lbl">Today</div></div>
    <div class="card"><div class="num">{week_analyses}</div><div class="lbl">Last 7 days</div></div>
    <div class="card"><div class="num">{success_rate}</div><div class="lbl">Success rate</div></div>
    <div class="card"><div class="num">{len(feedback_db)}</div><div class="lbl">Feedback items</div></div>
    <div class="card"><div class="num">{v_triggered}</div><div class="lbl">Verifications run</div></div>
    <div class="card"><div class="num" style="color:{'#b91c1c' if v_flagged else '#15803d'}">{v_flagged}</div><div class="lbl">Verification flags</div></div>
  </div>

  <h2>🕐 Recent Analyses</h2>
  <table>
    <tr><th>Time (UTC)</th><th>Doc type</th><th>Language</th><th>Persona</th><th>Location</th><th>✓</th></tr>
    {recent_rows}
  </table>

  <h2>🔍 Breakdowns</h2>
  <div class="grid2">
    <div>
      <table>
        <tr><th colspan=2>Top Languages</th></tr>
        {lang_rows}
      </table>
    </div>
    <div>
      <table>
        <tr><th colspan=2>Persona Level</th></tr>
        {persona_rows}
      </table>
    </div>
    <div>
      <table>
        <tr><th colspan=2>Document Types</th></tr>
        {doctype_rows}
      </table>
    </div>
    <div>
      <table>
        <tr><th colspan=2>Top Locations</th></tr>
        {location_rows}
      </table>
    </div>
  </div>

  <h2>💬 Feedback</h2>
  {rows}
</div>
</body></html>"""

    return HTMLResponse(content=page)
