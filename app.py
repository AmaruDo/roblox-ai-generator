import os
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Roblox AI Model Generator")
HTML_PATH = Path(__file__).parent / "index.html"
HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=2, max_length=2000)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PATH.read_text(encoding="utf-8")

@app.get("/health")
async def health():
    return {"ok": True, "service": "Roblox AI Model Generator"}

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="Server ešte nemá nastavený HF_TOKEN. Pridaj ho v Railway → Variables.")
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert Roblox Studio Luau developer. Return only a complete, safe Luau script. Prefer creating a Model with anchored Parts, clear names, comments, and no external asset dependencies. Do not use loadstring, HTTP requests, obfuscated code, or exploit APIs. If the prompt is unclear, make a reasonable simple model. The script should be suitable for a normal Script in Roblox Studio."},
            {"role": "user", "content": "Create Roblox Studio Luau for this model: " + req.prompt}
        ],
        "temperature": 0.25,
        "max_tokens": 1800
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(HF_ENDPOINT, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=body)
        if response.status_code >= 400:
            detail = "Hugging Face API chyba (" + str(response.status_code) + "). Skontroluj token, model a dostupnosť inference providerov."
            try:
                data = response.json()
                if isinstance(data, dict) and data.get("error"):
                    detail += " " + str(data["error"])[:350]
            except Exception:
                pass
            raise HTTPException(status_code=502, detail=detail)
        data = response.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not text:
            raise HTTPException(status_code=502, detail="AI vrátila prázdny výsledok. Skús kratší prompt.")
        # Strip common markdown fences if the model adds them.
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return {"code": text, "model": model}
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI odpovedala príliš pomaly. Skús to znova.")
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Nepodarilo sa spojiť s AI službou: " + str(exc)[:250])
