from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import requests
import io
from PIL import Image
import re
from dotenv import load_dotenv
import os
import logging
import mimetypes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Enable CORS for Live Server compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


from typing import Optional

@app.post("/upload_and_query")
async def upload_and_query(query: str = Form(...), image: Optional[UploadFile] = File(None)):
    try:
        messages_content = []
        
        medical_prompt = f"""You are a careful medical assistant AI.

Analyze the user's question and image if provided.

Give a SHORT, clear, and practical response. Output ONLY the final medical answer.

Do NOT include any reasoning, thinking process, self-review, internal analysis, or introductory phrases like "Here's a thinking process", "I will analyze...", "Let's check...", or discussion of instructions.

Use this exact response structure:

1. Possible Condition:
- Mention only 1–3 likely possibilities using cautious words such as "may" or "could".

2. Explanation:
- Explain the main reason in 2–3 simple sentences.

3. What to Do:
- Give 3–5 practical steps.

4. When to See a Doctor:
- Mention important warning signs briefly.

Rules:
- Keep the entire response under 200–250 words.
- Do not repeat the user's question.
- Do not give a final diagnosis.
- Do not provide unnecessary medical details.
- Use simple, easy-to-understand language.

User Query:
{query}"""
        messages_content.append({"type": "text", "text": medical_prompt})

        if image and image.filename:
            image_content = await image.read()
            if image_content:
                # Validate image
                try:
                    img = Image.open(io.BytesIO(image_content))
                    img.verify()
                except Exception as e:
                    logger.error(f"Invalid image format: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

                encoded_image = base64.b64encode(image_content).decode("utf-8")
                mime_type = image.content_type or mimetypes.guess_type(image.filename or "")[0] or "image/jpeg"
                
                messages_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}"
                    },
                })

        messages = [
            {
                "role": "user",
                "content": messages_content,
            }
        ]

        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "qwen/qwen3.6-27b",
                "messages": messages,
                "max_tokens": 600,
                "temperature": 0.3,
                "reasoning_effort": "none",
                "reasoning_format": "hidden"
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Groq API error: {response.status_code} - {response.text}"
            )

        result = response.json()

        try:
            answer = result["choices"][0]["message"]["content"]
            if answer:
                # Clean up any leftover thinking process or preamble if present
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                elif "<think>" in answer:
                    answer = re.sub(r'^\s*<think>', '', answer, flags=re.IGNORECASE).strip()
                answer = re.sub(r"^(Here's a thinking process:?|Thinking process:?|Draft:?|I will analyze.*?|Let's check.*?)\s*", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
        except (KeyError, IndexError, TypeError):
            logger.error(f"Unexpected response format: {result}")
            raise HTTPException(status_code=500, detail="Unexpected API response format")

        if not answer or not str(answer).strip():
            answer = "No meaningful response was returned by the model."

        return JSONResponse(
            status_code=200,
            content={
                "output": answer,
                "model": "qwen/qwen3.6-27b"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
