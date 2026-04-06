from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import requests
import io
from PIL import Image
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
        
        medical_prompt = f"""
                        You are a helpful and careful medical assistant AI.

                        Analyze the given image (if provided) and user query carefully.

                        User Query:
                        {query}

                        Provide a concise, structured response in the format below (limit to 4–6 lines total):

                        1. Possible Condition:
                        - Briefly mention likely causes (use "may be" or "could be")

                        2. Explanation:
                        - Short and simple reason

                        3. Management:
                        - 1–2 practical steps

                        4. Diet:
                        - Key foods to eat/avoid (1 line)

                        5. Precautions:
                        - 1 important tip

                        6. Doctor Visit:
                        - When to seek help (1 line)

                        Rules:
                        - No final diagnosis
                        - Keep response short, clear, and practical
                        """
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
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.6
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
        except (KeyError, IndexError, TypeError):
            logger.error(f"Unexpected response format: {result}")
            raise HTTPException(status_code=500, detail="Unexpected API response format")

        if not answer or not str(answer).strip():
            answer = "No meaningful response was returned by the model."

        return JSONResponse(
            status_code=200,
            content={
                "output": answer,
                "model": "meta-llama/llama-4-scout-17b-16e-instruct"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)