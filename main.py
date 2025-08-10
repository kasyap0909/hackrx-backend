from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Set your Groq API key here (or use Render environment variables later)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

class QueryRequest(BaseModel):
    documents: str
    question: str = None

@app.post("/hackrx/run")
async def hackrx_run(req: QueryRequest):
    try:
        # Example document fetch
        doc_url = req.documents
        try:
            r = requests.get(doc_url, timeout=10)
            if r.status_code != 200:
                return {"answers": [f"Failed to fetch document: HTTP {r.status_code}"]}
            document_text = r.text[:3000]  # Limit size
        except Exception as e:
            return {"answers": [f"Document fetch failed: {e}"]}

        # Query Groq LLM
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for HackRx."},
                {"role": "user", "content": f"Context: {document_text}\n\nQuestion: {req.question or 'Summarize the document'}"}
            ]
        }
        groq_resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if groq_resp.status_code != 200:
            return {"answers": [f"GROQ API Error: {groq_resp.text}"]}

        answer = groq_resp.json()["choices"][0]["message"]["content"]
        return {"answers": [answer]}

    except Exception as e:
        return {"answers": [f"Server error: {e}"]}
