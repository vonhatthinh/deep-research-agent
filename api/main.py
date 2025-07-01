import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# import python_multipart_form  # This is needed for Form/File to work

from agent.agent import run_agent_research
from core.config import settings
from openai import OpenAI

app = FastAPI(
    title="Deep Research Assistant API",
    description="An API for conducting deep research with an AI agent."
)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status", summary="Health Check")
async def status():
    """Provides the operational status of the API."""
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "message": "Deep Research Assistant API is running."}
    )


@app.post("/query", summary="Start a Research Task")
async def query(query: str = Form(...), file: UploadFile = File(None)):
    """
    Accepts a user query and an optional file, then streams the agent's research process.
    """
    file_id = None
    if file:
        try:
            # Upload the file to OpenAI's servers for the assistant to use
            uploaded_file = client.files.create(file=file.file, purpose='assistants')
            file_id = uploaded_file.id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")

    # Use a streaming response to send server-sent events (SSE)
    return StreamingResponse(
        run_agent_research(query, file_id),
        media_type="text/event-stream"
    )


app.get("/files/{file_id}", summary="Retrieve a Generated File")
async def get_file(file_id: str):
    """
    Downloads a file (e.g., a generated chart) from OpenAI and returns it.
    """
    try:
        file_content = client.files.content(file_id)
        # The content is an APIResponse object, get the raw bytes
        file_bytes = file_content.read()
        return StreamingResponse(iter([file_bytes]), media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found or could not be retrieved: {e}")
