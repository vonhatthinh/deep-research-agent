import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# import python_multipart_form  # This is needed for Form/File to work

from agent.multi_agent import run_multi_agent_research
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
    allow_origins=["http://localhost:3000"],  # Add your actual frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast(f"Client #{client_id} left the chat")


@app.get("/", summary="Root Endpoint")
async def root():
    """Provides a welcome message and a link to the API documentation."""
    return JSONResponse(
        content={
            "message": "this is the Deep Research Assistant API",
            "docs_url": "/docs"
        }
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
        run_multi_agent_research(query, file_id),
        media_type="text/event-stream"
    )


@app.get("/files/{file_id}", summary="Retrieve a Generated File")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug")