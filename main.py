import os
import asyncio
import json
import pypdf
import docx
import csv
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# import python_multipart_form  # This is needed for Form/File to work

from agent.multi_agent import MultiAgent
from core.config import settings
from openai import OpenAI, APIStatusError
from google import genai
from google.genai import types

app = FastAPI(
    title="Deep Research Assistant API",
    description="An API for conducting deep research with an AI agent."
)

gg_client = genai.Client(api_key=settings.GEMINI_API_KEY)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # Add Streamlit's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static File Serving ---
# Get the absolute path to the 'ui/public' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.join(current_dir, 'ui', 'public')

# Ensure the directory exists
os.makedirs(public_dir, exist_ok=True)

# Mount the directory to serve files under the "/files" path
app.mount("/files", StaticFiles(directory=public_dir), name="files")


multi_agent = MultiAgent()

# Helper functions for parsing files
def parse_pdf(content: bytes) -> str:
    text = ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        return f"[Error parsing PDF: {e}]"
    return text

def parse_docx(content: bytes) -> str:
    text = ""
    try:
        doc = docx.Document(io.BytesIO(content))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        return f"[Error parsing DOCX: {e}]"
    return text

def parse_csv(content: bytes) -> str:
    text = ""
    try:
        # Decode bytes to string for csv reader
        content_str = content.decode('utf-8')
        # Use a string stream for the csv reader
        reader = csv.reader(io.StringIO(content_str))
        for row in reader:
            text += ", ".join(row) + "\n"
    except Exception as e:
        return f"[Error parsing CSV: {e}]"
    return text


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


@app.delete("/files/{filename}", summary="Delete a Generated File")
async def delete_file(filename: str):
    """
    Deletes a file from the public directory.
    """
    try:
        file_path = os.path.join(public_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return JSONResponse(status_code=200, content={"message": f"File {filename} deleted successfully."})
        else:
            raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")


async def cleanup_local_file(path: str):
    """Deletes the local file after it has been served."""
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted temporary local file: {path}")
    except Exception as e:
        print(f"Error deleting local file {path}: {e}")


@app.post("/query", summary="Start a Research Task")
async def query(query: str = Form(...), files: list[UploadFile] = File(None), session_id: str = Form(...)):
    """
    Accepts a user query, optional files, and a session_id, then streams the agent's research process.
    This version handles multiple files: it parses content from all supported files (images, PDFs, DOCX, CSV)
    and appends it to the query for the agent.
    """
    updated_query = query
    file_id = None  # file_id is no longer used as we inject content directly into the query.

    if files:
        for file in files:
            try:
                file_content = await file.read()
                file_mime_type = file.content_type
                file_name = file.filename or "unknown_file"
                
                extracted_text = ""

                if file_mime_type and file_mime_type.startswith('image/'):
                    # It's an image file. Use Gemini to describe it.
                    response = gg_client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=[
                            types.Part.from_bytes(data=file_content, mime_type=file_mime_type),
                            'Analyze the content of this image and provide a detailed description. This description will be used as context for a research query.'
                        ]
                    )
                    extracted_text = response.text
                    print(f"Processed image file: {file_name}")
                
                elif file_mime_type == 'application/pdf':
                    extracted_text = parse_pdf(file_content)
                    print(f"Parsed PDF file: {file_name}")

                elif file_mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or (file_name and file_name.endswith('.docx')):
                     extracted_text = parse_docx(file_content)
                     print(f"Parsed DOCX file: {file_name}")

                elif file_mime_type == 'text/csv' or (file_name and file_name.endswith('.csv')):
                    extracted_text = parse_csv(file_content)
                    print(f"Parsed CSV file: {file_name}")
                
                else:
                    # If the file type is not supported for parsing, we inform and skip it.
                    print(f"Skipping unsupported file type: {file_name} ({file_mime_type})")
                    updated_query += f"\n\n[Skipped unsupported file: {file_name}]"
                    continue

                if extracted_text:
                    updated_query += f"\n\n[Content from file: {file_name}]:\n{extracted_text}"

            except Exception as e:
                # Inform about the failure for a specific file and continue
                error_message = f"Failed to process file {file.filename}: {str(e)}"
                print(error_message)
                updated_query += f"\n\n[Error processing file {file.filename or 'unknown'}: {str(e)}]"
    
    return StreamingResponse(
        multi_agent.run_multi_agent_research(
            query=updated_query, 
            file_id=file_id, 
            session_id=session_id
        ),
        media_type="text/event-stream"
    )


@app.get("/files/{file_id}", summary="Retrieve a File from OpenAI")
async def get_file(file_id: str, background_tasks: BackgroundTasks):
    """
    Retrieves a file from OpenAI's servers by its file_id.
    It downloads the file, saves it temporarily, serves it, and then cleans up the local copy.
    This is used for files generated by assistants (e.g., code_interpreter) that are not local.
    """
    try:
        # Step 1: Get file metadata to find the filename
        print(f"Retrieving metadata for file_id: {file_id}")
        file_info = client.files.retrieve(file_id)
        filename = file_info.filename
        print(f"Filename from metadata: {filename}")

        local_file_path = os.path.join(public_dir, filename)

        # Step 2: Download the file content from OpenAI
        print(f"Downloading file content for {file_id}")
        file_content_response = client.files.content(file_id)
        file_bytes = file_content_response.read()

        # Step 3: Save the file locally
        with open(local_file_path, "wb") as f:
            f.write(file_bytes)
        print(f"File {file_id} saved to {local_file_path}")

        # Step 4: Add a background task to delete the local file after serving.
        # We do NOT delete the OpenAI file, as it might be needed for future reference.
        background_tasks.add_task(cleanup_local_file, path=local_file_path)

        # Step 5: Serve the file
        content_type = "application/octet-stream"
        if filename.lower().endswith(".png"):
            content_type = "image/png"
        elif filename.lower().endswith((".jpg", ".jpeg")):
            content_type = "image/jpeg"
        elif filename.lower().endswith(".pdf"):
            content_type = "application/pdf"
        elif filename.lower().endswith(".txt"):
            content_type = "text/plain"

        return FileResponse(local_file_path, media_type=content_type)

    except APIStatusError as e:
        # Handle OpenAI specific errors, e.g., file not found
        raise HTTPException(status_code=e.status_code, detail=e.response.json())
    except Exception as e:
        # Handle other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching file {file_id}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug")