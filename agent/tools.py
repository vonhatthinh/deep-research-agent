import os
import base64
from tavily import TavilyClient
from core.config import settings
from openai import OpenAI
import PyPDF2
from io import BytesIO

from database.vector_store import add_text as add_text_to_vector_store
from database.vector_store import query_store as query_vector_store

# Initialize clients
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
client = OpenAI(api_key=settings.OPENAI_API_KEY)


def tavily_web_search(query: str) -> str:
    """
    Performs a web search using the Tavily API.
    This function is designed to be called by the OpenAI Assistant.
    """
    print(f"--- Running Tool: tavily_web_search ---")
    print(f"INFO: Performing Tavily search for query: {query}")
    try:
        result = tavily_client.search(query, search_depth="advanced", max_results=5)
        # Format the results into a concise string for the LLM
        formatted_results = "\n\n".join(
            [f"Source: {res['url']}\nContent: {res['content']}" for res in result['results']]
        )
        print(f"INFO: Tavily search results:\n{formatted_results}...")
        return formatted_results
    except Exception as e:
        print(f"ERROR: Tavily search failed: {e}")
        return f"Error performing web search: {e}"


def process_and_store_file(file_id: str) -> str:
    try:
        file_info = client.files.retrieve(file_id)
        filename = file_info.filename.lower()
        file_size = file_info.bytes if hasattr(file_info, 'bytes') else None
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        if file_size and file_size > MAX_FILE_SIZE:
            return f"File too large ({file_size} bytes). Maximum supported size is {MAX_FILE_SIZE} bytes."
        file_content_response = client.files.content(file_id)
        file_content = file_content_response.read()
        text_content = ""
        if filename.endswith('.pdf'):
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                if pdf_reader.is_encrypted:
                    return f"Cannot process encrypted PDF file: {filename}"
                for page in pdf_reader.pages:
                    text_content += page.extract_text() or ""
            except Exception as pdf_error:
                return f"Failed to process PDF file: {str(pdf_error)}"
        elif filename.endswith(('.txt', '.md', '.csv')):
            text_content = file_content.decode('utf-8')
        else:
            return f"Unsupported file type: {filename}. Only PDF, TXT, MD, and CSV are supported for text extraction."

        print(f"INFO: Extracted content from {filename}:\n--- START OF CONTENT ---\n{text_content[:2000]}...\n--- END OF CONTENT ---")
        # Add extracted text to the vector store
        add_text_to_vector_store(text_content)
        return f"Successfully processed and stored the content of file {file_id} in the vector database."

    except Exception as e:
        print(f"ERROR: Failed to process file {file_id}: {e}")
        return f"Error processing file: {e}"


def add_text_to_store(text: str) -> str:
    """
    Adds a given text string to the vector database for future reference.
    Use this to store important pieces of information, like the initial user prompt or key findings.
    """
    print(f"--- Running Tool: add_text_to_store ---")
    print(f"INFO: Adding text to vector store:\n--- START OF TEXT ---\n{text}\n--- END OF TEXT ---")
    try:
        add_text_to_vector_store(text)
        return "Text successfully added to the vector store."
    except Exception as e:
        print(f"ERROR: Failed to add text to vector store: {e}")
        return f"Error adding text to vector store: {e}"


def query_knowledge_base(query: str) -> str:
    """
    Queries the vector database to find information relevant to the query.
    Use this to retrieve contextually similar information that has been stored from files or text.
    """
    print(f"--- Running Tool: query_knowledge_base ---")
    print(f"INFO: Querying knowledge base with: '{query}'")
    try:
        results = query_vector_store(query)
        print(f"INFO: Found {len(results)} results from knowledge base.")
        if not results:
            return "No relevant information found in the knowledge base."
        
        full_results = "\n\n".join(results)
        print(f"INFO: Knowledge base results:\n{full_results[:2000]}...")
        return full_results
    except Exception as e:
        print(f"ERROR: Failed to query knowledge base: {e}")
        return f"Error querying knowledge base: {e}"


def analyze_image_content(file_id: str) -> str:
    """
    Analyzes the content of an image file and returns a detailed text description.
    Use this tool when the user provides an image to understand its content.
    The agent must be provided with the file_id to use this tool.
    """
    print(f"--- Running Tool: analyze_image_content ---")
    print(f"INFO: Analyzing image content for file_id: {file_id}")
    try:
        # Get file metadata
        file_info = client.files.retrieve(file_id)
        filename = file_info.filename.lower()
        
        # Check if file is an image
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in image_extensions:
            return f"File {filename} is not a supported image format. Supported formats: {', '.join(image_extensions)}"
        
        # Map extensions to MIME types
        mime_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.bmp': 'image/bmp', '.webp': 'image/webp'
        }
        mime_type = mime_types.get(file_ext, 'image/jpeg')
        
        # Retrieve the file content
        file_content_response = client.files.content(file_id)
        file_content = file_content_response.read()
        
        # Check file size (e.g., 20MB limit for images)
        MAX_IMAGE_SIZE = 20 * 1024 * 1024
        if len(file_content) > MAX_IMAGE_SIZE:
            return f"Image file too large ({len(file_content)} bytes). Maximum supported size is {MAX_IMAGE_SIZE} bytes."

        # Encode the image in base64
        base64_image = base64.b64encode(file_content).decode('utf-8')

        # Call Chat Completions API with the image
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image in detail. Describe the key elements, objects, people, setting, and any text present. This description will be used for research, so be as comprehensive as possible."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024,
        )
        description = response.choices[0].message.content
        if description:
            print(f"INFO: Image analysis successful. Description:\n--- START OF DESCRIPTION ---\n{description}\n--- END OF DESCRIPTION ---")
            # Also add the description to the vector store
            add_text_to_vector_store(f"Image Description for {file_id}: {description}")
            return description
        else:
            return "Could not generate a description for the image."
    except Exception as e:
        print(f"ERROR: Image analysis for file {file_id} failed: {e}")
        return f"Error analyzing image: {e}"


available_tools = {
    "tavily_web_search": tavily_web_search,
    "process_and_store_file": process_and_store_file,
    "add_text_to_store": add_text_to_store,
    "query_knowledge_base": query_knowledge_base,
    "analyze_image_content": analyze_image_content,
}


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "tavily_web_search",
            "description": "Get information from the web using the Tavily search engine. Use this for recent events or information not found in uploaded files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find information on the web."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "Queries the vector database to find information relevant to the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for in the knowledge base."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image_content",
            "description": "Analyzes an image file and returns a detailed text description of its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The ID of the image file to analyze."
                    }
                },
                "required": ["file_id"]
            }
        }
    }
]