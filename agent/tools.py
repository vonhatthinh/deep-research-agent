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
    print(f"INFO: Performing Tavily search for query: {query}")
    try:
        result = tavily_client.search(query, search_depth="advanced", max_results=5)
        # Format the results into a concise string for the LLM
        formatted_results = "\n\n".join(
            [f"Source: {res['url']}\nContent: {res['content']}" for res in result['results']]
        )
        return formatted_results
    except Exception as e:
        print(f"ERROR: Tavily search failed: {e}")
        return f"Error performing web search: {e}"


def selenium_get_web_content(url: str) -> str:
    """
    Uses Selenium to fetch the main content of a web page from the given URL.
    This function is designed to be called by the OpenAI Assistant.
    """
    print(f"INFO: Fetching web content via Selenium for URL: {url}")
    driver = None
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time

        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)  # 30 second timeout
        driver.get(url)

        # Wait until the <body> element is present (up to 10 seconds)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
def process_and_store_file(file_id: str) -> str:
    """
    Processes a file (PDF or text), extracts its content, and stores it in the vector database.
    Use this tool to make the content of a file available for semantic search.
    """
    print(f"INFO: Processing file for vector storage: {file_id}")
    try:
        # Get file metadata first
        file_info = client.files.retrieve(file_id)
        filename = file_info.filename.lower()
        file_size = file_info.bytes if hasattr(file_info, 'bytes') else None

        # Check file size limit (e.g., 50MB)
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

        if not text_content.strip():
            return f"No text content could be extracted from file {file_id}"

        # Add extracted text to the vector store
        add_text_to_vector_store(text_content)
        return f"Successfully processed and stored the content of file {file_id} in the vector database."

    except Exception as e:
        print(f"ERROR: Failed to process file {file_id}: {e}")
        return f"Error processing file: {e}"
        elif filename.endswith(('.txt', '.md', '.csv')):
            text_content = file_content.decode('utf-8')
        else:
            return f"Unsupported file type: {filename}. Only PDF, TXT, MD, and CSV are supported for text extraction."

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
    print(f"INFO: Adding text to vector store: '{text[:50]}...'")
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
    print(f"INFO: Querying knowledge base with: '{query}'")
    try:
        results = query_vector_store(query)
        return "\n\n".join(results) if results else "No relevant information found in the knowledge base."
    except Exception as e:
        print(f"ERROR: Failed to query knowledge base: {e}")
        return f"Error querying knowledge base: {e}"


def analyze_image_content(file_id: str) -> str:
    """
    Analyzes the content of an image file and returns a detailed text description.
    Use this tool when the user provides an image to understand its content.
    The agent must be provided with the file_id to use this tool.
    """
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
            model="gpt-4o",
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
            print(f"INFO: Image analysis successful. Description length: {len(description)}")
            add_text_to_vector_store(f"Image Description for {file_id}: {description}")
            return description
        else:
            return "Could not generate a description for the image."
    except Exception as e:
        print(f"ERROR: Image analysis for file {file_id} failed: {e}")
        return f"Error analyzing image: {e}"

available_tools = {
    "tavily_web_search": tavily_web_search,
    "selenium_get_web_content": selenium_get_web_content,
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
            "name": "selenium_get_web_content",
            "description": "Fetch the main content of a web page using Selenium. Use this for pages that require JavaScript rendering or dynamic content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page to fetch content from."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_and_store_file",
            "description": "Processes a file (PDF or text) and stores its content in the vector database for semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The ID of the file to process and store."
                    }
                },
                "required": ["file_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_text_to_store",
            "description": "Adds a given text string to the vector database for future reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to add to the vector store."
                    }
                },
                "required": ["text"]
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