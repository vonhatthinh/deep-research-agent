import os
from tavily import TavilyClient
from core.config import settings

# Initialize the Tavily client
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)


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
    

available_tools = {
    "tavily_web_search": tavily_web_search,
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
    }
]