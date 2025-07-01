import os
import json
import time
import asyncio
from openai import OpenAI
from typing import AsyncGenerator

from core.config import settings
from agent.tools import tools_schema, available_tools
from agent.schema import ResearchReport, ThinkingProcess
from typing import Optional

client = OpenAI(api_key=settings.OPENAI_API_KEY)
ASSISTANT_ID_FILE = "assistant_id.txt"

def get_assistant_id():
    """Retrieves the assistant ID from a local file, or returns None if not found."""
    if os.path.exists(ASSISTANT_ID_FILE):
        with open(ASSISTANT_ID_FILE, 'r') as f:
            return f.read().strip()
    return None


def save_assistant_id(assistant_id):
    """Saves the assistant ID to a local file."""
    with open(ASSISTANT_ID_FILE, 'w') as f:
        f.write(assistant_id)



def create_or_get_assistant():
    """
    Creates a new OpenAI Assistant if one doesn't exist,
    otherwise retrieves the existing one.
    """
    assistant_id = get_assistant_id()
    if assistant_id:
        try:
            assistant = client.beta.assistants.retrieve(assistant_id)
            print(f"INFO: Retrieved existing assistant with ID: {assistant.id}")
            return assistant
        except Exception as e:
            print(f"WARN: Could not retrieve assistant {assistant_id}. Creating a new one. Error: {e}")

    # Define the assistant's instructions and configuration
    assistant_instructions = """
    You are a world-class AI Deep Research Assistant. Your goal is to conduct thorough, unbiased research on a given topic, synthesize the findings, and present them in a structured, professional report.

    **Your Process:**
    1.  **Deconstruct the Query:** Fully understand the user's request, including any nuances from text, images, or provided files.
    2.  **Plan Your Research:** Formulate a plan. Decide which tools to use (web search, file analysis, code execution for charts).
    3.  **Execute Research:**
        - Use the `tavily_web_search` tool for up-to-date information from the web.
        - Use the `file_search` tool to analyze user-provided documents.
        - Use the `code_interpreter` tool to analyze data (e.g., CSVs) and generate visualizations (charts, graphs). When creating a visual, save it as a file and make a note of its `file-id`.
    4.  **Synthesize and Structure:** Combine all gathered information.
    5.  **Generate Final Output:** Produce TWO JSON objects at the very end of your response, and nothing else.
        a.  A `ThinkingProcess` object outlining your methodology.
        b.  A `ResearchReport` object containing the comprehensive findings. Ensure all fields of the report are populated. Include `file-ids` in the `visuals` field for any charts you created.
    """

    assistant = client.beta.assistants.create(
        name="Deep Research Assistant",
        instructions=assistant_instructions,
        model="gpt-4o", # Use the latest powerful model
        tools=tools_schema + [{"type": "code_interpreter"}, {"type": "file_search"}]
    )
    save_assistant_id(assistant.id)
    print(f"INFO: Created new assistant with ID: {assistant.id}")
    return assistant

# --- Agent Execution ---
async def run_agent_research(query: str, file_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Runs the research agent and streams back the thinking process and final report.
    """
    assistant = create_or_get_assistant()
    thread = client.beta.threads.create()

    yield f"event: thinking\ndata: Creating a research plan...\n\n"

    # Add the user's message to the thread
    attachments = []
    if file_id:
        attachments.append({"file_id": file_id, "tools": [{"type": "file_search"}]})

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query,
        attachments=attachments
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Poll the run status and handle tool calls
    while run.status in ['queued', 'in_progress', 'requires_action']:
        if run.status == 'requires_action':
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                yield f"event: thinking\ndata: Using tool `{tool_name}` with arguments: {tool_args}\n\n"

                if tool_name in available_tools:
                    output = available_tools[tool_name](**tool_args)
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": output,
                    })

            # Submit tool outputs back to the run
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
        
        # Give a heartbeat to the client
        yield f"event: thinking\ndata: The agent is currently processing...\n\n"
        await asyncio.sleep(2) # Use asyncio.sleep in async functions
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # Process the final response
    if run.status == 'completed':
        yield f"event: thinking\ndata: Finalizing the report...\n\n"
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        # The response is typically the last message from the assistant
        assistant_message = messages.data[0].content[0].text.value
        
        # The agent should output two JSON objects. We need to parse them.
        try:
            # Find the start of the first JSON and the end of the last one
            start_index = assistant_message.find('{')
            end_index = assistant_message.rfind('}') + 1
            json_str = assistant_message[start_index:end_index].replace("```json", "").replace("```", "").strip()
            
            # The agent might output two separate JSON objects. Let's try to parse them as a list.
            # A simple way is to wrap them in an array.
            json_str_as_array = f"[{json_str.replace('}{', '},{')}]"
            
            parsed_json = json.loads(json_str_as_array)
            
            final_data = {"thinking_process": parsed_json[0], "report": parsed_json[1]}

            yield f"event: report\ndata: {json.dumps(final_data)}\n\n"
        except Exception as e:
            print(f"ERROR: Failed to parse final JSON from agent response: {e}")
            yield f"event: error\ndata: Could not parse the final report from the agent's response. Raw response: {assistant_message}\n\n"

    else:
        yield f"event: error\ndata: The research task failed with status: {run.status}\n\n"