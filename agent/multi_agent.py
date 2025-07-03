import asyncio
import json
import typing
import requests
import io
import os
import uuid
from openai import OpenAI
from openai.types.beta.assistant_tool_param import AssistantToolParam
from typing import AsyncGenerator, Optional

from core.config import settings
from agent.tools import tools_schema, available_tools
from database.vector_store import clear_store
from agent.base_agent import BaseAgent
from google import genai
from google.genai import types

gg_client = genai.Client(api_key=settings.GEMINI_API_KEY) # type: ignore

client = OpenAI(api_key=settings.OPENAI_API_KEY)
session_threads = {} # Maps session_id to thread_id

class MultiAgent(BaseAgent):
    """
    Multi-agent system for conducting deep research.
    This class orchestrates the interaction between multiple agents to analyze a query,
    plan tasks, conduct research, generate visuals, and evaluate the final report.
    """

    def __init__(self):
        super().__init__()

    async def run(self, *args, **kwargs):
        """Execute the agent's main logic."""
        pass

    async def run_agent(self, assistant_id: str, thread_id: str, user_input: str, file_id: Optional[str] = None) -> dict:
        attachments = []
        content_to_send = user_input
        if file_id:
            try:
                # Since image handling is now done in the API endpoint before this function is called,
                # any file_id passed here is for a document (e.g., PDF, DOCX) uploaded to OpenAI.
                # We just need to attach it to the message for the assistant's tools.
                attachments.append({"file_id": file_id, "tools": [{"type": "file_search"}, {"type": "code_interpreter"}]})
            except Exception as e:
                print(f"Could not attach file_id {file_id}. Error: {e}")
                # If there's an error, we proceed without the file to avoid crashing the run.
                pass
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content_to_send,
            attachments=attachments
        )

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        while True:
            while run.status in ['queued', 'in_progress', 'cancelling']:
                await asyncio.sleep(1)
                run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

            if run.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)
                message = messages.data[0]
                response_text = ""
                file_ids = []

                if message.role == 'assistant' and message.content:
                    for content_part in message.content:
                        if content_part.type == "text":
                            response_text += content_part.text.value
                        elif content_part.type == "image_file":
                            file_ids.append(content_part.image_file.file_id)
                
                return {"text": response_text, "file_ids": file_ids}

            elif run.status == 'requires_action' and run.required_action:
                tool_outputs = await self._handle_tool_calls(run.required_action, file_id)
                
                try:
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                except Exception as e:
                    return {"text": f"Error submitting tool outputs: {e}", "file_ids": []}
                continue # Loop back to check run status
            
            return {"text": f"Run ended with status: {run.status}", "file_ids": []}

    async def _handle_tool_calls(self, required_action, file_id: Optional[str]) -> list:
        tool_outputs = []
        for tool_call in required_action.submit_tool_outputs.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            
            output = ""
            if function_to_call:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Special handling for file_id injection
                    if function_name in ["process_and_store_file", "analyze_image_content"] and 'file_id' not in function_args:
                        if file_id:
                            function_args['file_id'] = file_id
                        else:
                            print(f"Skipping tool call {function_name}: file_id is required but not provided.")
                            tool_outputs.append({"tool_call_id": tool_call.id, "output": f"Error: Tool '{function_name}' requires a file, but no file was uploaded."})
                            continue

                    if asyncio.iscoroutinefunction(function_to_call):
                        output = await function_to_call(**function_args)
                    else:
                        output = function_to_call(**function_args)
                    
                    # If the tool is tavily_web_search and it returns an image, pass the file_id to the next agent
                    if function_name == "tavily_web_search" and isinstance(output, dict) and output.get("image_file_id"):
                        # This is a simplification. In a real scenario, you'd need a more robust way
                        # to manage state or pass this file_id to the visualizer agent.
                        # For now, we can try to pass it via the thread or a shared state.
                        # Let's assume the researcher's output will include this info.
                        print(f"Web search returned an image. File ID: {output['image_file_id']}")


                    output_str = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
                    tool_outputs.append({"tool_call_id": tool_call.id, "output": output_str})

                except Exception as e:
                    print(f"Error calling tool {function_name}: {e}")
                    tool_outputs.append({"tool_call_id": tool_call.id, "output": f"Error: {e}"})
            else:
                tool_outputs.append({"tool_call_id": tool_call.id, "output": f"Error: Tool '{function_name}' not found."})
        
        return tool_outputs

    def _create_assistant(self, name: str, instructions: str, model: str = "gpt-4o", tools: list = [], response_format: Optional[dict] = None):
        params = {
            "name": name,
            "instructions": instructions,
            "model": model,
            "tools": tools
        }
        if response_format:
            params["response_format"] = response_format
        return client.beta.assistants.create(**params)

    async def _run_analyzer_agent(self, thread_id: str, query: str, file_id: Optional[str]) -> dict:
        """Runs the query analyzer agent to determine intent and analyze the query."""
        analyzer_instructions = """
You are an expert at analyzing user input and conversation history. Your assistant is part of a thread, so you have access to the previous messages.
Determine if the user's current request is a simple chat message or requires in-depth research.

**Instructions:**
1.  Review the conversation history in the thread.
2.  Analyze the current user query.
3.  Respond with a JSON object containing the 'intent' ('chat' or 'research') and the 'analyzed_query'. The 'analyzed_query' should be a refined version of the user's query based on the context of the conversation.
"""
        analyzer_assistant = self._create_assistant(
            name="Query Analyzer Agent",
            instructions=analyzer_instructions,
            response_format={"type": "json_object"}
        )
        response = await self.run_agent(analyzer_assistant.id, thread_id, query, file_id)
        analyzed_query_json = response.get("text", "{}")
        print(f"Analyzed Query JSON: {analyzed_query_json}")
        try:
            return json.loads(analyzed_query_json)
        except json.JSONDecodeError:
            print(f"Warning: Query Analyzer returned invalid JSON: {analyzed_query_json}")
            return {"intent": "research", "analyzed_query": query, "error": "Invalid JSON response"}

    async def _run_chat_agent(self, thread_id: str, query: str) -> str:
        """Runs the chat agent for simple queries."""
        chat_assistant = self._create_assistant(
            name="Chat Agent",
            instructions="You are a helpful and friendly chatbot. You have access to the conversation history in this thread. Respond to the user's query directly, using the context from the conversation if relevant.",
        )
        response = await self.run_agent(chat_assistant.id, thread_id, query)
        return response.get("text", "Sorry, I could not process your request.")

    async def _run_planner_agent(self, thread_id: str, query: str, analyzed_query_text: str, file_id: Optional[str]) -> str:
        """Runs the task planner agent to create a research plan."""
        planner_instructions = "You are a meticulous planner. Your first step is always to store the original user query in the vector database using the `add_text_to_store` tool."
        if file_id:
            planner_instructions += " If a file is provided, your second step is to process it using either `process_and_store_file` for documents or `analyze_image_content` for images."
        planner_instructions += " After that, create a plan to query the knowledge base and the web to answer the user's request. Output the plan as a clear, plain text list of tasks."

        planner_assistant = self._create_assistant(
            name="Task Planner Agent",
            instructions=planner_instructions,
            tools=[typing.cast(AssistantToolParam, tool) for tool in tools_schema if tool['function']['name'] in ['add_text_to_store', 'process_and_store_file', 'analyze_image_content']]
        )
        planner_input = f"Original User Query: {query}\nAnalyzed Query: {analyzed_query_text}"
        if file_id:
            planner_input += f"\nFile ID: {file_id}"
        response = await self.run_agent(planner_assistant.id, thread_id, planner_input, file_id)
        task_list = response.get("text", "No task list was generated.")
        print(f"Task List: {task_list}")
        return task_list

    async def _run_researcher_agent(self, thread_id: str, task_list: str, file_id: Optional[str]) -> str:
        """Runs the researcher agent to execute tasks and gather information."""
        researcher_assistant = self._create_assistant(
            name="Researcher Agent",
            instructions="You are a diligent researcher. Execute the given list of tasks to gather information. Use the `tavily_web_search` and `query_knowledge_base` tools to find relevant data. When searching the web, consider if an image would be beneficial for the final report and set `include_images` to true if so. Synthesize the findings into a comprehensive research report.",
            tools=[typing.cast(AssistantToolParam, tool) for tool in tools_schema]
        )
        response = await self.run_agent(researcher_assistant.id, thread_id, f"Research tasks: {task_list}", file_id)
        return response.get("text", "No research report was generated.")

    async def _run_visualizer_agent(self, thread_id: str, research_report: str, file_id: Optional[str]) -> dict:
        """Runs the visualizer agent to generate visuals for the report."""
        
        # Step 1: Check if the research report (or tool outputs) contains an image file_id
        # This is a simple check; a more robust solution might involve parsing tool outputs explicitly.
        image_file_id_from_research = None
        if "image_file_id" in research_report:
             # This is a placeholder for logic to extract the file_id from the research report
            pass

        # If an image was found during research, use it. Otherwise, generate a new one.
        if image_file_id_from_research:
            print(f"Visualizer is using image from research: {image_file_id_from_research}")
            return {
                "summary": "Using a relevant image found during the research phase.",
                "file_ids": [image_file_id_from_research]
            }

        # Step 2: Create a prompt-generator assistant to decide if a new image is needed.
        prompt_generator_instructions = """
You are an expert in data visualization and creative prompting. Your task is to analyze a research report and determine if a visual representation (like a diagram, chart, or illustrative image) would enhance it.

**Instructions:**
1.  Read the provided research report.
2.  Decide if a visual is appropriate. The visual should be illustrative of the key themes, not a complex data chart that requires precise data plotting. Think about a cover image or a diagram that captures the essence of the report.
3.  If a visual is not needed, respond with a JSON object: `{"generate": false, "summary": "A brief explanation of why no visual is needed."}`.
4.  If a visual is needed, create a concise, descriptive prompt for an image generation model like DALL-E. The prompt should be in English and be highly descriptive to generate a visually appealing and relevant image.
5.  Respond with a JSON object: `{"generate": true, "prompt": "<your_dalle_prompt>", "summary": "<A description of the visual you are proposing>"}`.

Generate ONLY the JSON object as your response.
"""
        
        prompt_generator_assistant = self._create_assistant(
            name="Visual Prompt Generator",
            instructions=prompt_generator_instructions,
            response_format={"type": "json_object"}
        )

        # Run the prompt generator agent
        response = await self.run_agent(prompt_generator_assistant.id, thread_id, f"Research Report:\n\n{research_report}")
        decision_json = response.get("text", "{}")

        try:
            decision = json.loads(decision_json)
        except json.JSONDecodeError:
            print(f"Warning: Visualizer prompt generator returned invalid JSON: {decision_json}")
            return {"summary": "Failed to decide on visual generation due to invalid JSON.", "file_ids": []}

        if not decision.get("generate"):
            summary = decision.get("summary", "No visuals were deemed necessary for this report.")
            print(f"Visualizer decided not to generate an image. Reason: {summary}")
            return {"summary": summary, "file_ids": []}

        # Step 2: Generate the image using DALL-E if generate is true
        dalle_prompt = decision.get("prompt")
        summary = decision.get("summary", "Generating a visual for the report.")
        if not dalle_prompt:
            return {"summary": "Visual generation was requested, but no prompt was provided.", "file_ids": []}

        print(f"Generating image with DALL-E prompt: {dalle_prompt}")

        try:
            # Generate the image
            image_response = client.images.generate(
                model="dall-e-3",
                prompt=dalle_prompt,
                n=1,
                size="1024x1024",
                response_format="url"
            )

            if not image_response.data or not image_response.data[0].url:
                raise Exception("DALL-E did not return an image URL.")

            image_url = image_response.data[0].url

            # Download the image
            image_download_response = requests.get(image_url)
            image_download_response.raise_for_status()
            image_bytes = image_download_response.content

            # Save the image to a local file in the ui/public directory
            file_extension = ".png" 
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            
            # Correctly construct the path relative to the project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            public_dir = os.path.join(project_root, 'ui', 'public')
            os.makedirs(public_dir, exist_ok=True)
            
            local_image_path = os.path.join(public_dir, unique_filename)
            
            with open(local_image_path, "wb") as f:
                f.write(image_bytes)

            # The "file_id" will now be the local URL
            local_file_url = f"/files/{unique_filename}"
            
            print(f"Successfully generated and saved image locally. URL: {local_file_url}")
            return {"summary": summary, "file_ids": [local_file_url]}

        except Exception as e:
            error_message = f"An error occurred during image generation or saving: {e}"
            print(error_message)
            return {"summary": error_message, "file_ids": []}

    async def _run_evaluator_agent(self, thread_id: str, task_list: str, research_report: str, visuals_summary: dict, query: str, file_id: Optional[str]) -> dict:
        """Runs the evaluator agent to compile the final structured report."""
        
        visuals_summary_text = visuals_summary.get('summary', 'No visuals summary provided.')
        visual_file_ids = visuals_summary.get('file_ids', [])

        evaluator_instructions = f"""
        You are a professional report writer. Your task is to compile a final, comprehensive research report based on the provided information.

        The final report must be a JSON object with the following structure:
        {{
        "executive_summary": "A comprehensive summary of the research. It should contain a brief overview and a 'Detailed Content' section with the main findings.",
        "key_findings": [
            "A concise insight with citations if applicable.",
            "Another concise insight."
        ],
        "visuals": [
            {{
            "title": "Title of the visual",
            "description": "Description of the visual.",
            "file_id": "The file ID if available, otherwise null."
            }}
        ],
        "conclusion": "A summary of key takeaways and suggested next steps.",
        "references": [
            "List of sources used in the research."
        ]
        }}

        **Information to use:**
        - Original User Query: {query}
        - Research Plan: {task_list}
        - Research Report: {research_report}
        - Visuals Summary: {visuals_summary_text}

        **Instructions:**
        1.  **Executive Summary (Required):** This section must be comprehensive.
            - Start with a brief overview of the research objective and outcome.
            - Then, add a markdown heading `### Detailed Content`.
            - Under this heading, provide a well-structured summary of the research plan (`Research Plan`) and the detailed findings from the `Research Report`. For example, if the user asked for an itinerary, the full itinerary should be formatted nicely here.
        2.  **Key Findings (Optional):** Extract any secondary, important, concise insights from the research report. Do not repeat what is in the executive summary. If the report includes citations, include them. If there are no specific key findings, this can be an empty list.
        3.  **Visuals (Optional):** Summarize the visuals from the 'Visuals Summary'. If the summary indicates no visuals were created, this can be an empty list. Each visual should be an object with a title, description, and file_id if provided.
        4.  **Conclusion (Required):** Provide a summary of the key takeaways from the research and suggest potential next steps for the user.
        5.  **References (Required):** List all the sources, websites, or documents mentioned in the research report. If no sources are mentioned, this can be an empty list.

        Generate ONLY the JSON object as your response.
        """
        evaluator_assistant = self._create_assistant(
            name="Final Report Generator Agent",
            instructions=evaluator_instructions,
            response_format={"type": "json_object"}
        )
        
        # The input to the agent is now just a trigger, as the instructions have all the context.
        response = await self.run_agent(evaluator_assistant.id, thread_id, "Generate the final report based on the provided context.", file_id)
        final_report_json = response.get("text", "{}")
        
        print(f"Final Report JSON from Evaluator: {final_report_json}")
        try:
            final_report_data = json.loads(final_report_json)
            final_report_data['detailed_report'] = research_report
            # Add file_ids to visuals section if not already present
            if visual_file_ids and 'visuals' in final_report_data:
                for i, file_id in enumerate(visual_file_ids):
                    if i < len(final_report_data['visuals']):
                        if not final_report_data['visuals'][i].get('file_id'):
                            final_report_data['visuals'][i]['file_id'] = file_id
                    else:
                        # If the agent didn't create a visual entry, create one.
                        final_report_data['visuals'].append({
                            "title": "Generated Visual",
                            "description": "A visual generated by the visualizer agent.",
                            "file_id": file_id
                        })
            # Ensure file_id is a URL
            if 'visuals' in final_report_data:
                for visual in final_report_data['visuals']:
                    if 'file_id' in visual and visual['file_id'] and not visual['file_id'].startswith('/files/'):
                        # This logic might need to be adjusted based on how file_ids are handled elsewhere
                        pass

            return final_report_data
        except json.JSONDecodeError:
            print(f"Warning: Evaluator returned invalid JSON: {final_report_json}")
            # Fallback to a simple structure if JSON parsing fails
            return {
                "executive_summary": "Failed to generate structured report.",
                "detailed_report": research_report,
                "key_findings": [],
                "visuals": [],
                "conclusion": "The evaluator agent failed to produce a valid JSON report. The raw output is provided below.",
                "references": [],
                "raw_evaluator_output": final_report_json
            }

    async def run_multi_agent_research(self, 
                                       query: str, 
                                       file_id: Optional[str] = None, 
                                       session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Runs the multi-agent research process and streams back the results.
        """
        if session_id and session_id in session_threads:
            thread_id = session_threads[session_id]
        else:
            thread = client.beta.threads.create()
            thread_id = thread.id
            if session_id:
                session_threads[session_id] = thread_id

        yield "event: thinking\ndata: Starting process...\n\n"

        # Agent 1: Query Analyzer
        yield "event: thinking\ndata: Agent 1/?: Analyzing query and determining intent...\n\n"
        analyzed_query_obj = await self._run_analyzer_agent(thread_id, query, file_id)
        intent = analyzed_query_obj.get("intent", "research")
        analyzed_query_text = analyzed_query_obj.get("analyzed_query", query)
        
        # Format the thinking process output for better readability
        analyzer_response_str = json.dumps(analyzed_query_obj, indent=2)
        yield f"event: thinking\ndata: **Agent: Query Analyzer**\n```json\n{analyzer_response_str}\n```\n\n"

        if intent == "chat" and not file_id:
            yield "event: thinking\ndata: Intent classified as chat. Responding directly.\n\n"
            chat_response = await self._run_chat_agent(thread_id, query)
            yield f"event: report\ndata: {json.dumps({'response': chat_response})}\n\n"
            final_report_data = {
                "analyzed_query": analyzed_query_text,
                "task_list": None,
                "research_report": None,
                "visuals_summary": None,
                "final_evaluation": None,
                "chat_response": chat_response
            }
        else:
            clear_store() # Clear vector store only for new research tasks
            yield "event: thinking\ndata: Intent classified as research or file processing. Starting research pipeline.\n\n"
            
            # Agent 2: Task Planner
            yield "event: thinking\ndata: Agent 2/5: Planning tasks...\n\n"
            task_list = await self._run_planner_agent(thread_id, query, analyzed_query_text, file_id)
            yield f"event: thinking\ndata: **Agent: Task Planner**\n\n**Response:**\n{task_list}\n\n"

            # Agent 3: Researcher
            yield "event: thinking\ndata: Agent 3/5: Researching based on tasks...\n\n"
            research_report = await self._run_researcher_agent(thread_id, task_list, file_id)
            yield f"event: thinking\ndata: **Agent: Researcher**\n\n**Response:**\n{research_report}\n\n"

            # Agent 4: Visualizer
            yield "event: thinking\ndata: Agent 4/5: Generating visuals...\n\n"
            visuals_summary_obj = await self._run_visualizer_agent(thread_id, research_report, file_id)
            visuals_summary_text = visuals_summary_obj.get('summary', 'No visuals generated.')
            # Pass the image file_id from the visualizer to the evaluator
            visual_file_ids = visuals_summary_obj.get('file_ids', [])
            
            yield f"event: thinking\ndata: **Agent: Visualizer**\n\n**Response:**\n{visuals_summary_text}\n\n"

            # Agent 5: Evaluator
            yield "event: thinking\ndata: Agent 5/5: Evaluating final report...\n\n"
            final_report_data = await self._run_evaluator_agent(thread_id, task_list, research_report, visuals_summary_obj, query, file_id)
            yield f"event: thinking\ndata: **Agent: Evaluator**\n\n**Response:**\nFinal report generated.\n\n"

            # The final report data is now the direct output of the evaluator
            yield f"event: report\ndata: {json.dumps(final_report_data)}\n\n"
        print(final_report_data)

        yield "event: end\ndata: Process complete.\n\n"