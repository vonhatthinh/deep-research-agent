import asyncio
import json
import typing
from openai import OpenAI
from openai.types.beta.assistant_tool_param import AssistantToolParam
from typing import AsyncGenerator, Optional

from core.config import settings
from agent.tools import tools_schema, available_tools
from database.vector_store import clear_store
from agent.base_agent import BaseAgent
from agents import Agent, Runner

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

    async def run_agent(self, assistant_id: str, thread_id: str, user_input: str, file_id: Optional[str] = None) -> str:
        attachments = []
        if file_id:
            attachments.append({"file_id": file_id, "tools": [{"type": "file_search"}, {"type": "code_interpreter"}]})

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_input,
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
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                for message in messages.data:
                    if message.role == 'assistant' and message.content and message.content[0].type == "text":
                        return message.content[0].text.value
                return "No assistant message found."
            
            elif run.status == 'requires_action' and run.required_action:
                tool_outputs = await self._handle_tool_calls(run.required_action, file_id)
                
                try:
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                except Exception as e:
                    return f"Error submitting tool outputs: {e}"
                continue # Loop back to check run status
            
            return f"Run ended with status: {run.status}"

    async def _handle_tool_calls(self, required_action, file_id: Optional[str]) -> list:
        tool_outputs = []
        for tool_call in required_action.submit_tool_outputs.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            
            output = ""
            if function_to_call:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    if function_name in ["process_and_store_file", "analyze_image_content", "add_text_to_store"] and 'file_id' not in function_args:
                        if file_id:
                            function_args['file_id'] = file_id
                         # If file_id is required by the tool but not available, skip the tool call or handle appropriately
                        elif function_name in ["process_and_store_file", "analyze_image_content"]:
                            print(f"Skipping tool call {function_name}: file_id is required but not provided.")
                            tool_outputs.append({"tool_call_id": tool_call.id, "output": f"Error: Tool '{function_name}' requires a file, but no file was uploaded."})
                            continue # Skip to the next tool call

                    if asyncio.iscoroutinefunction(function_to_call):
                        output = await function_to_call(**function_args)
                    else:
                        output = function_to_call(**function_args)
                    
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
        analyzed_query_json = await self.run_agent(analyzer_assistant.id, thread_id, query, file_id)
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
        return await self.run_agent(chat_assistant.id, thread_id, query)

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
        task_list = await self.run_agent(planner_assistant.id, thread_id, planner_input, file_id)
        print(f"Task List: {task_list}")
        return task_list

    async def _run_researcher_agent(self, thread_id: str, task_list: str, file_id: Optional[str]) -> str:
        """Runs the researcher agent to execute tasks and gather information."""
        researcher_assistant = self._create_assistant(
            name="Researcher Agent",
            instructions="You are a diligent researcher. Execute the given list of tasks to gather information. Use the \"tavily_web_search\" tools to find relevant data. Synthesize the findings into a comprehensive research report.",
            tools=[typing.cast(AssistantToolParam, tool) for tool in tools_schema]
        )
        return await self.run_agent(researcher_assistant.id, thread_id, f"Research tasks: {task_list}", file_id)

    async def _run_visualizer_agent(self, thread_id: str, research_report: str, file_id: Optional[str]) -> str:
        """Runs the visualizer agent to generate visuals for the report."""
        visualizer_assistant = self._create_assistant(
            name="Visualizer Agent",
            instructions="You are a data visualization expert. Review the research report and determine if any charts, graphs, or diagrams would enhance it. If so, use the code interpreter to generate them and provide their file IDs. Otherwise, state that no visuals are needed.",
            tools=[{"type": "code_interpreter"}]
        )
        visuals_summary = await self.run_agent(visualizer_assistant.id, thread_id, research_report, file_id)
        print(f"Visuals Summary: {visuals_summary}")
        return visuals_summary

    async def _run_evaluator_agent(self, thread_id: str, research_report: str, visuals_summary: str, query: str, file_id: Optional[str]) -> dict:
        """Runs the evaluator agent to compile the final structured report."""
        evaluator_instructions = f"""
You are a professional report writer. Your task is to compile a final, comprehensive research report based on the provided information.

The final report must be a JSON object with the following structure:
{{
  "executive_summary": "A brief overview of the research objective and outcome.",
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
- Research Report: {research_report}
- Visuals Summary: {visuals_summary}

**Instructions:**
1.  **Executive Summary (Required):** Write a brief overview of the research objective (based on the user query) and the outcome (based on the research report).
2.  **Key Findings (Optional):** Extract the most important, concise insights from the research report. If the report includes citations, include them. If there are no specific key findings, this can be an empty list.
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
        final_report_json = await self.run_agent(evaluator_assistant.id, thread_id, "Generate the final report based on the provided context.", file_id)
        
        print(f"Final Report JSON from Evaluator: {final_report_json}")
        try:
            return json.loads(final_report_json)
        except json.JSONDecodeError:
            print(f"Warning: Evaluator returned invalid JSON: {final_report_json}")
            # Fallback to a simple structure if JSON parsing fails
            return {
                "executive_summary": "Failed to generate structured report.",
                "key_findings": [],
                "visuals": [],
                "conclusion": "The evaluator agent failed to produce a valid JSON report. The raw output is provided below.",
                "references": [],
                "raw_evaluator_output": final_report_json
            }

    async def run_multi_agent_research(self, query: str, file_id: Optional[str] = None, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
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
        yield f"event: thinking\ndata: {json.dumps({'agent': 'Query Analyzer', 'response': analyzed_query_obj})}\n\n"

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
            yield f"event: thinking\ndata: {json.dumps({'agent': 'Task Planner', 'response': task_list})}\n\n"

            # Agent 3: Researcher
            yield "event: thinking\ndata: Agent 3/5: Researching based on tasks...\n\n"
            research_report = await self._run_researcher_agent(thread_id, task_list, file_id)
            yield f"event: thinking\ndata: {json.dumps({'agent': 'Researcher', 'response': research_report})}\n\n"

            # Agent 4: Visualizer
            yield "event: thinking\ndata: Agent 4/5: Generating visuals...\n\n"
            visuals_summary = await self._run_visualizer_agent(thread_id, research_report, file_id)
            yield f"event: thinking\ndata: {json.dumps({'agent': 'Visualizer', 'response': visuals_summary})}\n\n"

            # Agent 5: Evaluator
            yield "event: thinking\ndata: Agent 5/5: Evaluating final report...\n\n"
            final_report_data = await self._run_evaluator_agent(thread_id, research_report, visuals_summary, query, file_id)
            yield f"event: thinking\ndata: {json.dumps({'agent': 'Evaluator', 'response': 'Final report generated.'})}\n\n"

            # The final report data is now the direct output of the evaluator
            yield f"event: report\ndata: {json.dumps(final_report_data)}\n\n"
        print(final_report_data)
        yield "event: end\ndata: Process complete.\n\n"
        return