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

    async def run_agent(self, assistant_id: str, user_input: str, file_id: Optional[str] = None) -> str:
        thread = client.beta.threads.create()
        print(f"User input: {user_input}")
        attachments = []
        if file_id:
            attachments.append({"file_id": file_id, "tools": [{"type": "file_search"}, {"type": "code_interpreter"}]})

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input,
            attachments=attachments
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

        while run.status in ['queued', 'in_progress', 'cancelling']:
            await asyncio.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for message in messages.data:
                if message.role == 'assistant' and message.content and message.content[0].type == "text":
                    return message.content[0].text.value
            return "No assistant message found."
        
        elif run.status == 'requires_action' and run.required_action:
            tool_outputs = await self._handle_tool_calls(run.required_action, file_id)
            
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            
            while run.status in ['queued', 'in_progress']:
                await asyncio.sleep(1)
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if run.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                for message in messages.data:
                    if message.role == 'assistant' and message.content and message.content[0].type == "text":
                        return message.content[0].text.value
                return "No assistant message found after tool calls."

        return f"Run failed with status: {run.status}"

    async def _handle_tool_calls(self, required_action, file_id: Optional[str]) -> list:
        tool_outputs = []
        for tool_call in required_action.submit_tool_outputs.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            
            output = ""
            if function_to_call:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    # Pass file_id to tools that might need it, even if not explicitly in args
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

    async def _run_analyzer_agent(self, query: str, file_id: Optional[str]) -> dict:
        """Runs the query analyzer agent to determine intent and analyze the query."""
        analyzer_assistant = self._create_assistant(
            name="Query Analyzer Agent",
            instructions="You are an expert at analyzing user input. Determine if the user's request is a simple chat message or requires in-depth research, potentially involving file analysis if a file is attached. Respond with a JSON object containing the 'intent' ('chat' or 'research') and the 'analyzed_query'.",
            response_format={"type": "json_object"}
        )
        analyzed_query_json = await self.run_agent(analyzer_assistant.id, query, file_id)
        print(f"Analyzed Query JSON: {analyzed_query_json}")
        try:
            return json.loads(analyzed_query_json)
        except json.JSONDecodeError:
            print(f"Warning: Query Analyzer returned invalid JSON: {analyzed_query_json}")
            return {"intent": "research", "analyzed_query": query, "error": "Invalid JSON response"}

    async def _run_chat_agent(self, query: str) -> str:
        """Runs the chat agent for simple queries."""
        chat_assistant = self._create_assistant(
            name="Chat Agent",
            instructions="You are a helpful and friendly chatbot. Respond to the user's query directly.",
        )
        return await self.run_agent(chat_assistant.id, query)

    async def _run_planner_agent(self, query: str, analyzed_query_text: str, file_id: Optional[str]) -> str:
        """Runs the task planner agent to create a research plan."""
        planner_instructions = "You are a meticulous planner. Your first step is always to store the original user query in the vector database using the `add_text_to_store` tool."
        if file_id:
            planner_instructions += " If a file is provided, your second step is to process it using either `process_and_store_file` for documents or `analyze_image_content` for images."
        planner_instructions += " After that, create a plan to query the knowledge base and the web to answer the user's request. Output a JSON object with a list of tasks."

        planner_assistant = self._create_assistant(
            name="Task Planner Agent",
            instructions=planner_instructions,
            response_format={"type": "json_object"},
            tools=[typing.cast(AssistantToolParam, tool) for tool in tools_schema if tool['function']['name'] in ['add_text_to_store', 'process_and_store_file', 'analyze_image_content']]
        )
        planner_input = f"Original User Query: {query}\nAnalyzed Query: {analyzed_query_text}"
        if file_id:
            planner_input += f"\nFile ID: {file_id}"
        task_list_json = await self.run_agent(planner_assistant.id, planner_input, file_id)
        print(f"Task List JSON: {task_list_json}")
        return task_list_json

    async def _run_researcher_agent(self, task_list_json: str, file_id: Optional[str]) -> str:
        """Runs the researcher agent to execute tasks and gather information."""
        researcher_assistant = self._create_assistant(
            name="Researcher Agent",
            instructions="You are a diligent researcher. Execute the given list of tasks to gather information. Use the available tools to find relevant data. Synthesize the findings into a comprehensive research report.",
            tools=[typing.cast(AssistantToolParam, tool) for tool in tools_schema]
        )
        return await self.run_agent(researcher_assistant.id, f"Research tasks: {task_list_json}", file_id)

    async def _run_visualizer_agent(self, research_report: str, file_id: Optional[str]) -> str:
        """Runs the visualizer agent to generate visuals for the report."""
        visualizer_assistant = self._create_assistant(
            name="Visualizer Agent",
            instructions="You are a data visualization expert. Review the research report and determine if any charts, graphs, or diagrams would enhance it. If so, use the code interpreter to generate them and provide their file IDs. Otherwise, state that no visuals are needed.",
            tools=[{"type": "code_interpreter"}]
        )
        visuals_summary = await self.run_agent(visualizer_assistant.id, research_report, file_id)
        print(f"Visuals Summary: {visuals_summary}")
        return visuals_summary

    async def _run_evaluator_agent(self, research_report: str, visuals_summary: str, file_id: Optional[str]) -> str:
        """Runs the evaluator agent to assess the final report."""
        evaluator_assistant = self._create_assistant(
            name="Evaluator Agent",
            instructions="You are a critical evaluator. Assess the final research report for completeness, accuracy, and objectivity. Provide a final evaluation and suggest any areas for improvement.",
        )
        final_evaluation = await self.run_agent(evaluator_assistant.id, f"Report: {research_report}\n\nVisuals Summary: {visuals_summary}", file_id)
        print(f"Final Evaluation: {final_evaluation}")
        return final_evaluation

    async def run_multi_agent_research(self, query: str, file_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Runs the multi-agent research process and streams back the results.
        """
        clear_store()
        yield "event: thinking\ndata: Starting process...\n\n"

        # Agent 1: Query Analyzer
        yield "event: thinking\ndata: Agent 1/?: Analyzing query and determining intent...\n\n"
        analyzed_query_obj = await self._run_analyzer_agent(query, file_id)
        intent = analyzed_query_obj.get("intent", "research")
        analyzed_query_text = analyzed_query_obj.get("analyzed_query", query)
        yield f"event: thinking\ndata: {json.dumps({'agent': 'Query Analyzer', 'response': analyzed_query_obj})}\n\n"

        if intent == "chat" and not file_id:
            yield "event: thinking\ndata: Intent classified as chat. Responding directly.\n\n"
            chat_response = await self._run_chat_agent(query)
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
            yield "event: thinking\ndata: Intent classified as research or file processing. Starting research pipeline.\n\n"
            
            # Agent 2: Task Planner
            yield "event: thinking\ndata: Agent 2/5: Planning tasks...\n\n"
            task_list_json = await self._run_planner_agent(query, analyzed_query_text, file_id)
            try:
                task_list_obj = json.loads(task_list_json)
                yield f"event: thinking\ndata: {json.dumps({'agent': 'Task Planner', 'response': task_list_obj})}\n\n"
            except json.JSONDecodeError:
                yield f"event: thinking\ndata: {json.dumps({'agent': 'Task Planner', 'error': 'Invalid JSON response', 'raw_response': task_list_json})}\n\n"
                task_list_json = json.dumps({"tasks": ["Could not parse planner output."]})

            # Agent 3: Researcher
            yield "event: thinking\ndata: Agent 3/5: Researching based on tasks...\n\n"
            research_report = await self._run_researcher_agent(task_list_json, file_id)
            yield f"event: thinking\ndata: {json.dumps({'agent': 'Researcher', 'response': research_report})}\n\n"

            # Agent 4: Visualizer
            yield "event: thinking\ndata: Agent 4/5: Generating visuals...\n\n"
            visuals_summary = await self._run_visualizer_agent(research_report, file_id)
            yield f"event: thinking\ndata: {json.dumps({'agent': 'Visualizer', 'response': visuals_summary})}\n\n"

            # Agent 5: Evaluator
            yield "event: thinking\ndata: Agent 5/5: Evaluating final report...\n\n"
            final_evaluation = await self._run_evaluator_agent(research_report, visuals_summary, file_id)
            yield f"event: thinking\ndata: {json.dumps({'agent': 'Evaluator', 'response': final_evaluation})}\n\n"

            # Combine results into a final report
            try:
                tasks = json.loads(task_list_json)
            except json.JSONDecodeError:
                tasks = task_list_json
                
            final_report_data = {
                "analyzed_query": analyzed_query_text,
                "task_list": tasks,
                "research_report": research_report,
                "visuals_summary": visuals_summary,
                "final_evaluation": final_evaluation
            }
            yield f"event: report\ndata: {json.dumps(final_report_data, indent=2)}\n\n"
        print(final_report_data)
        yield "event: end\ndata: Process complete.\n\n"
        return