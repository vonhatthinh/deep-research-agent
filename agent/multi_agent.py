import asyncio
import json
import typing
from openai import OpenAI
from openai.types.beta.assistant_tool_param import AssistantToolParam
from typing import AsyncGenerator, Optional

from core.config import settings
from agent.tools import tools_schema, available_tools
from database.vector_store import clear_store

client = OpenAI(api_key=settings.OPENAI_API_KEY)

async def run_multi_agent_research(query: str, file_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Runs the multi-agent research process and streams back the results.
    """
    # Clear the vector store at the beginning of each research task
    clear_store()

    yield "event: thinking\ndata: Starting multi-agent research process...\n\n"

    # Agent 1: Query Analyzer
    yield "event: thinking\ndata: Agent 1/5: Analyzing query...\n\n"
    analyzer_assistant = client.beta.assistants.create(
        name="Query Analyzer Agent",
        instructions="You are an expert at analyzing and interpreting user queries. Your goal is to understand the user's intent, identify key entities, and clarify any ambiguities. Provide a clear and concise interpretation of the query.",
        model="gpt-4o",
    )
    analyzed_query = await run_agent(analyzer_assistant.id, query, file_id)
    yield f"event: agent_response\ndata: {{\"agent\": \"Query Analyzer\", \"response\": \"{analyzed_query.replace('\"', '\\\"')}\"}}\n\n"

    # Agent 2: Task Planner
    yield "event: thinking\ndata: Agent 2/5: Planning tasks...\n\n"
    planner_assistant = client.beta.assistants.create(
        name="Task Planner Agent",
        instructions="You are a meticulous planner. Your first step is always to store the original user query in the vector database using the `add_text_to_store` tool. If a file is provided, your second step is to process it using either `process_and_store_file` for documents or `analyze_image_content` for images. After that, create a plan to query the knowledge base and the web to answer the user's request. Output a JSON object with a list of tasks.",
        model="gpt-4o",
        response_format={"type": "json_object"}
    )
    planner_input = f"Original User Query: {query}"
    if file_id:
        planner_input += f"\nFile ID: {file_id}"
    task_list_json = await run_agent(planner_assistant.id, planner_input, file_id)
    yield f"event: agent_response\ndata: {{\"agent\": \"Task Planner\", \"response\": {task_list_json}}}\n\n"

    # Agent 3: Researcher
    yield "event: thinking\ndata: Agent 3/5: Researching based on tasks...\n\n"
    researcher_assistant = client.beta.assistants.create(
        name="Researcher Agent",
        instructions="You are a diligent researcher. Execute the given list of tasks to gather information. Use the available tools to find relevant data. Synthesize the findings into a comprehensive research report.",
        model="gpt-4o",
        tools=[typing.cast(AssistantToolParam, tool) for tool in tools_schema]
    )
    research_report = await run_agent(researcher_assistant.id, f"Research tasks: {task_list_json}")
    yield f"event: agent_response\ndata: {{\"agent\": \"Researcher\", \"response\": \"{research_report.replace('\"', '\\\"')}\"}}\n\n"

    # Agent 4: Visualizer
    yield "event: thinking\ndata: Agent 4/5: Generating visuals...\n\n"
    visualizer_assistant = client.beta.assistants.create(
        name="Visualizer Agent",
        instructions="You are a data visualization expert. Review the research report and determine if any charts, graphs, or diagrams would enhance it. If so, use the code interpreter to generate them and provide their file IDs. Otherwise, state that no visuals are needed.",
        model="gpt-4o",
        tools=[{"type": "code_interpreter"}]
    )
    visuals_summary = await run_agent(visualizer_assistant.id, research_report)
    yield f"event: agent_response\ndata: {{\"agent\": \"Visualizer\", \"response\": \"{visuals_summary.replace('\"', '\\\"')}\"}}\n\n"

    # Agent 5: Evaluator
    yield "event: thinking\ndata: Agent 5/5: Evaluating final report...\n\n"
    evaluator_assistant = client.beta.assistants.create(
        name="Evaluator Agent",
        instructions="You are a critical evaluator. Assess the final research report for completeness, accuracy, and objectivity. Provide a final evaluation and suggest any areas for improvement.",
        model="gpt-4o",
    )
    final_evaluation = await run_agent(evaluator_assistant.id, f"Report: {research_report}\n\nVisuals Summary: {visuals_summary}")
    yield f"event: agent_response\ndata: {{\"agent\": \"Evaluator\", \"response\": \"{final_evaluation.replace('\"', '\\\"')}\"}}\n\n"

    yield "event: end\ndata: Multi-agent research process complete.\n\n"

async def run_agent(assistant_id: str, user_input: str, file_id: Optional[str] = None) -> str:
    thread = client.beta.threads.create()

    # Pass the file_id to the thread if it exists, making it accessible to all agents
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
            if message.role == 'assistant':
                if message.content and message.content[0].type == "text":
                    return message.content[0].text.value
    elif run.status == 'requires_action':
        if run.required_action:
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools.get(function_name)
                
                if function_to_call:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        # The file_id is not automatically passed to the tool, 
                        # so we add it to the arguments if the tool needs it
                        if function_name in ["process_and_store_file", "analyze_image_content"] and 'file_id' not in function_args:
                            function_args['file_id'] = file_id

                        output = function_to_call(**function_args)
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": output,
                        })
                    except Exception as e:
                        print(f"Error calling tool {function_name}: {e}")
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": f"Error: {e}",
                        })
                else:
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"Error: Tool '{function_name}' not found.",
                    })

            # Submit tool outputs back to the Assistant
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            
            # Poll for the run to complete after submitting tool outputs
            while run.status in ['queued', 'in_progress']:
                await asyncio.sleep(1)
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if run.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                for message in messages.data:
                    if message.role == 'assistant':
                        if message.content and message.content[0].type == "text":
                            return message.content[0].text.value

    return f"Run failed with status: {run.status}"
