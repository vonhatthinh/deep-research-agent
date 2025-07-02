import streamlit as st
import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://localhost:8000"


st.set_page_config(page_title="Deep Research Assistant", layout="wide")

st.title("🤖 Deep Research Assistant")
st.markdown("Provide a topic, question, or document, and the AI agent will conduct in-depth research and compile a report.")

# --- State Management ---
# Use a list to store the chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
# Remove old state variables
if 'thinking_process' in st.session_state:
    del st.session_state.thinking_process
if 'report' in st.session_state:
    del st.session_state.report
if 'error' in st.session_state:
    del st.session_state.error
if 'user_query' in st.session_state:
    del st.session_state.user_query


def start_research_task(query, file):
    """Sends the research request to the FastAPI backend and updates chat history."""
    
    files = {'file': (file.name, file.getvalue(), file.type)} if file else None
    data = {'query': query}

    try:
        with requests.post(f"{API_BASE_URL}/query", files=files, data=data, stream=True) as r:
            if r.status_code != 200:
                error_message = f"Error from server: {r.status_code} - {r.text}"
                st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})
                st.rerun() # Rerun to display the error
                return

            event_type = None
            data_buffer = []

            for line in r.iter_lines(decode_unicode=True):
                if not line:  # An empty line marks the end of an event
                    if event_type and data_buffer:
                        full_data = "".join(data_buffer)
                        
                        if event_type == 'thinking':
                            # Append thinking steps as assistant messages
                            st.session_state.messages.append({'role': 'assistant', 'content': full_data, 'type': 'thinking'})
                        elif event_type == 'report':
                            try:
                                report_data = json.loads(full_data)
                                # Append report as assistant message
                                st.session_state.messages.append({'role': 'assistant', 'content': "Here is the final research report:", 'type': 'report_intro'})
                                st.session_state.messages.append({'role': 'assistant', 'content': report_data, 'type': 'report_json'})
                            except json.JSONDecodeError:
                                error_message = f"Failed to decode report JSON: {full_data}"
                                st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})
                        elif event_type == 'error':
                            error_message = full_data
                            st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})

                    # Reset for the next event
                    event_type = None
                    data_buffer = []
                    # Rerun to update the display with new messages
                    st.rerun()
                    continue

                if line.startswith('event:'):
                    event_type = line.split(':', 1)[1].strip()
                elif line.startswith('data:'):
                    # Append the data part, stripping "data:" prefix
                    data_buffer.append(line.split(':', 1)[1].strip())

            # After the loop, process any remaining data that might not have been followed by a blank line
            if event_type and data_buffer:
                full_data = "".join(data_buffer)
                if event_type == 'thinking':
                    st.session_state.messages.append({'role': 'assistant', 'content': full_data, 'type': 'thinking'})
                elif event_type == 'report':
                    try:
                        report_data = json.loads(full_data)
                        st.session_state.messages.append({'role': 'assistant', 'content': "Here is the final research report:", 'type': 'report_intro'})
                        st.session_state.messages.append({'role': 'assistant', 'content': report_data, 'type': 'report_json'})
                    except json.JSONDecodeError:
                        error_message = f"Failed to decode report JSON: {full_data}"
                        st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})
                elif event_type == 'error':
                    error_message = full_data
                    st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})
                # Rerun to update the display with the last message
                st.rerun()


    except requests.exceptions.RequestException as e:
        error_message = f"Connection error: {e}"
        st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})
        st.rerun() # Rerun to display the error


# --- Main App Logic ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get('type') == 'report_json':
            st.json(message["content"])
            # Add download button for the report
            report_str = json.dumps(message["content"], indent=2)
            st.download_button(
                label="Download Report",
                data=report_str,
                file_name="research_report.json",
                mime="application/json",
                key=f"download_{id(message)}" # Use a unique key
            )
        elif message.get('type') == 'thinking':
             st.info(message["content"])
        elif message.get('type') == 'error':
             st.error(message["content"])
        else:
            st.markdown(message["content"])


# Chat input at the bottom
query = st.chat_input("Enter your research query:")
uploaded_file = st.file_uploader("Upload a document (optional)", type=['pdf', 'docx', 'csv', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg'])
# Start button will be triggered by the chat input
debug_mode = st.checkbox("Enable debug mode")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Start the research task
    # Need to rerun to display the user message immediately
    st.rerun()

print(st.session_state)
# This part will be triggered after the rerun caused by the chat_input or start_research_task
# Check if the last message was from the user and no assistant response has started yet
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # Find the user query message
    user_message = st.session_state.messages[-1]
    # Check if the next message is NOT from the assistant (meaning the task hasn't started processing yet)
    if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "user":
         with st.spinner("The agent is now conducting research... This may take a few minutes."):
            # Pass the uploaded file from the sidebar
            start_research_task(user_message["content"], uploaded_file)


if debug_mode:
    if os.getenv('STREAMLIT_ENV') == 'development':
        with st.expander("Debug Information"):
            # Only display safe debugging information
            debug_info = {
                'message_count': len(st.session_state.get('messages', [])),
                'session_keys': list(st.session_state.keys())
            }
            st.write(debug_info)
    else:
        st.warning("Debug mode is only available in development environment.")
