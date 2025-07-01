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


# --- UI Components ---
with st.sidebar:
    st.header("Controls")
    query = st.text_area("Enter your research query:", height=150)
    uploaded_file = st.file_uploader("Upload a document (optional)", type=['pdf', 'docx', 'csv', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg'])
    start_button = st.button("Send Message")
    debug_mode = st.checkbox("Enable debug mode")


# --- State Management ---
if 'thinking_process' not in st.session_state:
    st.session_state.thinking_process = []
if 'report' not in st.session_state:
    st.session_state.report = None
if 'error' not in st.session_state:
    st.session_state.error = None
if 'user_query' not in st.session_state:
    st.session_state.user_query = None


def start_research_task(query, file):
    """Sends the research request to the FastAPI backend."""
    st.session_state.thinking_process = []
    st.session_state.report = None
    st.session_state.error = None
    
    files = {'file': (file.name, file.getvalue(), file.type)} if file else None
    data = {'query': query}

    try:
        with requests.post(f"{API_BASE_URL}/query", files=files, data=data, stream=True) as r:
            if r.status_code != 200:
                st.session_state.error = f"Error from server: {r.status_code} - {r.text}"
                return

            event_type = None
            data_buffer = []

            for line in r.iter_lines(decode_unicode=True):
                if not line:  # An empty line marks the end of an event
                    if event_type and data_buffer:
                        full_data = "".join(data_buffer)
                        
                        if event_type == 'thinking':
                            st.session_state.thinking_process.append(full_data)
                        elif event_type == 'report':
                            try:
                                report_data = json.loads(full_data)
                                st.session_state.report = report_data
                            except json.JSONDecodeError:
                                st.session_state.error = f"Failed to decode report JSON: {full_data}"
                        elif event_type == 'error':
                            st.session_state.error = full_data

                    # Reset for the next event
                    event_type = None
                    data_buffer = []
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
                    st.session_state.thinking_process.append(full_data)
                elif event_type == 'report':
                    try:
                        report_data = json.loads(full_data)
                        st.session_state.report = report_data
                    except json.JSONDecodeError:
                        st.session_state.error = f"Failed to decode report JSON: {full_data}"
                elif event_type == 'error':
                    st.session_state.error = full_data

    except requests.exceptions.RequestException as e:
        st.session_state.error = f"Connection error: {e}"


# --- Main App Logic ---
if start_button:
    if not query:
        st.warning("Please enter a research query.")
    else:
        # Reset state for new query
        st.session_state.thinking_process = []
        st.session_state.report = None
        st.session_state.error = None
        st.session_state.user_query = query # Store the query

        with st.spinner("The agent is now conducting research... This may take a few minutes."):
            start_research_task(query, uploaded_file)

# --- Display Results ---
col1, col2 = st.columns([1, 2])


with col1:
    st.subheader("Thinking Process")
    if st.session_state.thinking_process:
        thinking_container = st.container()
        with thinking_container:
            for step in st.session_state.thinking_process:
                st.info(step)
    else:
        st.info("The agent's thought process will appear here once research begins.")

with col2:
    st.subheader("Research Conversation")

    if not st.session_state.user_query:
        st.info("The research conversation will appear here.")
    else:
        with st.chat_message("user"):
            st.markdown(st.session_state.user_query)

        if st.session_state.error:
            with st.chat_message("assistant"):
                st.error(st.session_state.error)
        
        if st.session_state.report:
            with st.chat_message("assistant"):
                st.info("Here is the final research report:")
                report = st.session_state.report
                st.json(report)

                # Prepare report for download
                report_str = json.dumps(report, indent=2)
                st.download_button(
                    label="Download Report",
                    data=report_str,
                    file_name="research_report.json",
                    mime="application/json"
                )

if debug_mode:
    if os.getenv('STREAMLIT_ENV') == 'development':
        with st.expander("Debug Information"):
            # Only display safe debugging information
            debug_info = {
                'thinking_process_count': len(st.session_state.get('thinking_process', [])),
                'report_available': st.session_state.get('report') is not None,
                'error_present': st.session_state.get('error') is not None,
                'session_keys': list(st.session_state.keys())
            }
            st.write(debug_info)
    else:
        st.warning("Debug mode is only available in development environment.")
