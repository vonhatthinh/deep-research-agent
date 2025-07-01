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

            for chunk in r.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    if decoded_chunk.startswith('event: '):
                        event_type = decoded_chunk.split('event: ')[1]
                        data_line = next(r.iter_lines()).decode('utf-8')
                        data_content = data_line.split('data: ')[1]
                        
                        if event_type == 'thinking':
                            st.session_state.thinking_process.append(data_content)
                        elif event_type == 'report':
                            report_data = json.loads(data_content)
                            st.session_state.report = report_data
                        elif event_type == 'error':
                            st.session_state.error = data_content
                        
                        # Force a rerun to update the UI
                        st.rerun()
    
    except requests.exceptions.RequestException as e:
        st.session_state.error = f"Connection error: {e}"


# --- Main App Logic ---
if start_button:
    if not query:
        st.warning("Please enter a research query.")
    else:
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
    st.subheader("Research Report")
    if st.session_state.error:
        st.error(st.session_state.error)
    
    if st.session_state.report:
        report = st.session_state.report
        st.json(report) # Display the raw JSON for now

        # Prepare report for download
        report_str = json.dumps(report, indent=2)
        st.download_button(
            label="Download Report",
            data=report_str,
            file_name="research_report.json",
            mime="application/json"
        )

    else:
        st.info("The final research report will be displayed here.")

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
