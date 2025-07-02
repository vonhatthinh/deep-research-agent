import sys
import streamlit as st
import requests
import json
import time
import os
import uuid
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
API_BASE_URL = "http://localhost:8000"
sys.path.append(root_dir)

st.set_page_config(page_title="Deep Research Assistant", layout="wide")

st.title("🤖 Deep Research Assistant")
st.markdown("Provide a topic, question, or document, and the AI agent will conduct in-depth research and compile a report.")

# --- State Management ---
# Use a list to store the chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
# Remove old state variables
if 'thinking_process' in st.session_state:
    del st.session_state.thinking_process
if 'report' in st.session_state:
    del st.session_state.report
if 'error' in st.session_state:
    del st.session_state.error
if 'user_query' in st.session_state:
    del st.session_state.user_query


def create_pdf_report(report_data):
    """Generates a PDF report from the structured research data."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Deep Research Report", ln=True, align='C')
    pdf.ln(10)

    # Helper function to write sections
    def write_section(title, content):
        if not content:  # Don't write empty sections
            return
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, title, ln=True, align='L')
        pdf.set_font("Arial", '', 12)
        
        if isinstance(content, str):
            pdf.multi_cell(0, 10, content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict): # For visuals
                    item_title = item.get('title', 'N/A')
                    item_desc = item.get('description', 'No description.')
                    item_id = item.get('file_id', 'N/A')
                    pdf.set_font("Arial", 'I', 12)
                    pdf.multi_cell(0, 10, f"- {item_title}:")
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(0, 10, f"  Description: {item_desc}")
                    pdf.multi_cell(0, 10, f"  File ID: {item_id}")
                else: # For key findings, references
                    pdf.multi_cell(0, 10, f"- {item}")
                pdf.ln(2)
        pdf.ln(5)

    # Write content based on the new structure
    write_section("Executive Summary", report_data.get("executive_summary"))
    write_section("Key Findings", report_data.get("key_findings"))
    write_section("Visuals", report_data.get("visuals"))
    write_section("Conclusion", report_data.get("conclusion"))
    write_section("References", report_data.get("references"))
    
    if "raw_evaluator_output" in report_data:
        write_section("Raw Evaluator Output (Error)", report_data.get("raw_evaluator_output"))

    return pdf.output(dest='S').encode('latin-1')


def start_research_task(query, uploaded_files):
    """Sends the research request to the FastAPI backend and updates chat history."""
    files_payload = [('files', (file.name, file.type)) for file in uploaded_files] if uploaded_files else None
    data = {'query': query, 'session_id': st.session_state.session_id}
    print("total input content:", data, "files:", files_payload)
    try:
        with requests.post(f"{API_BASE_URL}/query", files=files_payload, data=data, stream=True) as r:
            if r.status_code != 200:
                error_message = f"Error from server: {r.status_code} - {r.text}"
                st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})
                st.rerun() # Rerun to display the error
                return
            
            event_buffer = ""
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                event_buffer += chunk
                while '\n\n' in event_buffer:
                    event_str, event_buffer = event_buffer.split('\n\n', 1)
                    
                    event_type = None
                    data_lines = []
                    for line in event_str.strip().split('\n'):
                        if line.startswith('event:'):
                            event_type = line.split(':', 1)[1].strip()
                        elif line.startswith('data:'):
                            data_lines.append(line.split(':', 1)[1].strip())
                    
                    if event_type and data_lines:
                        full_data = "".join(data_lines)
                        print(f"Received Event: {event_type}, Data: {full_data}")

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
                            st.session_state.messages.append({'role': 'assistant', 'content': full_data, 'type': 'error'})
            
            # After processing all events, rerun to update the display
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
            pdf_report = create_pdf_report(message["content"])
            st.download_button(
                label="Download Report as PDF",
                data=pdf_report,  # Read the PDF content
                file_name="research_report.pdf",
                mime="application/pdf",
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
uploaded_files = st.file_uploader("Upload a document (optional)", type=['pdf', 'docx', 'csv', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg'], accept_multiple_files=True)
# Start button will be triggered by the chat input
debug_mode = st.checkbox("Enable debug mode")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # Find the user query message
    user_message = st.session_state.messages[-1]
    if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "user":
         with st.spinner("The agent is now conducting research... This may take a few minutes."):
            start_research_task(user_message["content"], uploaded_files)


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
