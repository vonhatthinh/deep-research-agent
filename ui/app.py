import sys
import streamlit as st
import requests
import json
import time
import os
import uuid
from dotenv import load_dotenv
from fpdf import FPDF
import subprocess
import socket

load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
API_BASE_URL = "http://localhost:8000"
sys.path.append(root_dir)

st.set_page_config(page_title="Deep Research Assistant", layout="wide")

st.title("🤖 Deep Research Assistant")
st.markdown("Provide a topic, question, or document, and the AI agent will conduct in-depth research and compile a report.")

# --- FastAPI Server Auto-Start ---
def is_port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

if not is_port_in_use("0.0.0.0", 8000):
    try:
        subprocess.Popen([sys.executable, os.path.join(root_dir, "main.py")])
        st.info("Starting backend server on 0.0.0.0:8000...")
        time.sleep(2)  # Give the server a moment to start
    except Exception as e:
        st.error(f"Failed to start backend server: {e}")

# --- State Management ---
# Use a list to store the chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- File serving setup ---
# This is a workaround for Streamlit's lack of direct file serving.
# In a production environment, a proper web server (like Nginx) should handle this.
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
if not os.path.exists(PUBLIC_DIR):
    os.makedirs(PUBLIC_DIR)

# --- PDF Generation ---
def create_pdf_report(report_data):
    """Generates a PDF report from the structured research data."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Deep Research Report", ln=True, align='C')
    pdf.ln(10)

    # Helper function to write sections
    def write_section(title, content):
        if not content:  # Don't write empty sections
            return
        try:
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(0, 10, title, ln=True, align='L')
            pdf.set_font("Helvetica", '', 12)
            
            if isinstance(content, str):
                # Handle simple markdown (### headers and lists)
                # Encode the whole content once to handle special characters for FPDF
                safe_content = content.encode('latin-1', 'replace').decode('latin-1')
                lines = safe_content.split('\n')
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line.startswith("### "):
                        pdf.set_font("Helvetica", 'B', 13)
                        pdf.multi_cell(0, 8, stripped_line.replace("### ", ""))
                        pdf.set_font("Helvetica", '', 12)
                    elif stripped_line.startswith("- "):
                        pdf.multi_cell(0, 8, f"  {stripped_line}")
                    else:
                        pdf.multi_cell(0, 8, stripped_line)
                pdf.ln(2) # Add a little space after a block of text
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict): # For visuals
                        item_title = item.get('title', 'N/A').encode('latin-1', 'replace').decode('latin-1')
                        item_desc = item.get('description', 'No description.').encode('latin-1', 'replace').decode('latin-1')
                        item_id = item.get('file_id', 'N/A') # This will be a URL path
                        pdf.set_font("Helvetica", 'I', 12)
                        pdf.multi_cell(0, 10, f"- {item_title}:")
                        pdf.set_font("Helvetica", '', 12)
                        pdf.multi_cell(0, 10, f"  Description: {item_desc}")
                        # Try to embed image if it's a supported format and accessible
                        if item_id and (str(item_id).lower().endswith(('.png', '.jpg', '.jpeg')) or str(item_id).lower().endswith(('.gif', '.bmp'))):
                            try:
                                # Download the image from the API endpoint
                                import requests
                                image_url = f"{API_BASE_URL}{item_id}"
                                response = requests.get(image_url)
                                if response.status_code == 200:
                                    from tempfile import NamedTemporaryFile
                                    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(item_id)[-1]) as tmp_img:
                                        tmp_img.write(response.content)
                                        tmp_img.flush()
                                        pdf.image(tmp_img.name, w=100)  # width in mm
                                    os.unlink(tmp_img.name)
                                else:
                                    pdf.multi_cell(0, 10, f"  [Image could not be loaded: {image_url}]")
                            except Exception as e:
                                pdf.multi_cell(0, 10, f"  [Error embedding image: {e}]")
                        else:
                            # If not embeddable, just show the public URL
                            pdf.multi_cell(0, 10, f"  Image URL: {API_BASE_URL}{item_id}")
                    else: # For key findings, references
                        pdf.multi_cell(0, 10, f"- {str(item)}".encode('latin-1', 'replace').decode('latin-1'))
                    pdf.ln(2)
            pdf.ln(5)
        except Exception as e:
            pdf.set_font("Helvetica", 'B', 12)
            pdf.multi_cell(0, 10, f"Error rendering section '{title}': {e}")

    # Write content based on the new structure
    write_section("Executive Summary", report_data.get("executive_summary"))
    write_section("Detailed Report", report_data.get("detailed_report"))
    write_section("Key Findings", report_data.get("key_findings"))
    write_section("Visuals", report_data.get("visuals"))
    write_section("Conclusion", report_data.get("conclusion"))
    write_section("References", report_data.get("references"))
    
    if "raw_evaluator_output" in report_data:
        write_section("Raw Evaluator Output (Error)", report_data.get("raw_evaluator_output"))

    return pdf.output(dest='S').encode('latin-1') # type: ignore[return-value]


def stream_research(query, uploaded_files):
    """Generator function that streams the research process."""
    files_payload = [('files', (file.name, file.getvalue(), file.type)) for file in uploaded_files] if uploaded_files else None
    data = {'query': query, 'session_id': st.session_state.session_id}
    
    try:
        with requests.post(f"{API_BASE_URL}/query", files=files_payload, data=data, stream=True) as r:
            if r.status_code != 200:
                error_message = f"Error from server: {r.status_code} - {r.text}"
                st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})
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
                        
                        if event_type == 'thinking':
                            yield full_data + "\n"
                            time.sleep(0.1)
                        elif event_type == 'report':
                            try:
                                report_data = json.loads(full_data)
                                if 'response' in report_data and len(report_data) == 1:
                                    st.session_state.messages.append({'role': 'assistant', 'content': report_data['response']})
                                else:
                                    st.session_state.messages.append({'role': 'assistant', 'content': "Research complete. Here is the final report:", 'type': 'report_intro'})
                                    st.session_state.messages.append({'role': 'assistant', 'content': report_data, 'type': 'report_json'})
                            except json.JSONDecodeError:
                                error_msg = f"Failed to decode report JSON: {full_data}"
                                st.session_state.messages.append({'role': 'assistant', 'content': error_msg, 'type': 'error'})
                        elif event_type == 'error':
                            st.session_state.messages.append({'role': 'assistant', 'content': full_data, 'type': 'error'})
                        elif event_type == 'end':
                            pass
    
    except requests.exceptions.RequestException as e:
        error_message = f"Connection error: {e}"
        st.session_state.messages.append({'role': 'assistant', 'content': error_message, 'type': 'error'})


# --- Main App Logic ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get('type') == 'report_json':
            report_data = message["content"]
            
            st.header("Final Research Report")

            if report_data.get("executive_summary"):
                st.subheader("Executive Summary")
                st.markdown(report_data["executive_summary"])

            if report_data.get("detailed_report"):
                st.subheader("Detailed Report")
                st.markdown(report_data["detailed_report"])

            if report_data.get("key_findings"):
                st.subheader("Key Findings")
                for finding in report_data["key_findings"]:
                    st.markdown(f"- {finding}")

            if report_data.get("visuals"):
                st.subheader("Visuals")
                for visual in report_data["visuals"]:
                    st.markdown(f"**{visual.get('title', 'Visual')}**")
                    st.markdown(visual.get('description', ''))
                    file_id = visual.get("file_id")
                    # --- FIX: Show image if file_id is in the format file-xxxx or file_xxxx ---
                    if file_id:
                        # If file_id is a local file path, show as before
                        if str(file_id).startswith("/files/"):
                            image_url = f"{API_BASE_URL}{file_id}"
                            st.image(image_url, caption=visual.get("description", "Generated Visual"))
                        # If file_id is an OpenAI file id (e.g., file-xxxx), fetch from backend
                        elif str(file_id).startswith("file-") or str(file_id).startswith("file_"):
                            image_url = f"{API_BASE_URL}/files/{file_id}"
                            st.image(image_url, caption=visual.get("description", "Generated Visual"))
                        else:
                            st.markdown(f"Image ID: {file_id}")
                        # Add a button to delete the image after viewing
                        if st.button(f"Acknowledge and Remove Image: {visual.get('title')}", key=f"del_{file_id}"):
                            try:
                                # Make a request to the backend to delete the file (only for local files)
                                if str(file_id).startswith("/files/"):
                                    delete_url = f"{API_BASE_URL}/files/{os.path.basename(file_id)}"
                                    response = requests.delete(delete_url)
                                    if response.status_code == 200:
                                        st.success(f"Image {visual.get('title')} removed.")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to remove image: {response.text}")
                                else:
                                    st.info("Cannot delete remote/OpenAI images from here.")
                            except Exception as e:
                                st.error(f"Error removing image: {e}")
            
            if report_data.get("conclusion"):
                st.subheader("Conclusion")
                st.markdown(report_data["conclusion"])

            if report_data.get("references"):
                st.subheader("References")
                for ref in report_data["references"]:
                    st.markdown(f"- {ref}")

            # Add download button for the report
            print("Content of report_data:", report_data)
            pdf_report = create_pdf_report(report_data)
            st.download_button(
                label="Download Report as PDF",
                data=pdf_report,
                file_name="research_report.pdf",
                mime="application/pdf",
                key=f"download_{id(message)}"
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

# --- FIX: Always trigger research task on new user input ---
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("assistant"):
        st.write_stream(stream_research(query, uploaded_files))
    st.rerun()


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
