# Deep Research Agent

This project implements a multimodal Deep Research Agent capable of handling user queries with text, image, or file inputs, conducting deep web research, generating visual outputs, and compiling comprehensive reports. The solution is designed with a modular architecture, leveraging FastAPI for the backend and Streamlit for an interactive user interface.

## Workflow

![Workflow Diagram](./resources/workflow_diagram.png)

## Technical Overview and Key Components

The Deep Research Agent is built with a modern Python stack, emphasizing modularity and asynchronous operations to handle complex research tasks efficiently.

-   **Backend**: A robust API built with **FastAPI**, serving as the central hub for processing user requests. It handles file uploads, manages agent interactions, and streams responses back to the client.
-   **Frontend**: An interactive web application created with **Streamlit**. It provides a user-friendly interface for submitting queries, uploading files, viewing the agent's thinking process in real-time, and downloading the final report.
-   **Core Agent Logic**: The research capability is powered by a multi-agent system orchestrated using the **OpenAI Assistants API**. This allows for a clear separation of concerns, with specialized agents for query analysis, task planning, web research, data visualization, and final report evaluation.
-   **Multimodality**: The agent can process various input types:
    -   **Text**: Standard user queries.
    -   **Images**: Utilizes **Google's Gemini Pro Vision** to analyze and describe image content, providing context for the research.
    -   **Documents**: Supports parsing of PDF, DOCX, and CSV files to extract text content.
-   **Tooling**: The agents are equipped with tools for:
    -   **Web Search**: Using the **Tavily Search API** for up-to-date information gathering.
    -   **Image Generation**: Creating relevant visuals for reports with **DALL-E 3**.
    -   **Knowledge Base**: A vector store for storing and retrieving information relevant to the user's query.
-   **Asynchronous Streaming**: The entire process, from the backend to the frontend, is designed to be asynchronous. The FastAPI backend uses `StreamingResponse` to send Server-Sent Events (SSE), and the Streamlit UI uses `st.write_stream` to display the agent's progress live.

## Features

-   **Input Handling**: Accepts text prompts, image uploads, and document uploads (PDF, DOCX, CSV).
-   **Multi-Agent Research Pipeline**:
    -   **Query Analyzer**: Determines user intent (simple chat vs. deep research).
    -   **Task Planner**: Creates a step-by-step plan for the research.
    -   **Researcher**: Executes the plan by searching the web and internal knowledge.
    -   **Visualizer**: Generates a relevant image or diagram for the report.
    -   **Evaluator**: Compiles all gathered information into a structured, final report.
-   **Output Generation**: Produces a well-structured research report in JSON format, which is then displayed in the UI and can be downloaded as a PDF.
-   **FastAPI Backend**: Exposes a `/query` endpoint that streams the entire research process.
-   **Streamlit UI**: Provides an interactive web interface for input submission, real-time streamed responses, visualization previews, and report downloads.

## Setup Instructions

### Prerequisites

-   Python 3.9+
-   An OpenAI API Key
-   A Tavily API Key
-   A Gemini API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd deep-research-agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add your API keys:
    ```
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

### Running the Application

The application runs in two separate terminals: one for the backend and one for the frontend.

1.  **Start the Backend Server:**
    Open a terminal and run the following command from the project's root directory:
    ```bash
    python main.py
    ```
    The FastAPI server will start on `http://localhost:8000`.

2.  **Start the Frontend UI:**
    Open a second terminal and run the following command from the project's root directory:
    ```bash
    streamlit run ui/app.py
    ```
    The Streamlit application will open in your browser, typically at `http://localhost:8501`.

## API Documentation

The FastAPI application automatically generates interactive API documentation using Swagger UI. Once the backend is running, you can access it at `http://localhost:8000/docs`.

### Endpoints

-   `GET /status`:
    -   **Description**: Returns the health status of the API.
    -   **Response**: `{"status": "ok", "message": "Deep Research Assistant API is running."}`

-   `POST /query`:
    -   **Description**: Receives a user query and optional files, then initiates the deep research process, streaming the results back.
    -   **Request**: A `multipart/form-data` request containing:
        -   `query` (str): The user's text query.
        -   `session_id` (str): A unique ID for the user's session.
        -   `files` (UploadFile, optional): One or more files (images, PDF, DOCX, CSV).
    -   **Response**: A `text/event-stream` response with Server-Sent Events detailing the agent's progress.

## Sample Inputs and Outputs

### Sample Input

**Text Query:** "Plan a 10-day itinerary in Japan for a solo traveler who loves culture and food"

### Sample Output

The application will stream the agent's thinking process live. The final output will be a structured report displayed in the UI, similar to this:

**Executive Summary**: This report outlines a comprehensive 10-day itinerary for a solo traveler in Japan, appealing to those
who have a keen interest in culture and food. The itinerary is carefully designed to explore Japan's
rich cultural heritage, unique culinary experiences, and scenic landmarks, ensuring a fulfilling travel
experience.

**Key Findings**:
Cultural Experiences and Cities:
1. **Key Cultural Experiences:**
 - Major cultural highlights include visiting temples, participating in tea ceremonies, and exploring
traditional arts (ikebana, calligraphy).
 - Cultural festivals such as Gion Matsuri in Kyoto, Tenjin Matsuri in Osaka, and the Fireworks
Festivals.
2. **Unique Festivals and Historical Sites:**
 - Historical landmarks such as Kyoto's Kinkaku-ji (Golden Pavilion), Hiroshima Peace Memorial
Park, and Nara's Todai-ji Temple.
 - UNESCO World Heritage sites like the Historic Villages of Shirakawa-go and the
Shirakami-Sanchi.
3. **Popular Destinations:**
 - Must-visits: Tokyo for its mix of modern and tradition, Kyoto for historical and cultural depth, Nara
for ancient culture, and Osaka for vibrant nightlife and food.
 - Consider visiting Himeji Castle and Nikko for additional cultural insights.
Culinary Exploration:
1. **Food Tours and Local Specialties:**
 - Prominent food tours available in Osaka's Dotonbori and Kuromon Market.
 - Local dishes include sushi in Tokyo, Okonomiyaki in Hiroshima, and Ramen in Fukuoka.
2. **Seasonal Food Events:**
 - Summer festivals often feature street food like takoyaki and yakisoba.
 - July is a peak time for participation in **Kaiseki** dinners in Kyoto.
Transport and Accommodation:
1. **Japan Rail Pass:**
 - Ideal for cost-effective travel across cities. Valid on Shinkansen bullet trains and other major rail
lines, benefiting tourists traveling extensively.
2. **Accommodation for Solo Travelers:**
 - Hostels in Tokyo and Kyoto are abundant. Ryokan stays recommended for an authentic
experience, particularly outside major metros.
Events and Safety Tips:
1. **Local Events:**
 - July features various cultural and fireworks festivals such as the Tenjin Matsuri and Sumidagawa
Fireworks Festival.
 - Visit during EXPO 2025 for a mix of cultural exhibits and technology showcases.
2. **Travel and Safety Tips:**
 - Respect local traditions like removing shoes when entering homes and maintaining quietness on
public transport.
 - Major cities considered safe for solo travelers; keep personal belongings secure in crowded
places.
Itinerary Planning:
 - Allocate multiple days in Tokyo and Kyoto to explore diverse cultural offerings and cuisines.
 - Plan excursions to nearby historic sites and relax in traditional onsens (hot springs) in places like
Hakone.
 - Reserve some days for events in Osaka or other cities, depending on festival timings.
With this extensive research, crafting an engaging and culturally enriching 10-day itinerary for a solo
traveler in Japan becomes more effective. This approach ensures a blend of Japan?s food culture
and rich historical legacy.

**Visuals**:
-   Cultural and Culinary Highlights of a Solo 10-Day Trip to Japan:

-   (image goes here)

 Description: A colorful illustration showcasing iconic Japanese landmarks, food markets, and
cultural practices intertwined with efficient travel elements, reflecting a vibrant Japan experience.

**Conclusion**: The meticulously crafted itinerary provides travelers a deep dive into Japan's heritage, with ample
time to savor its diverse cuisine and partake in local festivals. For the solo adventurer, it balances
exploration, relaxation, and cultural immersion, making it an invaluable guide. Further refinement
may involve occasional updates for new events or travel advisories, ensuring the itinerary remains
relevant and enriching.

**References**:
- https://voyapon.com/best-events-japan-july-2025/
- https://japancheapo.com/events/july/
- https://www.magical-trip.com/media/osaka-in-july-2025-highlights-events-festivals/
- https://en.japantravel.com/events
- https://en.japantravel.com/event?p=74

## Future Enhancements

-   Integration with more data sources and APIs.
-   Advanced data analysis and charting capabilities within the Visualizer agent.
-   More robust error handling and state management.
-   Deployment to a cloud platform (e.g., Streamlit Cloud, AWS, GCP).
-   Caching of research results to speed up similar queries.


