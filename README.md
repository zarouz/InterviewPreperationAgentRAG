
## Prerequisites

*   **Python:** 3.10 or higher recommended.
*   **Git:** For cloning the repository.
*   **PostgreSQL Database:** Version 12+ recommended.
*   **pgvector Extension:** Needs to be installed in your PostgreSQL database for vector similarity search (RAG).
*   **Microphone:** Required for recording audio responses.
*   **Internet Connection:** Required for LLM API calls (Google Gemini), STT (Google Web Speech), and downloading models/packages.

## Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/zarouz/InterviewPreperationAgentRAG.git
    cd InterviewPreperationAgentRAG
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # On Windows use: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    *(You need to create a `requirements.txt` file first)*
    ```bash
    # Make sure pip is up-to-date
    pip install --upgrade pip
    # Create requirements.txt (if you haven't) e.g.: pip freeze > requirements.txt
    # OR Install necessary packages manually (see Technology Stack below)
    pip install -r requirements.txt
    ```
    **Important:** Ensure your environment matches the dependencies, especially handling the Keras/TensorFlow/Transformers versions (e.g., `tensorflow`, `tensorflow-macos`, `tensorflow-metal`, `keras`, `transformers`, `sentence-transformers`, `torch`, etc.).

4.  **Download NLTK Data:**
    The `utils.py` script attempts to download necessary NLTK data (`stopwords`, `punkt`) on first run if needed. Alternatively, run this in a Python interpreter within your venv:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

5.  **Database Setup (PostgreSQL + pgvector):**
    *   Install PostgreSQL if you haven't already.
    *   Create a database and a user for this project.
    *   Connect to your database (e.g., using `psql`).
    *   Install the pgvector extension: `CREATE EXTENSION vector;`
    *   *(Optional but Recommended):* Create a script (e.g., `populate_db.py`, not included here) to chunk documents and populate the `chunks` table in your database with text content and their vector embeddings using the chosen Sentence Transformer model. You will need a knowledge base of documents relevant to the interview topics.

6.  **Place the Emotion Model:**
    *   Download or ensure your pre-trained emotion recognition model file (`end_to_end_emotion_model.keras`) exists.
    *   Place it in the `emotions_model/` directory within the project, or update the `MODEL_PATH` in `confidence_analyzer.py` accordingly.

## Configuration

1.  **Environment Variables (`.env`):**
    *   Create a file named `.env` in the project's root directory.
    *   Add your API key and database credentials. **Do not commit this file to Git!** (`.gitignore` should prevent this).
    *   **Example `.env` content:**
        ```dotenv
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        DB_NAME="knowledgebase" # Your DB name
        DB_USER="your_db_user"
        DB_PASSWORD="your_db_password"
        DB_HOST="localhost" # Or your DB host
        ```

2.  **Configuration File (`config.py`):**
    *   Review the settings in `config.py`.
    *   Update `DEFAULT_RESUME_PATH` and `DEFAULT_JD_PATH` to point to your input files.
    *   Adjust LLM model names (`INTERVIEWER_LLM_MODEL_NAME`, `EVALUATOR_LLM_MODEL_NAME`), RAG parameters (`RETRIEVAL_TOP_K`, `RETRIEVAL_SIMILARITY_THRESHOLD`), or other settings as needed.

## Usage

1.  **Ensure Prerequisites & Configuration** are complete.
2.  **Activate the virtual environment:** `source .venv/bin/activate`
3.  **Run the simulator:**
    ```bash
    python interview_simulator.py
    ```
4.  **Follow the Console Prompts:**
    *   The script will load inputs, generate questions, and start the interview.
    *   When prompted by the AI Interviewer, the console will display `🎙️ Press [Enter] to START recording...`. Press Enter.
    *   It will then display `🔴 Recording... Press [Enter] again to STOP.`. Speak your answer clearly into the microphone.
    *   Press Enter again when finished speaking.
    *   The system will transcribe your audio, perform confidence analysis, and the AI interviewer will proceed.
    *   Type `quit` instead of recording to end the interview early.
5.  **Review Output:**
    *   Observe the interview flow in the console.
    *   After the interview concludes, the evaluation phase will run.
    *   Finally, check for the generated `interview_report.pdf` in the project directory.

## Output Report (`interview_report.pdf`)

The generated PDF report contains:

1.  **Title Page:** Candidate name, Role, Date.
2.  **Resume vs. JD Skill Analysis:** A table showing skills matched between the resume and JD, potential gaps (JD only), and skills present only in the resume.
3.  **Individual Question Evaluations:** For each question asked:
    *   The question text.
    *   The candidate's transcribed response.
    *   **Voice Confidence Analysis:** Overall score (%), rating (e.g., High, Low), and dominant aggregated emotion.
    *   **Text Response Evaluation:** LLM's assessment of alignment, technical accuracy, relevance, strengths, and areas for improvement, plus an overall score (1-5).
4.  **Overall Assessment Summary:**
    *   Average text evaluation score.
    *   Summary of key strengths and areas for improvement observed across all text responses.
    *   Average voice confidence score and rating across all analyzed responses.

## Technology Stack

*   **Programming Language:** Python 3
*   **LLMs:** Google Gemini (via `google-generativeai`)
*   **Embeddings:** Sentence Transformers (`sentence-transformers/all-mpnet-base-v2`)
*   **RAG Database:** PostgreSQL + pgvector extension
*   **PDF Parsing:** `pdfplumber`
*   **Audio Processing:** `sounddevice`, `soundfile`, `librosa`, `webrtcvad-wheels`
*   **Speech-to-Text:** `SpeechRecognition` (using Google Web Speech API)
*   **Emotion/Confidence Model:** TensorFlow / Keras
*   **PDF Generation:** `reportlab`
*   **Text Processing:** `nltk`
*   **Environment Management:** `venv`
*   **Database Connector:** `psycopg2-binary` (or `psycopg2`)

## Potential Improvements & Future Work

*   Implement a dedicated script (`populate_db.py`) for easier knowledge base creation.
*   Develop a more sophisticated RAG strategy (e.g., query expansion, reranking).
*   Improve error handling and user feedback during audio recording/STT.
*   Explore different STT providers (local options like Whisper, other cloud APIs).
*   Fine-tune the LLMs for better interview flow or evaluation.
*   Integrate different or fine-tuned emotion/confidence models.
*   Add more features to the confidence analysis (e.g., speaking rate, pause analysis).
*   Create a web interface (using Flask/Django/FastAPI) instead of a console application.
*   Refactor the emotion analysis into a separate microservice/API as discussed during debugging.
*   Add unit and integration tests.



