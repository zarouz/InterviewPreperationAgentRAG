# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- General ---
NUM_QUESTIONS = 6 # Target number of high-quality questions

# --- File Paths ---
# These can be overridden by command-line arguments if desired later
DEFAULT_RESUME_PATH = "/Users/karthikyadav/Desktop/Resume/resume_6 (1).pdf" # Your specific resume
DEFAULT_JD_PATH = "job_description_dbms.txt" # Junior DBA JD
REPORT_FILENAME = "interview_report.pdf" # <<< --- ADD THIS LINE HERE ---

# --- Database ---
DB_NAME = os.getenv("DB_NAME", "KnowledgeBase")
DB_USER = os.getenv("DB_USER", "karthikyadav")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")

# --- RAG ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
MAX_CONTEXT_LENGTH = 10000 # Max chars for formatted RAG context
RETRIEVAL_TOP_K = 6
RETRIEVAL_SIMILARITY_THRESHOLD = 0.58

# --- LLM Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Model for conducting the interview (needs good conversation flow)
INTERVIEWER_LLM_MODEL_NAME = "gemini-1.5-flash-latest"
# Model for evaluating responses (needs strong reasoning)
EVALUATOR_LLM_MODEL_NAME = "gemini-1.5-pro-latest" # Using Pro for better evaluation

# --- LLM Generation Parameters ---
INTERVIEWER_MAX_TOKENS = 500 # Tokens for AI interviewer's turn (question + brief ack)
INTERVIEWER_TEMPERATURE = 0.65
EVALUATOR_MAX_TOKENS = 700 # Allow more tokens for detailed evaluation
EVALUATOR_TEMPERATURE = 0.5 # Lower temp for more consistent evaluation

# --- Prompting Constants ---
MAX_SUMMARY_LENGTH = 1200 # Max chars for resume/JD summaries in prompts
MAX_TOTAL_PROMPT_CHARS = 100000 # Safety limit for Gemini prompt length

# --- Simulation ---
CANDIDATE_NAME = "Candidate" # Placeholder name
INTERVIEWER_AI_NAME = "Alexi" # Name for the AI interviewer
COMPANY_NAME = "SecureData Financial Corp." # Example, extract if needed