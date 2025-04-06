# llm_interface.py
import google.generativeai as genai
import logging
import re
import config # <-- ADD THIS IMPORT

# Import specific constants if preferred, but importing the module is simpler here
# from config import (
#     INTERVIEWER_LLM_MODEL_NAME, EVALUATOR_LLM_MODEL_NAME,
#     INTERVIEWER_MAX_TOKENS, INTERVIEWER_TEMPERATURE,
#     EVALUATOR_MAX_TOKENS, EVALUATOR_TEMPERATURE, NUM_QUESTIONS
# )

logger = logging.getLogger(__name__)

# Global cache for models
llm_models = {}

def initialize_llm(model_name):
    """Initializes and returns a Gemini GenerativeModel instance."""
    global llm_models
    if model_name not in llm_models:
        try:
            # Use model_name directly from config constants passed to functions
            model = genai.GenerativeModel(model_name)
            llm_models[model_name] = model
            logger.info(f"Initialized LLM Model: {model_name}")
            return model
        except Exception as model_init_err:
            logger.error(f"Failed to initialize Gemini model '{model_name}': {model_init_err}")
            raise RuntimeError(f"LLM Model '{model_name}' initialization failed") from model_init_err
    return llm_models[model_name]

def get_interviewer_model():
    """Gets the initialized interviewer model."""
    # Use constant from config directly
    return initialize_llm(config.INTERVIEWER_LLM_MODEL_NAME)

def get_evaluator_model():
    """Gets the initialized evaluator model."""
    # Use constant from config directly
    return initialize_llm(config.EVALUATOR_LLM_MODEL_NAME)


def query_llm(prompt, model_name, max_tokens, temperature):
    """Sends prompt to the specified Gemini LLM and gets response."""
    logger.info(f"Sending prompt to {model_name} (length: {len(prompt)} chars)...")
    try:
        model = initialize_llm(model_name) # Get or initialize model
    except RuntimeError as e:
        return f"Error initializing LLM: {e}"

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        # top_p=0.9 # Keep default or adjust if needed
    )
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    try:
        if len(prompt) > 900000:
            logger.warning(f"Prompt length ({len(prompt)}) nearing model limit. Truncating slightly.")
            prompt = prompt[:900000]

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
            )

        # --- Enhanced Response Handling ---
        try:
            generated_text = response.text # Preferred way
            if not generated_text.strip(): # Check if text is just whitespace
                 raise ValueError("Empty text content in response.")
        except ValueError: # Catches blocked/empty responses
             logger.warning("response.text failed or empty. Checking candidates/feedback.")
             if not response.candidates:
                 block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown reason (no candidates)')
                 logger.error(f"Gemini response blocked or empty. Block Reason: {block_reason}")
                 return f"Error: LLM response was blocked or empty. Reason: {block_reason}."
             try: # Try parts if text failed but candidates exist
                 generated_text = "".join(part.text for part in response.candidates[0].content.parts)
                 if not generated_text.strip(): raise ValueError("Empty parts")
             except (AttributeError, ValueError, IndexError, TypeError): # Catch errors accessing parts
                 finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown reason')
                 safety_info = getattr(response.candidates[0], 'safety_ratings', 'N/A')
                 logger.error(f"Gemini response blocked or empty. Finish Reason: {finish_reason}. Safety: {safety_info}")
                 return f"Error: LLM response was blocked or empty. Reason: {finish_reason}."
        # --- End Enhanced Response Handling ---

        logger.info(f"Received response from {model_name}.")
        return generated_text.strip()

    except Exception as e:
        logger.error(f"Error during {model_name} inference: {e}")
        # Consider more specific error catching (e.g., API key errors)
        return f"Error generating response from {model_name}: {e}"

def clean_llm_output(text, is_evaluation=False):
    """Cleans common artifacts from LLM responses."""
    if not text or not isinstance(text, str): return ""

    text = text.strip()
    response_lines = text.splitlines()
    cleaned_response = []
    found_content = False # Flag to track if main content started

    # Phrases to skip (more comprehensive)
    # Now 'config' is available via import
    skip_phrases_questions = (f"generate {config.NUM_QUESTIONS} interview questions:", f"here are {config.NUM_QUESTIONS}", "**answer key", "answer:", "note:", "remember to tailor")
    skip_phrases_evaluation = ("evaluation:", "here is the evaluation", "strengths:", "weaknesses:", "constructive criticism:", "overall assessment:", "rating:")
    q_type_tags = ["[Technical/Conceptual]", "[Database Concept]", "[Database Administration]", "[Database Concept/Administration]", "[SQL Query]", "[SQL Query Writing]", "[SQL Query (Advanced)]", "[Troubleshooting/Problem Solving]", "[Troubleshooting]", "[Troubleshooting/Performance]", "[Behavioral/Learning]", "[Coding/Algorithmic]", "[System Design]", "[Security Concept]", "[Cloud Concept (if relevant)]", "[Behavioral/Teamwork]", "[Behavioral/Problem Solving]", "[DB Concept]", "[DB Admin Task]", "[DB Design/Schema]", "[DB Scenario]", "[Performance Scenario]", "[Security Scenario]", "[Cloud Scenario]", "[Learning Scenario]", "[Backup/Recovery Scenario]"]

    skip_phrases = skip_phrases_evaluation if is_evaluation else skip_phrases_questions

    for line in response_lines:
         line_strip = line.strip()
         line_lower = line_strip.lower()

         # Skip preamble/postamble lines
         if any(line_lower.startswith(phrase) for phrase in skip_phrases):
              found_content = True # Content likely starts after these
              continue

         # Skip empty lines, especially after finding content
         if not line_strip:
             if found_content: continue # Skip empty lines within the main response block
             else: pass # Allow leading empty lines if needed

         # Check if it looks like a question number or bullet (only for non-eval)
         is_question_line = not is_evaluation and re.match(r"^\s*(\d{1,2}\.|\[|\*|\-|\•)", line_strip)

         if is_question_line:
             found_content = True
             # Skip tag-only lines
             if line_strip in q_type_tags: continue
             # Remove tags from actual question
             for tag in q_type_tags:
                 if line_strip.startswith(tag):
                     line_strip = line_strip[len(tag):].strip()
                     break
             if line_strip: cleaned_response.append(line_strip)

         # For evaluation or lines after questions started, keep non-empty lines
         elif line_strip:
              found_content = True
              cleaned_response.append(line_strip)

    # Join lines and remove potential excessive newlines
    final_text = "\n".join(cleaned_response)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text) # Consolidate multiple blank lines
    return final_text.strip()