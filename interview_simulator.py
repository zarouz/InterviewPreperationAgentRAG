# interview_simulator.py
import os
import re
import logging
import google.generativeai as genai
import audio_utils
# Import components from other modules
import utils
import llm_interface
import prompt_templates
import report_generator
import config
# import confidence_analyzer # <-- REMOVED THIS LINE

# +++ ADD THESE IMPORTS +++
import requests
import json
# +++++++++++++++++++++++++

# Setup basic logging (adjust level and format as needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# +++ ADD API URL CONSTANT +++
# Make sure the port matches the one in emotion_api.py (default 5001)
# Use http://<api-service-ip>:5001/analyze if running on different machines/containers
EMOTION_API_URL = os.environ.get("EMOTION_API_ENDPOINT", "http://127.0.0.1:5001/analyze")
logger.info(f"Using Emotion Analysis API endpoint: {EMOTION_API_URL}")
# ++++++++++++++++++++++++++++

# --- Helper Function Definitions ---

def parse_generated_questions(raw_text):
    """Parses the LLM response to extract a list of questions."""
    # (Using the refined version from previous steps - unchanged)
    questions = []
    pattern = re.compile(r"^\s*(?:\[.*?\])?\s*(?:\d{1,2}\.|[-*•])\s*(.*)")
    lines = raw_text.strip().split('\n')
    q_type_tags_to_remove = [
        "[Technical/Conceptual]", "[Database Concept]", "[Database Administration]",
        "[Database Concept/Administration]", "[SQL Query]", "[SQL Query Writing]",
        "[SQL Query (Advanced)]", "[Troubleshooting/Problem Solving]", "[Troubleshooting]",
        "[Troubleshooting/Performance]", "[Behavioral/Learning]", "[Coding/Algorithmic]",
        "[System Design]", "[Security Concept]", "[Cloud Concept (if relevant)]",
        "[Behavioral/Teamwork]", "[Behavioral/Problem Solving]", "[DB Concept]",
        "[DB Admin Task]", "[DB Design/Schema]", "[DB Scenario]", "[Performance Scenario]",
        "[Security Scenario]", "[Cloud Scenario]", "[Learning Scenario]", "[Backup/Recovery Scenario]"
    ]

    for line in lines:
        match = pattern.match(line)
        if match:
            question_text = match.group(1).strip()
            original_text = question_text
            for tag in q_type_tags_to_remove:
                tag_pattern = r'^\s*' + re.escape(tag) + r'\s*:?\s*'
                if re.match(tag_pattern, question_text, re.IGNORECASE):
                    question_text = re.sub(tag_pattern, '', question_text, count=1, flags=re.IGNORECASE).strip()
                    break

            if question_text and len(question_text) > 10:
                 questions.append(question_text)
            elif original_text and len(original_text) > 10 and question_text != original_text:
                 if len(question_text) > 5:
                     questions.append(question_text)
                 else:
                      questions.append(original_text)
            elif original_text and len(original_text) > 10:
                 questions.append(original_text)

    if not questions and len(lines) > 1:
         logger.warning("Regex failed to parse questions, using basic newline split fallback.")
         questions = [l.strip() for l in lines if l.strip() and len(l.strip()) > 15 and not l.strip().lower().startswith(("okay", "great", "thanks", "sure", "understood", "evaluation:", "alignment", "technical accuracy", "relevance", "strengths:", "areas for improvement", "overall score"))]

    questions = [q for q in questions if q] # Final filter for empty strings
    logger.info(f"Parsed {len(questions)} potential questions.")
    return questions

def format_conversation_history(history):
    """Formats the history list into a string for the prompt."""
    # (Unchanged)
    if not history: return "N/A"
    formatted = ""
    for turn in history:
        speaker = turn.get('speaker', 'Unknown')
        text = turn.get('text', '')
        formatted += f"**{speaker}:** {text}\n\n"
    return formatted.strip()

# --- Main Simulation Function ---

def run_simulation():
    """Orchestrates the interview simulation process with audio input and API-based confidence analysis."""
    logger.info("--- Starting Interview Simulation (with API Confidence Analysis) ---")

    # --- REMOVED Model Loading ---
    # The local loading of the emotion model is no longer needed here.
    # The API service handles its own model loading.
    # --------------------------

    # --- Initialize Variables ---
    interview_data = []
    role_title = "Default Role" # Will be updated from JD
    resume_text_raw = ""
    jd_text_raw = ""
    conversation_history = []
    interview_qna = []
    asked_questions_indices = set()

    try:
        # --- 1. Load Inputs ---
        logger.info("Loading input files (Resume and Job Description)...")
        try:
            resume_text_raw = utils.extract_text_from_pdf(config.DEFAULT_RESUME_PATH)
            with open(config.DEFAULT_JD_PATH, 'r', encoding='utf-8') as f:
                jd_text_raw = f.read()
            logger.info("Input files loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Input file not found: {e}")
            print(f"Error: Input file not found - {e}. Please check paths in config.py.")
            return
        except Exception as e:
            logger.error(f"Error loading input files: {e}", exc_info=True)
            print(f"Error loading input files: {e}")
            return

        resume_summary = re.sub(r'\s+', ' ', resume_text_raw).strip()[:config.MAX_SUMMARY_LENGTH]
        jd_summary = re.sub(r'\s+', ' ', jd_text_raw).strip()[:config.MAX_SUMMARY_LENGTH]

        # --- 2. Prepare Context & Initial Questions ---
        logger.info("Preparing context and generating initial questions...")
        jd_skills = utils.extract_skills_from_text(jd_text_raw)
        focus_topics = utils.get_focus_topics(resume_text_raw, jd_text_raw)

        role_title_match = re.search(r"^(?:Job\s+)?Title\s*:\s*(.*?)(\n|$)", jd_text_raw, re.IGNORECASE | re.MULTILINE)
        role_title = role_title_match.group(1).strip() if role_title_match else "Relevant Role (Not Found in JD)"
        if role_title == "Relevant Role (Not Found in JD)":
             logger.warning("Could not find 'Job Title:' in JD, using default 'Software Engineer'.")
             role_title = "Software Engineer"
        logger.info(f"Running simulation for Role: {role_title}")

        search_queries = utils.generate_search_queries(resume_text_raw, jd_text_raw, num_queries=config.NUM_QUESTIONS + 1)
        rag_context = "No specific context retrieved."
        if search_queries:
            all_retrieved_docs = []
            logger.info(f"Retrieving context for {len(search_queries)} queries...")
            for query in search_queries:
                docs = utils.retrieve_similar_documents(query)
                if docs: all_retrieved_docs.extend(docs)
            if all_retrieved_docs:
                unique_docs_dict = {doc['content']: doc for doc in all_retrieved_docs}
                unique_docs = list(unique_docs_dict.values())
                logger.info(f"Total unique documents retrieved: {len(unique_docs)}")
                sorted_unique_docs = sorted(unique_docs, key=lambda x: x.get("score", 0), reverse=True)
                rag_context = utils.format_rag_context(sorted_unique_docs)
            else:
                 logger.warning("No documents retrieved from knowledge base.")
        else:
            logger.warning("No search queries generated. Proceeding without RAG context.")

        if "database admin" in role_title.lower() or "dba" in role_title.lower() or "database" in role_title.lower():
            q_types_list = ["[DB Concept/Scenario]", "[SQL Query (Scenario)]", "[Troubleshooting Scenario]", "[DB Admin Task/Scenario]", "[Security Scenario]", "[Behavioral/Learning Scenario]"]
            role_guidance = {"role": "Probe core DB concepts.", "code": "Focus on practical SQL.", "solve": "Present common DBA challenges."}
        elif "software engineer" in role_title.lower() or "developer" in role_title.lower():
             q_types_list = ["[Technical Concept/Tradeoff]", "[Coding Challenge (Scenario)]", "[System Design (Scenario)]", "[Debugging Scenario]", "[Behavioral Scenario (Teamwork)]", "[Behavioral Scenario (Learning)]"]
             role_guidance = {"role": "Assess CS fundamentals.", "code": "Provide small coding problems.", "solve": "Debugging/design scenarios."}
        else:
             q_types_list = ["[Technical Scenario]", "[Problem Solving Scenario]", "[Tool/Concept Question]", "[Design Question]", "[Behavioral Question]", "[Learning Question]"]
             role_guidance = {"role": "Focus on general tech concepts.", "code": "Ask about high-level logic.", "solve": "Present general tech challenges."}
             logger.info("Using default question types and guidance for unrecognized role.")

        qg_prompt_args = {
            "role_title": role_title,
            "resume_summary": resume_summary if resume_summary else "Not provided.",
            "jd_summary": jd_summary if jd_summary else "Not provided.",
            "focus_str": ', '.join(focus_topics) if focus_topics else f'General skills for {role_title}',
            "context_str": rag_context,
            "num_questions": config.NUM_QUESTIONS,
            "role_specific_guidance": role_guidance["role"],
            "coding_guidance": role_guidance["code"],
            "problem_solving_guidance": role_guidance["solve"],
            "extra_hints_str": "",
        }
        for i in range(config.NUM_QUESTIONS):
             qg_prompt_args[f"q_type_{i}"] = q_types_list[i % len(q_types_list)]

        logger.info("Generating initial interview questions via LLM...")
        question_gen_prompt = prompt_templates.QUESTION_GENERATION_PROMPT_TEMPLATE.format(**qg_prompt_args)
        raw_questions_text = llm_interface.query_llm(
            question_gen_prompt, config.INTERVIEWER_LLM_MODEL_NAME,
            max(config.INTERVIEWER_MAX_TOKENS * 2, 1000), config.INTERVIEWER_TEMPERATURE
        )

        if raw_questions_text.startswith("Error:"):
            logger.error(f"Failed to generate initial questions: {raw_questions_text}")
            print(f"\nError: Failed to generate initial interview questions from LLM.\nDetails: {raw_questions_text}")
            return

        cleaned_questions_text = llm_interface.clean_llm_output(raw_questions_text)
        prepared_questions = parse_generated_questions(cleaned_questions_text)

        if not prepared_questions or len(prepared_questions) < config.NUM_QUESTIONS // 2:
            logger.error(f"Failed to parse generated questions reliably after cleaning. Have {len(prepared_questions)} questions.")
            print(f"\nError: Could not reliably parse the generated interview questions (found {len(prepared_questions)}). Target was {config.NUM_QUESTIONS}.")
            return

        prepared_questions = prepared_questions[:config.NUM_QUESTIONS]
        logger.info(f"Successfully generated and parsed {len(prepared_questions)} questions.")
        print(f"\n--- Prepared {len(prepared_questions)} Interview Questions ---")
        for i, q in enumerate(prepared_questions): print(f"{i+1}. {q}")
        print("------------------------------------")


        # --- 3. Conduct Interview Loop ---
        print("\n--- Starting Simulated Interview ---")
        print(f"AI Interviewer: {config.INTERVIEWER_AI_NAME}")
        candidate_name_display = config.CANDIDATE_NAME if config.CANDIDATE_NAME else "Candidate"
        print(f"Candidate: {candidate_name_display}")
        print("\nInstructions:")
        print("  - When prompted, press [Enter] to START recording your audio response.")
        print("  - Speak clearly into your microphone.")
        print("  - Press [Enter] again when you are finished speaking to STOP recording.")
        print("  - Alternatively, type 'quit' (and press Enter) instead of recording to end the interview.")
        print("------------------------------------\n")

        current_turn = 0
        while len(asked_questions_indices) < len(prepared_questions):
            current_turn += 1
            logger.info(f"Starting interview turn {current_turn}")

            remaining_indices = [i for i in range(len(prepared_questions)) if i not in asked_questions_indices]
            if not remaining_indices:
                 logger.info("All prepared questions have been asked.")
                 break

            asked_str = "None yet" if not asked_questions_indices else ", ".join(str(i+1) for i in sorted(list(asked_questions_indices)))
            remaining_str = ", ".join(str(i+1) for i in remaining_indices)
            history_str = format_conversation_history(conversation_history)
            conv_prompt_args = {
                "interviewer_name": config.INTERVIEWER_AI_NAME,
                "company_name": config.COMPANY_NAME,
                "role_title": role_title,
                "candidate_name": candidate_name_display,
                "resume_summary": resume_summary if resume_summary else "Not provided.",
                "jd_summary": jd_summary if jd_summary else "Not provided.",
                "focus_topics_str": ', '.join(focus_topics) if focus_topics else f'General skills for {role_title}',
                "prepared_questions_numbered": "\n".join(f"{i+1}. {q}" for i, q in enumerate(prepared_questions)),
                "asked_questions_str": asked_str,
                "remaining_questions_str": remaining_str,
                "conversation_history": history_str
            }
            interview_turn_prompt = prompt_templates.CONVERSATIONAL_INTERVIEW_PROMPT_TEMPLATE.format(**conv_prompt_args)

            ai_response_raw = llm_interface.query_llm(
                interview_turn_prompt, config.INTERVIEWER_LLM_MODEL_NAME,
                config.INTERVIEWER_MAX_TOKENS, config.INTERVIEWER_TEMPERATURE
            )
            ai_response = llm_interface.clean_llm_output(ai_response_raw)

            if ai_response.startswith("Error:"):
                logger.error(f"Interviewer LLM failed: {ai_response}")
                print(f"\nSYSTEM: Interviewer AI encountered an error: {ai_response}")
                break

            # --- Robust Question Detection Logic (Unchanged) ---
            asked_index = -1
            ai_question_text = ""
            detected_method = "None"
            for i in remaining_indices:
                 q_text = prepared_questions[i]
                 q_text_short_for_regex = q_text[:max(60, len(q_text)//3)]
                 number_pattern = r'(?:^|\s|\*)\s*' + str(i+1) + r'\.\s+'
                 number_match = re.search(number_pattern, ai_response)
                 regex_match = None
                 if len(q_text_short_for_regex) > 15:
                     try:
                         pattern = r'(?i)\b' + re.escape(q_text_short_for_regex)
                         regex_match = re.search(pattern, ai_response)
                     except re.error as e: logger.warning(f"Regex error checking for Q{i+1}: {e}")

                 is_match = False
                 if regex_match:
                     is_match = True; detected_method = "Regex"
                 elif number_match and len(ai_response) > 30:
                     is_match = True; detected_method = "Number"

                 if is_match:
                     if asked_index == -1:
                         asked_index = i; ai_question_text = q_text
                         logger.info(f"AI asking question {i+1}: '{ai_question_text[:50]}...' (Detected via {detected_method})")
                         break
                     else:
                         logger.warning(f"Multiple potential Q matches ({asked_index+1} vs {i+1}). Sticking with first: Q{asked_index+1}.")
                         break
            if asked_index == -1 and remaining_indices:
                 asked_index = remaining_indices[0]; ai_question_text = prepared_questions[asked_index]
                 detected_method = "Fallback (Assume Next)"
                 logger.warning(f"Could not reliably detect question, {detected_method}: Q{asked_index + 1}")
                 q_text_short_check = ai_question_text[:max(40, len(ai_question_text)//4)]
                 prepended_flag = False
                 try:
                    pattern_check = r'(?i)\b' + re.escape(q_text_short_check)
                    if not re.search(pattern_check, ai_response):
                         logger.info(f"Prepending assumed question Q{asked_index + 1}...")
                         ai_response = f"{ai_response}\n\n[System: Moving to Question {asked_index + 1}] {ai_question_text}"
                         prepended_flag = True
                    else: logger.info(f"Assumed Q{asked_index + 1} text partially present, not prepending.")
                 except re.error as fallback_re_err:
                     logger.warning(f"Regex failed during fallback check ({fallback_re_err}), prepending...")
                     ai_response = f"{ai_response}\n\n[System: Moving to Question {asked_index + 1}] {ai_question_text}"
                     prepended_flag = True
                 if prepended_flag: logger.debug(f"AI response after fallback prepend: {ai_response}")
            # --- End Detection Logic ---

            conversation_history.append({"speaker": config.INTERVIEWER_AI_NAME, "text": ai_response})
            print(f"\n{config.INTERVIEWER_AI_NAME}: {ai_response}") # Print AI turn

            # --- Get Candidate Response (Audio + STT + API Confidence) ---
            if asked_index != -1:
                # Define unique temp filename for this turn in a writable location
                # Consider using tempfile module for robustness
                temp_dir = os.path.join(os.getcwd(), "temp_audio") # Create a sub-directory
                os.makedirs(temp_dir, exist_ok=True)
                temp_audio_filename = os.path.join(temp_dir, f"temp_response_q{asked_index + 1}_{current_turn}.wav")

                recorded_audio_path = audio_utils.record_audio_interactive(filename=temp_audio_filename)

                candidate_response_text = ""
                # Default error state for confidence results from API
                confidence_results = {
                    'score': None, 'rating': "N/A", 'primary_emotion': "N/A",
                    'emotion_confidence': None, 'segment_results': [],
                    'all_probs': None, 'error': True, 'message': 'Analysis not performed'
                }

                # Proceed only if audio was successfully recorded
                if recorded_audio_path:
                    logger.info(f"Attempting to transcribe audio from: {recorded_audio_path}")
                    candidate_response_text = audio_utils.transcribe_audio(recorded_audio_path)

                    # Check transcription result
                    if not candidate_response_text or candidate_response_text.startswith("[Error") or candidate_response_text == "[Audio unintelligible or empty]":
                        logger.warning(f"Transcription failed or unusable for {recorded_audio_path}. Text: '{candidate_response_text}'")
                        if not candidate_response_text: candidate_response_text = "[Audio Recorded - Transcription Failed]"
                        print(f"System: Transcription may have failed. Proceeding...")
                    else:
                        print(f"\n{candidate_name_display} (Transcribed): {candidate_response_text}")

                    # --- Call Emotion Analysis API ---
                    logger.info(f"Calling Emotion Analysis API for: {recorded_audio_path}")
                    # Sending absolute path is safer if API service might run from different working directory
                    abs_audio_path = os.path.abspath(recorded_audio_path)
                    api_payload = {'audio_path': abs_audio_path}
                    try:
                        # Make the API request
                        api_response = requests.post(EMOTION_API_URL, json=api_payload, timeout=30) # 30-second timeout

                        # Check for HTTP errors (like 404, 500)
                        api_response.raise_for_status()

                        # Get JSON results from API response
                        confidence_results = api_response.json()
                        logger.info(f"API Analysis completed. API Error Status: {confidence_results.get('error')}, Message: {confidence_results.get('message', 'N/A')}")

                        if confidence_results.get('error'):
                             logger.warning(f"API reported an analysis error: {confidence_results.get('message', 'Unknown error')}")
                             print(f"System: Confidence analysis via API reported an error: {confidence_results.get('message', 'Unknown error')}")
                        else:
                            # Log successful analysis details
                            logger.info(f"API Confidence analysis successful: Score={confidence_results.get('score'):.1f}, Rating={confidence_results.get('rating')}")

                    # Handle potential errors during the API call
                    except requests.exceptions.ConnectionError as conn_err:
                        logger.error(f"API Connection Error: Could not connect to {EMOTION_API_URL}. Is the API service running? Details: {conn_err}")
                        print(f"System Warning: Could not connect to the Emotion Analysis API. Skipping confidence analysis.")
                        confidence_results['message'] = 'API Connection Error' # Update default message
                    except requests.exceptions.Timeout:
                        logger.error(f"API Timeout: Request to {EMOTION_API_URL} timed out.")
                        print(f"System Warning: Emotion Analysis API request timed out. Skipping confidence analysis.")
                        confidence_results['message'] = 'API Timeout' # Update default message
                    except requests.exceptions.RequestException as req_err:
                        status_code = getattr(req_err.response, 'status_code', 'N/A')
                        response_text = getattr(req_err.response, 'text', 'N/A')
                        logger.error(f"API Request Error: Call to {EMOTION_API_URL} failed. Status Code: {status_code}. Response: {response_text}. Details: {req_err}")
                        print(f"System Warning: Error calling Emotion Analysis API (Status: {status_code}). Skipping confidence analysis.")
                        confidence_results['message'] = f'API Request Error: Status {status_code}' # Update default message
                    except json.JSONDecodeError:
                         logger.error(f"API Response Error: Could not decode JSON response from {EMOTION_API_URL}.")
                         print(f"System Warning: Invalid response from Emotion Analysis API. Skipping confidence analysis.")
                         confidence_results['message'] = 'API Invalid JSON Response' # Update default message
                    # --- End API Call ---

                    # Clean up temp audio file after transcription and API call
                    try:
                        os.remove(recorded_audio_path)
                        logger.debug(f"Removed temporary audio file: {recorded_audio_path}")
                    except OSError as e:
                        logger.warning(f"Could not remove temp audio file {recorded_audio_path}: {e}")

                else: # Recording failed
                    candidate_response_text = "[Audio Recording Failed]"
                    confidence_results['message'] = 'Audio Recording Failed' # Ensure message reflects this
                    print(f"\n{candidate_name_display}: [Audio Recording Failed]")
                    logger.error(f"Audio recording failed for question {asked_index + 1}")

                # Allow user to quit via text even if audio/API failed
                if candidate_response_text.strip().lower() == 'quit':
                     print("\nEnding interview by user request ('quit' detected).")
                     break

                # Store Q&A pair including text and confidence results *from API*
                interview_qna.append({
                    "question_index": asked_index + 1,
                    "question": ai_question_text,
                    "response": candidate_response_text, # Store transcribed text (or error placeholder)
                    "detection_method": detected_method,
                    # Use results obtained from the API call (or the default error values if API failed)
                    "confidence_score": confidence_results.get('score'),
                    "confidence_rating": confidence_results.get('rating', 'N/A'), # Get rating directly
                    "primary_emotion": confidence_results.get('primary_emotion', 'N/A'),
                    "confidence_analysis_error": confidence_results.get('error', True) # Reflect API error status
                })

                # Append candidate *text* turn to history for LLM context
                conversation_history.append({"speaker": candidate_name_display, "text": candidate_response_text})
                asked_questions_indices.add(asked_index) # Mark as asked

            else: # Should not happen if fallback works
                logger.error("No question index identified this turn. Ending loop.")
                print("SYSTEM: Internal error determining question. Ending interview.")
                break
            # --- End Candidate Response Block ---

        # --- End Interview Loop ---
        logger.info(f"Interview conversation finished. {len(asked_questions_indices)} questions attempted.")
        print("\n--- Interview Complete ---")

        # --- 4. Deferred Evaluation Phase ---
        if not interview_qna:
             logger.warning("No Q&A pairs recorded. Skipping evaluation.")
             print("\nNo questions were answered. Skipping evaluation.")
        else:
            logger.info(f"Starting evaluation of {len(interview_qna)} Q&A pairs...")
            print(f"\n--- Evaluating {len(interview_qna)} Responses ---")
            evaluated_data = [] # This will hold the combined Q&A, confidence, and text eval data
            for item in interview_qna: # item already contains Q, A, and confidence results from API
                q_idx = item["question_index"]
                q_text = item["question"]
                c_response_text = item["response"]

                # Skip text evaluation if response indicates a recording/transcription failure
                if c_response_text.startswith("[Audio") or c_response_text.startswith("[STT"):
                     logger.warning(f"Skipping text evaluation for Q{q_idx} due to recording/transcription failure.")
                     item["evaluation"] = "Evaluation skipped (Audio/STT failed)"
                     item["score"] = None
                else:
                     # Perform text evaluation using LLM
                     logger.info(f"Evaluating response text for question {q_idx}")
                     print(f"Evaluating Q{q_idx} (Text Response)...")
                     eval_prompt_args = {
                         "role_title": role_title,
                         "jd_summary": jd_summary if jd_summary else "Not provided.",
                         "resume_summary": resume_summary if resume_summary else "Not provided.",
                         "interview_question": q_text,
                         "candidate_response": c_response_text
                     }
                     evaluation_prompt = prompt_templates.EVALUATION_PROMPT_TEMPLATE.format(**eval_prompt_args)
                     evaluation_raw = llm_interface.query_llm(
                         evaluation_prompt, config.EVALUATOR_LLM_MODEL_NAME,
                         config.EVALUATOR_MAX_TOKENS, config.EVALUATOR_TEMPERATURE
                     )
                     evaluation = llm_interface.clean_llm_output(evaluation_raw, is_evaluation=True)

                     if evaluation.startswith("Error:"):
                         logger.error(f"Evaluator LLM failed for Q{q_idx}: {evaluation}")
                         item["evaluation"] = f"Evaluation Error: {evaluation}"
                         item["score"] = None
                     else:
                         item["evaluation"] = evaluation # Add evaluation text to the item
                         score_match = re.search(r"Overall Score\s*\(1-5\):\s*([1-5])", evaluation, re.IGNORECASE)
                         item["score"] = int(score_match.group(1)) if score_match else None # Add score to item
                         if not score_match: logger.warning(f"Could not parse text eval score for Q{q_idx}")

                # Add the item (now containing Q, A, confidence, text eval) to evaluated_data
                evaluated_data.append(item)

            logger.info("Evaluation phase complete.")
            print("-----------------------")

            # --- 5. Generate Report ---
            if evaluated_data:
                try:
                    report_filename = config.REPORT_FILENAME
                    logger.info(f"Generating PDF report with confidence data from API: {report_filename}")
                    # Pass the evaluated_data list, which now contains all info needed by the report generator
                    report_generator.generate_pdf_report(
                        evaluated_data=evaluated_data,
                        resume_text=resume_text_raw,
                        jd_text=jd_text_raw,
                        role_title=role_title,
                        report_filename=report_filename
                    )
                    print(f"\nPDF report generated: {report_filename}")
                except ImportError:
                     logger.error("ReportLab not found. Skipping PDF report generation.", exc_info=True)
                     print("\nPDF Report Generation Skipped: ReportLab library not found (`pip install reportlab`).")
                except Exception as report_err:
                     logger.error(f"Failed to generate PDF report: {report_err}", exc_info=True)
                     print(f"\nError generating PDF report: {report_err}")
            else:
                print("\nNo evaluated data available to generate a report.")


    # --- Error Handling & Finalization ---
    except RuntimeError as rte:
        logger.critical(f"A critical runtime error occurred: {rte}", exc_info=True)
        print(f"\nCritical Error: {rte}")
    except KeyboardInterrupt:
         logger.warning("Interview simulation interrupted by user (Ctrl+C).")
         print("\nInterview simulation stopped by user.")
    except Exception as main_err:
        logger.exception(f"An unexpected error occurred during interview simulation: {main_err}")
        print(f"\nAn unexpected error occurred: {main_err}")
    finally:
        # Clean up temp audio directory if it exists and is empty? Optional.
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        if os.path.exists(temp_dir):
            try:
                if not os.listdir(temp_dir): # Only remove if empty
                    os.rmdir(temp_dir)
                    logger.info(f"Removed empty temporary audio directory: {temp_dir}")
            except OSError as e:
                logger.warning(f"Could not remove temp audio directory {temp_dir}: {e}")
        logger.info("--- Interview Simulation Finished ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Prerequisite Checks & Setup
    utils.download_nltk_data('stopwords', 'corpora')
    utils.download_nltk_data('punkt', 'tokenizers')

    errors = []
    # Check essential files/configs (Removed emotion model check)
    if not os.path.exists(config.DEFAULT_RESUME_PATH): errors.append(f"Resume file not found: '{config.DEFAULT_RESUME_PATH}'")
    if not os.path.exists(config.DEFAULT_JD_PATH): errors.append(f"JD file not found: '{config.DEFAULT_JD_PATH}'")
    if not config.GOOGLE_API_KEY: errors.append("GOOGLE_API_KEY missing in config/.env.")
    # Emotion model check removed as it's handled by the API service now.

    # Check for audio device (basic check)
    try:
        import sounddevice as sd
        if len(sd.query_devices(kind='input')) == 0:
             errors.append("No input audio device (microphone) found.")
    except Exception as audio_err:
         errors.append(f"Error checking audio devices: {audio_err}")

    if errors:
        print("Error: Setup failed due to missing configuration, files, or devices:")
        for err in errors: print(f"- {err}")
        exit(1)

    # Configure Google API
    try:
        api_key = config.GOOGLE_API_KEY
        if not api_key: raise ValueError("GOOGLE_API_KEY is empty.")
        genai.configure(api_key=api_key)
        logger.info("Google Generative AI SDK configured.")
    except Exception as api_err:
         logger.error(f"Failed to configure Google API: {api_err}", exc_info=True)
         print(f"Error: Failed to configure Google API. Check Key. Details: {api_err}")
         exit(1)

    # --- Run Simulation ---
    run_simulation()