# audio_utils.py
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr_audio # Alias to avoid confusion
import queue
import threading
import time
import logging

logger = logging.getLogger(__name__)

# --- Audio Recording Configuration ---
# Global variables for recording control
recording_queue = queue.Queue()
stop_recording_event = threading.Event()
DEFAULT_AUDIO_SAMPLERATE = 16000

# --- Audio Callback (Internal) ---

def _audio_callback(indata, frames, time, status):
    """Internal callback function for sounddevice InputStream."""
    if status:
        # Log buffer overflows/underflows etc.
        logger.warning(f"Audio callback status: {status}")
    if not stop_recording_event.is_set():
        # Put the audio data block into the queue
        recording_queue.put(indata.copy())
    else:
        # Stop event is set, signal the stream to terminate
        logger.debug("Stop event set in callback, raising CallbackStop.")
        raise sd.CallbackStop

# --- Main Recording Function ---

def record_audio_interactive(filename="temp_candidate_response.wav", samplerate=DEFAULT_AUDIO_SAMPLERATE):
    """
    Records audio from the default input device interactively.

    Starts recording when Enter is pressed once.
    Stops recording when Enter is pressed again.
    Saves the recording to the specified filename.

    Args:
        filename (str): The path/name to save the recorded WAV file.
        samplerate (int): The sample rate for the recording (samples per second).

    Returns:
        str | None: The path to the saved audio file if successful, otherwise None.
    """
    global stop_recording_event
    stop_recording_event.clear() # Ensure stop event is reset
    # Clear the queue more robustly before starting
    while not recording_queue.empty():
        try:
            recording_queue.get_nowait()
        except queue.Empty:
            break
    logger.debug("Recording queue cleared and stop event reset.")

    try:
        # +++ DEBUGGING: Print available audio devices +++
        logger.info("Querying available audio devices...")
        print("\n--- Available Audio Devices ---")
        try:
            print(sd.query_devices())
        except Exception as query_err:
             print(f"Error querying devices: {query_err}")
             logger.error(f"Could not query audio devices: {query_err}")
        print("-----------------------------")
        # Identify default input device index for logging/reference
        try:
            default_input_device_index = sd.default.device[0] # Index 0 is input
            logger.info(f"Default input device index: {default_input_device_index}")
        except Exception:
            logger.warning("Could not determine default input device index.")
        # ++++++++++++++++++++++++++++++++++++++++++++++++++

        # Check if the *default* input device supports the requested settings
        logger.info(f"Checking settings for default input device ({samplerate} Hz, 1 channel)...")
        sd.check_input_settings(samplerate=samplerate, channels=1)
        logger.info(f"Default input device confirmed to support {samplerate} Hz mono.")

    except Exception as e:
        logger.error(f"Error checking input audio device settings: {e}", exc_info=True)
        print(f"\nError: Could not configure the default audio input device.")
        print(f"Please ensure a microphone is connected and selected in your OS sound settings.")
        print(f"Details: {e}")
        return None

    print(f"\n🎙️  Press [Enter] to START recording...")
    try:
        # Wait for user to press Enter to start
        input()
    except EOFError:
        logger.warning("EOFError while waiting for start input. Aborting recording.")
        print("Error: Input stream closed unexpectedly. Cannot start recording.")
        return None
    # Consider adding a small pause if recordings stop instantly sometimes:
    # time.sleep(0.1)

    stream = None # Initialize stream variable
    recording_thread = None
    audio_data = []

    try:
        # --- Start the input stream ---
        # Use device=INDEX here if you need to specify a non-default mic based on printout above
        # Example: device=1 if device 1 is your desired mic
        stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            callback=_audio_callback,
            blocksize=int(samplerate * 0.1) # 100ms blocks for reasonable latency
            # device=YOUR_DEVICE_INDEX # Optional: Uncomment and replace index if needed
        )

        # Start the stream in a separate thread so the main thread can wait for input
        # daemon=True allows the program to exit even if this thread hangs
        recording_thread = threading.Thread(target=stream.start, daemon=True)
        recording_thread.start()
        logger.info("Audio recording started.")
        print("🔴 Recording... Press [Enter] again to STOP.")

        # Wait for user to press Enter again to stop
        input()
        stop_recording_event.set() # Signal the callback and stream to stop
        logger.info("Stop recording requested via Enter key.")

    except EOFError:
         logger.warning("EOFError received while waiting for stop input. Stopping recording.")
         print("Input stream closed unexpectedly. Stopping recording...")
         if not stop_recording_event.is_set(): stop_recording_event.set() # Ensure stop is signaled
    except Exception as e:
         logger.error(f"Error during recording stream process: {e}", exc_info=True)
         print(f"An error occurred during recording: {e}")
         if not stop_recording_event.is_set(): stop_recording_event.set() # Ensure stop is signaled on error
         # Attempt cleanup immediately on error
         if stream:
             try: stream.close()
             except Exception: pass # Ignore errors during cleanup on error
         if recording_thread and recording_thread.is_alive():
             try: recording_thread.join(timeout=0.5)
             except Exception: pass
         return None # Indicate failure
    finally:
        # --- Graceful Stream Shutdown ---
        logger.debug("Entering finally block for stream cleanup...")
        if stream is not None:
            logger.debug(f"Stream active: {stream.active}, Stream closed: {stream.closed}")
            # Ensure stop_recording_event is set one last time
            if not stop_recording_event.is_set():
                 logger.warning("Stop event wasn't set before finally block, setting now.")
                 stop_recording_event.set()
            try:
                 # Wait for the recording thread to finish (important for callback completion)
                 if recording_thread is not None and recording_thread.is_alive():
                     logger.debug("Waiting for recording thread to finish...")
                     recording_thread.join(timeout=1.5) # Wait a bit longer
                     if recording_thread.is_alive():
                          logger.warning("Recording thread did not finish cleanly after timeout.")

                 # Now stop and close the stream
                 if not stream.stopped:
                     logger.debug("Stream not stopped yet, attempting stop...")
                     stream.stop()
                     logger.debug("Stream stopped.")
                 if not stream.closed:
                     logger.debug("Stream not closed yet, attempting close...")
                     stream.close()
                     logger.debug("Stream closed.")
                 logger.debug("Audio stream stopped and closed.")
            except Exception as e:
                 # Log errors during cleanup but don't crash
                 logger.warning(f"Exception during audio stream cleanup: {e}", exc_info=True)
        else:
            logger.debug("Stream object was None, no cleanup needed.")
        # --- End Graceful Stream Shutdown ---

    # Retrieve all data blocks from the queue
    logger.debug("Retrieving final audio data from queue...")
    while not recording_queue.empty():
        try:
            audio_data.append(recording_queue.get_nowait())
        except queue.Empty:
            break # Should not happen with check, but safety
    logger.debug(f"Retrieved {len(audio_data)} audio blocks from queue.")

    if not audio_data:
        logger.error("No audio data was captured in the queue after recording stopped.")
        print("Error: No audio was recorded. This might happen if recording stopped immediately or due to an internal issue.")
        return None # Indicate failure

    # Concatenate blocks and save to WAV file
    try:
        recording = np.concatenate(audio_data, axis=0)
        duration = len(recording) / samplerate
        logger.info(f"Recording concatenation complete. Samples: {len(recording)}, Duration: {duration:.2f}s")

        # Check for very short recordings which often fail STT
        if duration < 0.5: # Threshold might need adjustment
             logger.warning(f"Recorded audio is very short ({duration:.2f}s). STT may fail.")
             # Decide whether to proceed or return failure
             # return None # Optionally return None here

        # Ensure output directory exists
        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir):
             logger.info(f"Creating directory for audio file: {file_dir}")
             os.makedirs(file_dir, exist_ok=True)

        # Write the NumPy array to a WAV file using soundfile
        logger.info(f"Saving audio recording to {filename}...")
        sf.write(filename, recording, samplerate)
        logger.info(f"Audio successfully saved to {filename}")
        return filename # Return the path of the saved file

    except ValueError as e:
         # Error typically happens if audio_data list is empty or arrays have incompatible shapes
         logger.error(f"Error concatenating audio blocks: {e}. Number of blocks: {len(audio_data)}", exc_info=True)
         print("Error: Failed to process recorded audio data.")
         return None
    except Exception as e:
        logger.error(f"Error saving audio file {filename}: {e}", exc_info=True)
        print(f"Error: Could not save recorded audio to {filename}.")
        return None

# Note: The rest of audio_utils.py (transcribe_audio function) should also be present
# in the file for the complete module to work.
# --- Speech-to-Text (STT) ---

def transcribe_audio(audio_path):
    """
    Transcribes audio from a file using SpeechRecognition library (Google Web Speech API).

    Args:
        audio_path (str): Path to the WAV audio file.

    Returns:
        str: The transcribed text, or an error message string starting with '[Error'
             or '[Audio unintelligible or empty]'.
    """
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"STT Error: Audio file not found at '{audio_path}'")
        return "[Error: Audio file not found for transcription]"

    recognizer = sr_audio.Recognizer()
    audio_file_instance = None # To ensure it's closed

    try:
        # Open the audio file
        audio_file_instance = sr_audio.AudioFile(audio_path)
        with audio_file_instance as source:
            # Optional: Adjust for ambient noise - can improve accuracy but adds delay
            # logger.info(f"Adjusting for ambient noise in {audio_path}...")
            # try:
            #     recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # except Exception as noise_err:
            #      logger.warning(f"Could not adjust for ambient noise: {noise_err}")

            logger.info(f"Reading audio file data for STT: {audio_path}")
            # Record the audio data from the file source
            audio_data = recognizer.record(source)
            logger.info(f"Audio data loaded for STT.") # Removed duration access

    except ValueError as e:
         # This can happen if the WAV file is empty or corrupted
         logger.error(f"STT ValueError reading {audio_path}: {e}. File might be empty/corrupt.", exc_info=True)
         return f"[Error: Cannot process invalid WAV file {os.path.basename(audio_path)}]"
    except Exception as e:
        logger.error(f"Error reading audio file {audio_path} for STT: {e}", exc_info=True)
        # Ensure file handle is closed if opened
        # Note: 'with' statement handles closing automatically on exit/exception
        return f"[Error reading audio file for STT: {e}]"

    # Perform Transcription
    try:
        logger.info("Attempting transcription using Google Web Speech API...")
        # Use Google Web Speech API (requires internet connection)
        # language="en-US" # Optionally specify language
        text = recognizer.recognize_google(audio_data)
        logger.info(f"Transcription successful: '{text[:50]}...'")
        return text
    except sr_audio.UnknownValueError:
        # Speech was unintelligible or the segment was silence
        logger.warning(f"Google Web Speech API could not understand audio from {audio_path}")
        return "[Audio unintelligible or empty]"
    except sr_audio.RequestError as e:
        # Could not reach the Google API service
        logger.error(f"Could not request STT results from Google API; {e}")
        return f"[STT API Error: {e}]"
    except Exception as e:
         # Catch any other unexpected errors during recognition
         logger.error(f"Unexpected error during STT recognition: {e}", exc_info=True)
         return f"[Unexpected STT Error: {e}]"

# --- Example Usage (if run directly) ---
if __name__ == "__main__":
    print("Audio Utils Module - Example Usage")
    logging.basicConfig(level=logging.INFO) # Setup basic logging for direct run

    # Example 1: Record Audio
    print("\n--- Testing Audio Recording ---")
    output_file = "test_recording.wav"
    recorded_path = record_audio_interactive(filename=output_file)

    if recorded_path:
        print(f"Audio recorded and saved to: {recorded_path}")

        # Example 2: Transcribe the recorded audio
        print("\n--- Testing Transcription ---")
        transcription = transcribe_audio(recorded_path)
        print(f"Transcription Result: {transcription}")

        # Clean up the test file
        try:
            os.remove(recorded_path)
            print(f"Cleaned up {recorded_path}")
        except OSError as e:
            print(f"Could not remove test file {recorded_path}: {e}")
    else:
        print("Audio recording failed, skipping transcription test.")