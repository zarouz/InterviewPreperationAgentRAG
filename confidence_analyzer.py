# confidence_analyzer.py
import os
import numpy as np
import librosa
import tensorflow as tf
# tensorflow.keras layers might be needed for custom object scope if direct loading fails later
# from tensorflow.keras import layers
import warnings
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import tempfile
import soundfile as sf
import webrtcvad # Ensure this is installed: pip install webrtcvad-wheels
import logging

logger = logging.getLogger(__name__)

# Suppress common Librosa warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# --- Configuration ---
# *** USE THE PATH FROM YOUR WORKING INFERENCE SCRIPT ***
MODEL_PATH = "emotions_model/end_to_end_emotion_model.keras"
# If the .h5 version loads successfully in your inference script, use that instead:
# MODEL_PATH = "BestEmotionModel/emotion_recognition_models/end_to_end_model_from_h5.h5" # Or appropriate .h5 name

# Feature & Model parameters (MUST MATCH TRAINING and your inference script)
SR_TARGET_ENC_ATT = 16000
N_MFCC_ENC_ATT = 40
MAX_LEN_ENC_ATT = 300
FEATURE_DIM_ENC_ATT = N_MFCC_ENC_ATT + 7 + 12  # 59
# *** VERIFY THIS LIST MATCHES YOUR MODEL OUTPUT/INFERENCE SCRIPT ***
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_CLASSES)

# Confidence weights (Adjust as needed)
CONFIDENCE_WEIGHTS = {
    'angry': 0.3, 'disgust': 0.5, 'fear': 0.1, 'happy': 0.9,
    'neutral': 0.8, 'sad': 0.2, 'surprise': 0.4
}

# --- Global Model Variable ---
loaded_emotion_model = None

# --- Model Loading Function (Simplified) ---
def load_emotion_model():
    """Loads the Keras emotion model directly using tf.keras.models.load_model."""
    global loaded_emotion_model
    if loaded_emotion_model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Emotion model file not found at '{MODEL_PATH}'")
            raise FileNotFoundError(f"Emotion model not found: {MODEL_PATH}")
        try:
            logger.info(f"Loading emotion model from '{MODEL_PATH}'...")
            # --- Direct Loading Call ---
            # Set compile=False if you don't need the optimizer state for inference
            loaded_emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            # --------------------------
            logger.info("Emotion model loaded successfully.")
            loaded_emotion_model.summary(print_fn=logger.info) # Log summary on first load
        except Exception as e:
            # If direct loading fails HERE despite working in your inference script,
            # it might indicate subtle environment differences or issues with
            # how the model is loaded within the larger application context.
            # We might need the custom_object_scope workaround again in that specific case.
            logger.error(f"Error loading Keras emotion model directly: {e}", exc_info=True)
            # Attempt with custom object scope as a fallback if direct load fails?
            # try:
            #     logger.warning("Direct load failed, trying with custom object scope for Permute...")
            #     from tensorflow.keras import layers
            #     custom_objects = {'Transpose': layers.Permute} # Assuming Transpose -> Permute
            #     with tf.keras.utils.custom_object_scope(custom_objects):
            #          loaded_emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            #     logger.info("Model loaded successfully with custom object scope.")
            #     loaded_emotion_model.summary(print_fn=logger.info)
            # except Exception as e2:
            #      logger.error(f"Error loading Keras model even with custom scope: {e2}", exc_info=True)
            #      raise RuntimeError(f"Failed to load emotion model: {e2}") from e2
            # For simplicity now, just raise the original error
            raise RuntimeError(f"Failed to load emotion model directly: {e}") from e
    return loaded_emotion_model

# --- Feature Extraction (Use the one from your inference script) ---
def extract_features_encoder_attention(file_path, sr=SR_TARGET_ENC_ATT, n_mfcc=N_MFCC_ENC_ATT, max_len=MAX_LEN_ENC_ATT):
    """
    Extracts features matching the inference script.
    """
    try:
        audio, current_sr = librosa.load(file_path, sr=None, mono=True) # Load native SR
        if current_sr != sr:
             audio = librosa.resample(y=audio, orig_sr=current_sr, target_sr=sr)
        if len(audio) < 100: logger.warning(f"Audio too short: {file_path}"); return None
    except Exception as e: logger.error(f"Error loading audio {file_path}: {e}"); return None

    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T

        if mfccs.shape[0] < 2: logger.warning(f"Insufficient frames {file_path}"); return None

        mfccs_mean, mfccs_std = np.mean(mfccs, axis=0), np.std(mfccs, axis=0)
        contrast_mean, contrast_std = np.mean(contrast, axis=0), np.std(contrast, axis=0)
        chroma_mean, chroma_std = np.mean(chroma, axis=0), np.std(chroma, axis=0)

        mfccs = np.divide(mfccs - mfccs_mean, mfccs_std + 1e-10, out=np.zeros_like(mfccs), where=mfccs_std > 1e-10)
        contrast = np.divide(contrast - contrast_mean, contrast_std + 1e-10, out=np.zeros_like(contrast), where=contrast_std > 1e-10)
        chroma = np.divide(chroma - chroma_mean, chroma_std + 1e-10, out=np.zeros_like(chroma), where=chroma_std > 1e-10)

        features = np.nan_to_num(np.hstack([mfccs, contrast, chroma]))

        current_len = features.shape[0]
        if current_len == 0: logger.warning(f"Zero frames after features {file_path}"); return None
        if current_len < max_len: features = np.pad(features, ((0, max_len - current_len), (0, 0)), mode='constant')
        elif current_len > max_len: features = features[:max_len, :]

        if features.shape != (max_len, FEATURE_DIM_ENC_ATT):
            logger.error(f"Feature dim mismatch {features.shape} vs {(max_len, FEATURE_DIM_ENC_ATT)}"); return None
        return features.astype(np.float32)
    except Exception as e: logger.error(f"Error extracting features {file_path}: {e}", exc_info=True); return None


# --- VAD Segmentation (Keep your working version) ---
def split_audio_by_speech(audio_path, sr=SR_TARGET_ENC_ATT, min_segment_length=3.0, vad_aggressiveness=2):
    # ... (Keep the function code from your provided confidence script) ...
    segments_data = []
    try:
        audio, current_sr = librosa.load(audio_path, sr=None, mono=True)
        if current_sr != sr: audio = librosa.resample(y=audio, orig_sr=current_sr, target_sr=sr)
        if len(audio) < sr * 0.1: logger.warning("Audio too short for VAD"); return None, None
        audio_int = np.int16(audio * 32767)
        vad = webrtcvad.Vad(vad_aggressiveness)
        frame_duration_ms = 30; frame_size = int(sr * frame_duration_ms / 1000); num_frames = len(audio_int) // frame_size
        is_speaking = False; segment_start_frame = 0
        min_speech_frames = int((min_segment_length * 1000) / frame_duration_ms); padding_frames = int(0.1 * 1000 / frame_duration_ms)
        for i in range(num_frames):
            start_sample = i * frame_size; end_sample = start_sample + frame_size; frame = audio_int[start_sample:end_sample]
            if len(frame) < frame_size: frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
            try: frame_is_speech = vad.is_speech(frame.tobytes(), sr)
            except Exception: frame_is_speech = False
            if not is_speaking and frame_is_speech: is_speaking = True; segment_start_frame = max(0, i - padding_frames)
            elif is_speaking and not frame_is_speech:
                if (i - segment_start_frame) >= min_speech_frames:
                    segment_end_frame = min(num_frames, i + padding_frames); start = segment_start_frame * frame_size; end = segment_end_frame * frame_size
                    segments_data.append(audio[start:end])
                is_speaking = False
        if is_speaking and (num_frames - segment_start_frame) >= min_speech_frames:
             segment_end_frame = num_frames; start = segment_start_frame * frame_size; end = segment_end_frame * frame_size
             segments_data.append(audio[start:end])
        if not segments_data and len(audio) >= sr * min_segment_length: logger.warning("VAD found no segments, returning full audio."); return [audio], sr
        elif not segments_data: logger.warning("VAD found no segments and audio too short."); return None, None
        return segments_data, sr
    except Exception as e: logger.error(f"Error in VAD processing: {e}", exc_info=True); return None, None

# --- Weighted Aggregation (Keep your working version) ---
def aggregate_results(segment_results):
    # ... (Keep the function code from your provided confidence script) ...
    if not segment_results: return None
    valid_segments = [s for s in segment_results if 'probs' in s and s['probs'] is not None]
    if not valid_segments: logger.warning("No valid segments for aggregation."); return None
    weighted_probs = np.zeros_like(valid_segments[0]['probs']); total_weight = 0
    for seg in valid_segments:
        confidence = seg.get('confidence', 0); duration = seg.get('duration', 0); probs = seg.get('probs')
        if not isinstance(confidence, (int, float)) or not isinstance(duration, (int, float)): continue
        weight = 0 if duration < 0.1 else duration * (1 + confidence / 100.0) # Your weighting
        weighted_probs += probs * weight; total_weight += weight
    if total_weight <= 1e-6:
        logger.warning("Total weight zero in aggregation, fallback to average.")
        valid_probs = [seg['probs'] for seg in valid_segments]; return np.mean(valid_probs, axis=0) if valid_probs else None
    return weighted_probs / total_weight


# --- Main VAD-based Processor (Simplified - uses direct model load) ---
def process_long_audio_vad(audio_path, model, classes, feature_params):
    """Processes long audio using VAD and the pre-loaded model."""
    segments, sr = split_audio_by_speech(
        audio_path, sr=feature_params['sr'],
        min_segment_length=3.0, vad_aggressiveness=2
    )
    if segments is None or not segments:
        logger.warning(f"No valid segments from VAD for {os.path.basename(audio_path)}")
        return None, [], True # Error state

    segment_results = []
    logger.info(f"Processing {len(segments)} speech segments from {os.path.basename(audio_path)}...")

    for i, segment in enumerate(segments):
        duration = len(segment) / sr
        if duration < 0.5: continue # Skip very short segments

        temp_filename = None; features = None
        try:
            # Create temp file for feature extraction
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                temp_filename = tmpfile.name
            sf.write(temp_filename, segment, sr)

            # Extract features using the consistent function
            features = extract_features_encoder_attention(
                temp_filename, sr=feature_params['sr'],
                n_mfcc=feature_params['n_mfcc'], max_len=feature_params['max_len']
            )
        except Exception as proc_err:
            logger.error(f"Error preparing/extracting features for segment {i+1}: {proc_err}", exc_info=True)
        finally: # Ensure cleanup
            if temp_filename and os.path.exists(temp_filename):
                try: os.remove(temp_filename)
                except OSError: pass # Ignore cleanup error

        if features is None:
            logger.warning(f"Segment {i+1} feature extraction failed."); continue

        # Predict using the globally loaded model
        features_batch = np.expand_dims(features, axis=0)
        try:
            # --- Uses the model loaded by load_emotion_model ---
            probs = model.predict(features_batch, verbose=0)[0]
            # ---------------------------------------------------
            if probs is None or probs.shape != (len(classes),): continue # Skip invalid predictions

            pred_index = np.argmax(probs)
            pred_confidence = probs[pred_index] * 100

            segment_results.append({
                'segment': i + 1, 'emotion': classes[pred_index],
                'confidence': pred_confidence, 'probs': probs, 'duration': duration
            })
            logger.debug(f"Segment {i+1}: {classes[pred_index]} ({pred_confidence:.1f}%), Duration: {duration:.2f}s")
        except Exception as pred_err:
            logger.error(f"Error predicting segment {i+1}: {pred_err}", exc_info=True)

    if not segment_results:
        logger.error(f"Processing failed for all segments in {os.path.basename(audio_path)}")
        return None, [], True

    final_probs = aggregate_results(segment_results)
    if final_probs is None:
        logger.error(f"Aggregation failed for {os.path.basename(audio_path)}")
        return None, segment_results, True # Return segments found, but error state

    return final_probs, segment_results, False # Success


# --- Confidence Scoring & Rating (Keep your versions) ---
def calculate_confidence_score(emotion_probs, emotion_classes, confidence_weights):
    # ... (Keep the function from your provided confidence script) ...
    weighted_sum = 0; matched_classes = 0
    for i, emotion in enumerate(emotion_classes):
        weight = confidence_weights.get(emotion)
        if weight is not None: weighted_sum += emotion_probs[i] * weight; matched_classes += 1
        else: logger.warning(f"Emotion '{emotion}' not in CONFIDENCE_WEIGHTS.")
    if matched_classes == 0: logger.error("No emotion classes matched weights!"); return 0.0
    confidence_score = min(max(weighted_sum * 100, 0), 100)
    return confidence_score

def get_confidence_rating(score):
    # ... (Keep the function from your provided confidence script) ...
    if score is None: return "N/A"
    if score >= 80: return "Very High"
    elif score >= 65: return "High"
    elif score >= 45: return "Moderate"
    elif score >= 25: return "Low"
    else: return "Very Low"

# --- Main Analysis Function (Simplified loading) ---
def analyze_confidence(audio_path):
    """Performs full confidence analysis on an audio file using direct model loading."""
    results = {'score': None, 'rating': "N/A", 'primary_emotion': "N/A", 'emotion_confidence': None, 'segment_results': [], 'all_probs': None, 'error': True}
    try:
        # --- Uses the simplified load_emotion_model ---
        model = load_emotion_model()
        # ---------------------------------------------
        if model is None: raise RuntimeError("Emotion model is None after load_emotion_model call.")
    except Exception as load_err:
        logger.error(f"Failed to load model for analysis: {load_err}", exc_info=True)
        return results # Return default error state

    feature_params = {'sr': SR_TARGET_ENC_ATT, 'n_mfcc': N_MFCC_ENC_ATT, 'max_len': MAX_LEN_ENC_ATT}

    agg_probs, segment_details, error_flag = process_long_audio_vad(
        audio_path, model, EMOTION_CLASSES, feature_params
    )
    results['segment_results'] = segment_details

    if error_flag or agg_probs is None: logger.error(f"Analysis failed during processing for {os.path.basename(audio_path)}")
    else:
        predicted_index = np.argmax(agg_probs); primary_emotion = EMOTION_CLASSES[predicted_index]
        emotion_confidence_percent = agg_probs[predicted_index] * 100
        overall_score = calculate_confidence_score(agg_probs, EMOTION_CLASSES, CONFIDENCE_WEIGHTS)
        rating = get_confidence_rating(overall_score)
        results.update({'score': overall_score, 'rating': rating, 'primary_emotion': primary_emotion, 'emotion_confidence': emotion_confidence_percent, 'all_probs': agg_probs.tolist(), 'error': False})
        logger.info(f"Confidence analysis complete for {os.path.basename(audio_path)}: Score={overall_score:.1f}, Rating={rating}")

    return results


# --- Visualization Function (Keep your version or remove) ---
def visualize_confidence_analysis(agg_probs, score, rating, classes=EMOTION_CLASSES):
    # ... (Keep the function code from your provided confidence script) ...
    pass # Keep or remove based on whether you need it called from here

# --- Example Usage (Keep your version or remove) ---
if __name__ == "__main__":
    # ... (Keep the example usage code from your provided confidence script) ...
    pass # Keep or remove