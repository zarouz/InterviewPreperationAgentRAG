import os
import numpy as np
import librosa
import tensorflow as tf
import argparse
import warnings
import sys # <--- IMPORT SYS

print("--- SCRIPT START: inference.py ---") # <--- ADD THIS
print(f"--- Python receiving args: {sys.argv}") # <--- ADD THIS

# Suppress common Librosa warnings (optional)
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# --- Configuration ---

# Path to the saved end-to-end Keras model
MODEL_PATH = "emotions_model/end_to_end_emotion_model.keras"

# Feature extraction parameters (MUST EXACTLY MATCH the training of the loaded model)
SR_TARGET_ENC_ATT = 16000
N_MFCC_ENC_ATT = 40
MAX_LEN_ENC_ATT = 300
FEATURE_DIM_ENC_ATT = N_MFCC_ENC_ATT + 7 + 12 # 59

# Emotion classes (MUST match the order/labels used during training)
# --- !!! IMPORTANT: Verify this list matches your model's output classes !!! ---
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral' ,'sad', 'surprise']
# If your model was trained on 8 classes including 'surprised', use:
# EMOTION_CLASSES = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# Check the classification report or confusion matrix from your training script if unsure.


# --- Feature Extraction Function (Copied from evaluation script) ---

def extract_features_encoder_attention(file_path, sr=SR_TARGET_ENC_ATT, n_mfcc=N_MFCC_ENC_ATT, max_len=MAX_LEN_ENC_ATT):
    """
    Extracts MFCC+Contrast+Chroma features suitable for the third model.
    Returns shape (max_len, feature_dim) or None on failure.
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        if len(audio) < 10: # Basic check for very short/empty audio
             print(f"Warning: Audio file seems too short or empty: {file_path}")
             return None
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).T
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr).T

        # Normalize (handle potential zero std dev)
        mfccs_mean, mfccs_std = np.mean(mfccs, axis=0), np.std(mfccs, axis=0)
        contrast_mean, contrast_std = np.mean(contrast, axis=0), np.std(contrast, axis=0)
        chroma_mean, chroma_std = np.mean(chroma, axis=0), np.std(chroma, axis=0)

        # Use np.divide for safe division, replacing potential NaN/inf with 0
        mfccs = np.divide(mfccs - mfccs_mean, mfccs_std + 1e-10, out=np.zeros_like(mfccs), where=mfccs_std!=0)
        contrast = np.divide(contrast - contrast_mean, contrast_std + 1e-10, out=np.zeros_like(contrast), where=contrast_std!=0)
        chroma = np.divide(chroma - chroma_mean, chroma_std + 1e-10, out=np.zeros_like(chroma), where=chroma_std!=0)

        features = np.nan_to_num(np.hstack([mfccs, contrast, chroma])) # Ensure no NaNs remain

        # Pad/Truncate
        current_len = features.shape[0]
        if current_len < max_len:
            features = np.pad(features, ((0, max_len - current_len), (0, 0)), mode='constant')
        elif current_len > max_len:
            features = features[:max_len, :]

        # Final dimension check
        if features.shape[1] != FEATURE_DIM_ENC_ATT:
            print(f"Error: Feature dimension mismatch after extraction. Expected {FEATURE_DIM_ENC_ATT}, got {features.shape[1]}.")
            return None

        return features.astype(np.float32)

    except Exception as e:
        print(f"Error during feature extraction for {file_path}: {e}")
        return None


# --- Inference Function ---

def predict_emotion(audio_path, loaded_model, classes, feature_params):
    """
    Performs inference on a single audio file using the loaded model.

    Args:
        audio_path (str): Path to the input audio file.
        loaded_model (tf.keras.Model): The loaded Keras model.
        classes (list): List of emotion class names in the correct order.
        feature_params (dict): Dictionary with 'sr', 'n_mfcc', 'max_len'.

    Returns:
        tuple: (predicted_emotion_label, confidence_score) or (None, None) on error.
    """
    print(f"\nProcessing audio file: {os.path.basename(audio_path)}")

    # 1. Extract features
    features = extract_features_encoder_attention(
        audio_path,
        sr=feature_params['sr'],
        n_mfcc=feature_params['n_mfcc'],
        max_len=feature_params['max_len']
    )

    if features is None:
        print("-> Feature extraction failed.")
        return None, None
    print(f"-> Features extracted, shape: {features.shape}") # Shape: (max_len, feature_dim)

    # 2. Prepare features for the model (add batch dimension)
    # Model expects input shape: (batch_size, time_steps, features)
    features_batch = np.expand_dims(features, axis=0)
    print(f"-> Input shape for model: {features_batch.shape}") # Shape: (1, max_len, feature_dim)

    # 3. Perform prediction
    try:
        prediction_probs = loaded_model.predict(features_batch, verbose=0)
        # prediction_probs shape should be (1, num_classes)

        if prediction_probs is None or prediction_probs.ndim != 2 or prediction_probs.shape[1] != len(classes):
             print(f"Error: Unexpected model prediction output shape. Expected: (1, {len(classes)}), Got: {prediction_probs.shape if prediction_probs is not None else 'None'}")
             return None, None

        # 4. Interpret results
        predicted_index = np.argmax(prediction_probs[0])
        predicted_emotion = classes[predicted_index]
        confidence = prediction_probs[0][predicted_index] * 100 # Convert to percentage

        print(f"-> Raw probabilities: {prediction_probs[0]}")
        return predicted_emotion, confidence

    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None, None


# --- Main Execution ---

# --- Main Execution ---

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file using the Encoder+Attention CNN model. Can also resave the model.")
    parser.add_argument("audio_file", help="Path to the input audio file (.wav, .mp3, etc.) for testing prediction.")
    parser.add_argument("--resave", action="store_true", help="Load the model and immediately re-save it to potentially fix loading issues.") # Add this argument
    args = parser.parse_args()

    input_audio_path = args.audio_file

    # 1. Validate input audio file path (Keep this for testing prediction)
    if not os.path.exists(input_audio_path):
        print(f"Error: Input audio file not found at '{input_audio_path}'")
        exit(1)
    if not os.path.isfile(input_audio_path):
         print(f"Error: Provided path '{input_audio_path}' is not a file.")
         exit(1)

    # 2. Validate model file path
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please ensure the path is correct relative to where you run the script.")
        exit(1)

    # 3. Load the Keras model
    print(f"Loading model from '{MODEL_PATH}'...")
    try:
        # Use the exact same loading method as your failing script initially
        model = tf.keras.models.load_model(MODEL_PATH, compile=False) # Set compile=False like in confidence_analyzer
        print("Model loaded successfully.")
        # model.summary() # Optional: uncomment to print model layers
    except Exception as e:
        print(f"Error loading Keras model even in inference.py: {e}")
        print("This indicates a deeper problem. Check the file path, file integrity, and TensorFlow/Keras installation.")
        exit(1)

    # <<< --- ADD THIS SECTION --- >>>
    if args.resave:
        print(f"\nRe-saving the loaded model back to '{MODEL_PATH}'...")
        try:
            # Save using the standard Keras format. Keep compile=False consistent with loading if you don't need optimizer state.
            # If you DO need the optimizer state later (e.g., for fine-tuning), you might load/save with compile=True,
            # but ensure the original compilation parameters are compatible. For pure inference, compile=False is safer.
            model.save(MODEL_PATH, save_format='keras')
            print(f"Model re-saved successfully to '{MODEL_PATH}'.")
            print("Try running your interview_simulator.py script again.")
            exit(0) # Exit after re-saving
        except Exception as e:
            print(f"Error re-saving the model: {e}")
            exit(1)
    # <<< --- END ADDED SECTION --- >>>


    # 4. Define feature parameters dictionary (Keep for prediction test)
    feature_parameters = {
        'sr': SR_TARGET_ENC_ATT,
        'n_mfcc': N_MFCC_ENC_ATT,
        'max_len': MAX_LEN_ENC_ATT
    }

    # 5. Perform prediction (Keep for testing)
    print("\nPerforming test prediction on the loaded model...")
    predicted_label, confidence_score = predict_emotion(
        input_audio_path,
        model,
        EMOTION_CLASSES,
        feature_parameters
    )

    # 6. Print the final result (Keep for testing)
    print("\n----------------------------------------")
    if predicted_label:
        print(f"      Predicted Emotion: {predicted_label}")
        print(f"      Confidence:        {confidence_score:.2f}%")
    else:
        print("      Prediction Failed.")
    print("----------------------------------------")