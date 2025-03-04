import streamlit as st
import librosa as lr
import numpy as np
from tensorflow.keras.models import load_model
import whisper
import io

# Load the LSTM model
loaded_model = load_model('lstm_language.h5')

# Load Whisper model
model = whisper.load_model("base")

# Language classes for the LSTM model
language_classes = ['Bengali', 'Gujarati', 'Hindi', 'Kannada', 'Malayalam', 'Marathi', 'Punjabi', 'Tamil', 'Telugu', 'Urdu']

# Streamlit app title
st.title("India Language Speech Recognition")

# File uploader for the audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # 1. Use librosa to extract features for LSTM model (MFCC)
    y, sr = lr.load(uploaded_file, sr=None)

    # Display audio playback in Streamlit
    st.audio(uploaded_file, format="audio/wav")

    # Extract MFCC features from the audio
    mfccs_feature = lr.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Scale the MFCC features
    mfccs_feature_scale = np.mean(mfccs_feature.T, axis=0)

    # Prepare input for the LSTM model
    m = np.expand_dims(mfccs_feature_scale.reshape(1, -1), axis=-1)

    # Make prediction using the LSTM model
    predictions = loaded_model.predict(m)

    # Get the predicted language class
    predicted_class = np.argmax(predictions)

    # Display the predicted language class
    st.success(f"Predicted Language (LSTM Model): {language_classes[predicted_class]}")

    # 2. Use Whisper for speech-to-text transcription
    try:
        # Convert the uploaded file to a NumPy array using librosa
        # librosa.load already returns the waveform as a NumPy array
        audio_np, _ = lr.load(uploaded_file, sr=16000)  # Whisper typically uses 16kHz sampling rate

        # Use Whisper for transcription
        result = model.transcribe(audio_np)

        # Display the transcription result
        st.write("Transcription (Whisper):")
        st.write(result['text'])
        # Optionally, print the transcription result to the console
        print("Transcription:", result['text'])

    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        print(f"Error during transcription: {e}")
