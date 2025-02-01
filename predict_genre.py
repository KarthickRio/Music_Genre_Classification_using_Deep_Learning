import os
import librosa
import numpy as np
from keras.models import load_model

# Define parameters
num_mfcc = 13
n_fft = 2048
hop_length = 512
num_segments = 5
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# Number of samples per segment
samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

# Function to preprocess and extract MFCCs
def preprocess_and_extract_mfcc(signal, sample_rate, num_mfcc, n_fft, hop_length, num_segments):
    mfccs = []

    # Compute MFCCs for each segment
    for s in range(num_segments):
        start_sample = s * samples_per_segment
        end_sample = (s + 1) * samples_per_segment
        mfcc = librosa.feature.mfcc(y=signal[start_sample:end_sample], sr=sample_rate,
                                    n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs.append(mfcc.T)

    # Combine the MFCCs from all segments
    combined_mfccs = np.concatenate(mfccs, axis=0)

    # Ensure the combined MFCCs have exactly 130 time steps
    if combined_mfccs.shape[0] < 130:
        combined_mfccs = np.pad(combined_mfccs, ((0, 130 - combined_mfccs.shape[0]), (0, 0)), mode='constant')
    elif combined_mfccs.shape[0] > 130:
        combined_mfccs = combined_mfccs[:130, :]

    # Add batch dimension
    X = np.expand_dims(combined_mfccs, axis=0)
    return X

# Load the pre-trained model
model = load_model("cnn_genre_classification_model.keras")
#cnn_genre_classification_model.keras
#Genre_classification_model.keras
# Path to the folder containing all classical songs
folder_path = r"C:\Users\SENSORS LAB-3\Downloads\genre_correct_output\Metal_correct"
results = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.mp3') or file_name.endswith('.wav'):  # Adjust extensions as needed
        file_path = os.path.join(folder_path, file_name)
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        # Preprocess the audio signal and extract MFCCs
        X = preprocess_and_extract_mfcc(signal, sample_rate, num_mfcc, n_fft, hop_length, num_segments)

        # Make a prediction
        prediction = model.predict(X)

        # Get the predicted index
        predicted_index = np.argmax(prediction, axis=1)

        # Define the genre directories
        genre_directories = [
            "geners_original//blues",
            "geners_original//classical",
            "geners_original//country",
            "geners_original//disco",
            "geners_original//hiphop",
            "geners_original//jazz",
            "geners_original//metal",
            "geners_original//pop",
            "geners_original//reggae",
            "geners_original//rock"
        ]

        # Get the predicted genre
        predicted_genre = genre_directories[predicted_index[0]]

        # Store the result
        result = (file_name, predicted_genre)
        results.append(result)

        # Print each result individually
        print(f'{result}')