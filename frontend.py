import os
import librosa
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from keras.models import load_model

# Flask App
app = Flask(__name__)

# Define parameters
num_mfcc = 13
n_fft = 2048
hop_length = 512
num_segments = 5
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

# Load the pre-trained model
model = load_model("cnn_genre_classification_model.keras")

# Define genre directories
genre_directories = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# Function to preprocess and extract MFCCs
def preprocess_and_extract_mfcc(signal, sample_rate, num_mfcc, n_fft, hop_length, num_segments):
    mfccs = []
    for s in range(num_segments):
        start_sample = s * samples_per_segment
        end_sample = (s + 1) * samples_per_segment
        mfcc = librosa.feature.mfcc(y=signal[start_sample:end_sample], sr=sample_rate,
                                    n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs.append(mfcc.T)
    combined_mfccs = np.concatenate(mfccs, axis=0)
    if combined_mfccs.shape[0] < 130:
        combined_mfccs = np.pad(combined_mfccs, ((0, 130 - combined_mfccs.shape[0]), (0, 0)), mode='constant')
    elif combined_mfccs.shape[0] > 130:
        combined_mfccs = combined_mfccs[:130, :]
    X = np.expand_dims(combined_mfccs, axis=0)
    return X

# HTML Template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Music Genre Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
                background-image: url('/static/1_QJe0yy4qjugIEX0vS5t5HA.jpg');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                color: white;
                font-family: Arial, sans-serif;
        }

        .container {
            background: rgba(0, 0, 0, 0.6);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-top: 50px;
        }

        h1, h3 {
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }

        .btn-primary {
            background-color: #ff6f61;
            border: none;
        }

        .btn-primary:hover {
            background-color: #e65c54;
        }

        /* Style the result with a glow and animation */
        #result {
            font-size: 2em;
            font-weight: bold;
            color: #ffcc00;
            text-shadow: 3px 3px 5px rgba(0, 0, 0, 0.8);
            animation: fadeIn 2s ease-out;
        }

        /* Animation for the result */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* Glow effect when the result appears */
        @keyframes glowEffect {
            0% { text-shadow: 0 0 5px #ffcc00, 0 0 10px #ffcc00, 0 0 15px #ffcc00; }
            50% { text-shadow: 0 0 15px #ffcc00, 0 0 25px #ffcc00, 0 0 35px #ffcc00; }
            100% { text-shadow: 0 0 5px #ffcc00, 0 0 10px #ffcc00, 0 0 15px #ffcc00; }
        }

        .btn-primary {
            background-color: #ff6f61;
            border: none;
            font-weight: bold;
        }

        .btn-primary:hover {
            background-color: #e65c54;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center">🎵 Music Genre Classifier 🎶</h1>
        <p class="text-center">Upload a song to classify its genre</p>
        <div class="d-flex justify-content-center">
            <form id="uploadForm">
                <div class="mb-3">
                    <input class="form-control" type="file" id="fileInput" accept=".mp3, .wav" required>
                </div>
                <button type="submit" class="btn btn-primary w-100">Classify Genre</button>
            </form>
        </div>
        <div class="mt-4 text-center">
            <h3 id="processing" class="text-warning"></h3>
            <h3 id="result" class="text-success"></h3>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const processingMessage = document.getElementById('processing');
            const resultMessage = document.getElementById('result');

            // Show processing message
            processingMessage.innerText = "Processing your file...";
            resultMessage.innerText = "";

            try {
                const response = await axios.post('/predict', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                });

                // Display results with animation
                processingMessage.innerText = "";
                resultMessage.innerText = `Predicted Genre: ${response.data.genre}`;
                resultMessage.style.animation = "glowEffect 1.5s infinite alternate";
            } catch (error) {
                processingMessage.innerText = "";
                resultMessage.innerText = `Error: ${error.response.data.error}`;
            }
        });
    </script>
</body>
</html>

"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('uploads', file.filename)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file.save(file_path)

    try:
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        X = preprocess_and_extract_mfcc(signal, sample_rate, num_mfcc, n_fft, hop_length, num_segments)
        prediction = model.predict(X)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_genre = genre_directories[predicted_index]
        os.remove(file_path)  # Clean up uploaded file
        return jsonify({'genre': predicted_genre})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
