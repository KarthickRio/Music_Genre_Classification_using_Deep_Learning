# 🎵 Music Genre Classification using CNN & RNN

This project explores the performance of **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)** for music genre classification. Both models are evaluated based on their accuracy in predicting genres from audio data.

**[Download Project Slides 📊](https://github.com/KarthickRio/Music_Genre_Classification_using_Deep_Learning/blob/main/Music%20Genre%20Classification.pptx)**

## 📂 Dataset

The dataset used for this project is the **GTZAN dataset**, which contains 1000 audio tracks, each 30 seconds long, categorized into **10 genres** (e.g., Classical, Jazz, Rock, Pop, etc.). The audio files are in WAV format, and the dataset is widely used for music genre classification research. The tracks are pre-labeled, making it ideal for supervised learning tasks.

## 🧠 Model Performance

### CNN Model:
The CNN model achieved an accuracy of **77%** on the test set. The CNN excels in capturing local spectral patterns within the spectrogram, which is critical for identifying genre-specific characteristics in audio. You can find the model file for CNN [here 🔗](https://github.com/KarthickRio/Music_Genre_Classification_using_Deep_Learning/blob/main/cnn_genre.py).

### RNN Model:
The RNN model, which was designed to capture temporal dependencies in the audio data, achieved a slightly lower accuracy of **66%**. While RNNs are great for sequence modeling, they didn't perform as well as CNNs for this task, possibly due to the nature of the audio data and the way features are captured.

## 📊 Accuracy Comparison
The following graph compares the performance of the CNN and RNN models on the training and validation sets. The CNN model outperforms the RNN.

## ✅ Conclusion

- **CNN Model**: Achieved **77%** accuracy and showed better performance in genre classification tasks, capturing local spectral features effectively.
- **RNN Model**: Achieved **66%** accuracy, but showed limitations in capturing complex genre patterns compared to CNNs.

## 📝 Code and Files
The code for both models is available in the repository.







