import os
import numpy as np
import librosa
import kagglehub
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt # <-- DODANA BIBLIOTEKA DO RYSOWANIA

from music_data_processing import data_processing, extract_spectrogram
from music_neural_network import msn

path = kagglehub.dataset_download("jbuchner/synthetic-speech-commands-dataset")

DATASET_PATH = path 
DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
SAMPLES_PER_DIGIT = 1000  
IMG_HEIGHT = 64

X_train, X_test, y_train, y_test, input_shape, DIGITS = data_processing(DATASET_PATH, DIGITS, SAMPLES_PER_DIGIT, IMG_HEIGHT)

model = msn(input_shape, DIGITS)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# --- 4. TRENOWANIE ---
print(f"\nRozpoczynam trenowanie na {len(X_train)} próbkach...")
model.fit(X_train, y_train, epochs=15, batch_size=32, 
          validation_data=(X_test, y_test))

# --- 5. FINALNY WYNIK ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ FINALNE ACCURACY: {acc*100:.2f}%")

model.save("audio_model_v1.h5")
print("Model zapisany.")

# --- 6. PREDYKCJA NA TWOIM PLIKU ---
# Podwójne ukośniki są jak najbardziej poprawne!
path_to_your_audio = "C:\\Users\\Karol\\Documents\\nauka_ai\\Multimodal-Sense-AI\\Multimodal-Sense-AI\\Nagrywanie.wav"
test = extract_spectrogram(path_to_your_audio, IMG_HEIGHT)
test_ready = np.expand_dims(test, axis=0)
    
# Wykonanie predykcji
predictions = model.predict(test_ready)
print(predictions)

# Znalezienie indeksu z najwyższym prawdopodobieństwem
predicted_index = np.argmax(predictions)
print(f"Predykcja sieci (indeks): {predicted_index}")

# --- 7. WIZUALIZACJA I PORÓWNANIE (NOWY KOD) ---
print("\nGenerowanie porównania spektrogramów...")

# Szukanie przykładowego pliku "one" z datasetu do porównania
dataset_one_path = None
actual_data_path = DATASET_PATH

# Upewnienie się, że jesteśmy we właściwym folderze
for root, dirs, files in os.walk(DATASET_PATH):
    if 'one' in dirs:
        actual_data_path = root
        break

one_folder = os.path.join(actual_data_path, 'one')
if os.path.exists(one_folder):
    sample_files = [f for f in os.listdir(one_folder) if f.lower().endswith('.wav')]
    if sample_files:
        dataset_one_path = os.path.join(one_folder, sample_files[0])

# Jeśli znalazło plik w datasecie, generujemy obrazek
if dataset_one_path and test is not None:
    dataset_spec = extract_spectrogram(dataset_one_path, IMG_HEIGHT)
    
    plt.figure(figsize=(14, 5))
    
    # Wykres 1: Plik z datasetu
    plt.subplot(1, 2, 1)
    plt.title("Idealne 'ONE' z datasetu")
    # Używamy squeeze(), żeby zrzucić ostatni wymiar (np. z (64, 32, 1) na (64, 32))
    plt.imshow(dataset_spec.squeeze(), origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Pasma Melowe (Częstotliwość)')
    plt.xlabel('Czas')

    # Wykres 2: Twoje nagranie
    plt.subplot(1, 2, 2)
    plt.title("Twój głos ('Nagrywanie.wav')")
    plt.imshow(test.squeeze(), origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Czas')

    plt.tight_layout()
    plt.show()
else:
    print("Nie udało się załadować obu plików do porównania.")