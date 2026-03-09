import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

# Importy z Twoich plików
from text_processing import text_processing
from tokens import tokens_feeding

# --- KONFIGURACJA ---
PATH_CAPTIONS = r'.\captions.txt'
# Zmień tę nazwę na zdjęcie, które chcesz sprawdzić!
TEST_IMAGE = r'.\\Images\\1000268201_693b08cb0e.jpg' 
MODEL_PATH = 'best_model_final_epoch_50.keras' # Plik, który zapisałeś po treningu

# --- 1. WCZYTANIE TOKENIZERA ---
# Musimy go odtworzyć, żeby model wiedział, co oznaczają liczby
mapping = text_processing(PATH_CAPTIONS)
tokenizer = tokens_feeding(mapping)
vocab_size = len(tokenizer.word_index) + 1

# --- 2. WCZYTANIE WYTRENOWANEGO MODELU ---
print("Wczytywanie modelu...")
model = load_model(MODEL_PATH)

# --- 3. PRZYGOTOWANIE SKANERA (INCEPTION) ---
base_model = InceptionV3(weights='imagenet')
feat_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def generate_caption(model, tokenizer, photo_path, max_length=38):
    # Ekstrakcja cech dla zdjęcia
    img = load_img(photo_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    photo_feature = feat_model.predict(img, verbose=0)

    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text

# --- 4. URUCHOMIENIE I WYŚWIETLENIE ---
if os.path.exists(TEST_IMAGE):
    print(f"Generowanie opisu dla: {TEST_IMAGE}...")
    result = generate_caption(model, tokenizer, TEST_IMAGE, 38)
    
    # Wyświetlanie wyniku
    img = mpimg.imread(TEST_IMAGE)
    plt.imshow(img)
    plt.title(f"Wynik: {result}")
    plt.axis('off')
    print(f"\nGotowe! Opis: {result}")
    plt.show()
else:
    print(f"Nie znaleziono zdjęcia: {TEST_IMAGE}")