import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model
from tqdm import tqdm

# Importy z Twoich plików
from text_processing import text_processing
from tokens import tokens_feeding, tokens

# --- KONFIGURACJA ŚCIEŻEK ---
PATH_CAPTIONS = r'.\captions.txt'
PATH_IMAGES = r'.\Images' 
TEST_IMAGE = r'.\Images\1000268201_693b08cb0e.jpg'

# --- 1. PRZYGOTOWANIE DANYCH TEKSTOWYCH ---
print("Przygotowywanie tekstu...")
mapping = text_processing(PATH_CAPTIONS)
tokenizer = tokens_feeding(mapping)
mapping_sequences = tokens(mapping) 
vocab_size = len(tokenizer.word_index) + 1

# --- 2. EKSTRAKCJA CECH ---
def get_all_features(directory):
    if os.path.exists('features.pkl'):
        print("Wczytywanie gotowych cech z pliku...")
        return pickle.load(open('features.pkl', 'rb'))
    
    print("Ekstrakcja cech z obrazów (to potrwa)...")
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    features = {}
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[img_name] = feature.reshape(-1)
    
    pickle.dump(features, open('features.pkl', 'wb'), protocol=4)
    return features

all_features = get_all_features(PATH_IMAGES)

# --- 3. GENERATOR DANYCH (Zaktualizowany o krotki/tuples) ---
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            if key not in photos: continue
            n += 1
            photo = photos[key]
            for desc in desc_list:
                for i in range(1, len(desc)):
                    in_seq, out_seq = desc[:i], desc[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                # Zmieniono na krotki (tuples), aby uniknąć TypeError w TF
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

# --- 4. BUDOWA I TRENING MODELU ---
def build_model(vocab_size, max_length=38):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = build_model(vocab_size, 38)

# ZWIĘKSZONA LICZBA EPOK
epochs = 50 
batch_size = 32
steps = len(mapping_sequences) // batch_size

# Tworzenie podpisu wyjściowego (Output Signature) dla TensorFlow
output_signature = (
    (
        tf.TensorSpec(shape=(None, 2048), dtype=tf.float32), # Cechy obrazu
        tf.TensorSpec(shape=(None, 38), dtype=tf.float32)    # Sekwencje tekstu
    ),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32) # Wyjście (słowo)
)

# Konwersja generatora na Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(mapping_sequences, all_features, tokenizer, 38, vocab_size, batch_size),
    output_signature=output_signature
)

# DODANY MODEL CHECKPOINT
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_best_so_far.keras', 
    monitor='loss',                     
    save_best_only=True,                
    mode='min',                         
    verbose=1                           
)

print(f"Rozpoczynam trening na {epochs} epok z automatycznym zapisem najlepszych wyników...")

# Trening z użyciem callbacks
model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks=[checkpoint])

# Zapisanie modelu na samym końcu dla pewności
model.save('best_model_final_epoch_50.keras') 

# --- 5. GENEROWANIE OPISU PO TRENINGU ---
def generate_caption(model, tokenizer, photo_path, max_length=38):
    # Załadowanie InceptionV3 do predykcji jednego zdjęcia
    base_model = InceptionV3(weights='imagenet')
    feat_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
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

print("\n--- TEST PO TRENINGU ---")
if os.path.exists(TEST_IMAGE):
    result = generate_caption(model, tokenizer, TEST_IMAGE, 38)
    print(f"Wygenerowany opis: {result}")
else:
    print(f"Błąd: Nie znaleziono pliku testowego pod ścieżką {TEST_IMAGE}")