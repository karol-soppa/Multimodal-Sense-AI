import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokens_feeding(dictionary):

    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>")
    all_captions = []
    for value in dictionary.values():
        all_captions.extend(value)
    tokenizer.fit_on_texts(all_captions)

    return tokenizer

def tokens(dictionary):

    tokenizer = tokens_feeding(dictionary)
    for key, value in dictionary.items():
        dictionary[key] = tokenizer.texts_to_sequences(value)
    
    return dictionary

def tokens_with_padding(dictionary, tokenizer):

    for key, value in dictionary.items():
        padded = pad_sequences(value, maxlen=38, padding='post')
        dictionary[key] = padded
        
    return dictionary
