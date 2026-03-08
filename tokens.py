import tensorflow as tf
from nltk.tokenize import word_tokenize

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