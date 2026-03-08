from text_processing import text_processing
from tokens import tokens
from text_processing import *
from tokens import *
from tensorflow.keras.applications.inception_v3 import InceptionV3

model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')

dictionary = text_processing(r'.\\captions.txt')
tokenizer = tokens_feeding(dictionary)
dictionary = tokens(dictionary)
final = tokens_with_padding(dictionary, tokenizer)