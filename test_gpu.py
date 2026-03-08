from text_processing import text_processing
from tokens import tokens
from text_processing import *
from tokens import *


def get_max_length(dictionary):
    max_l = 0
    for captions in dictionary.values():
        for cap in captions:
            if len(cap) > max_l:
                max_l = len(cap)
    return max_l

#dictionary = text_processing('C:\\Users\\Karol\\Documents\\nauka_ai\\Multimodal-Sense-AI\\Multimodal-Sense-AI\\captions.txt')
#dictionary = tokens(dictionary)

#max_l = get_max_length(dictionary)
#print(f"Najdłuższe zdanie ma: {max_l} tokenów")

dictionary = text_processing(r'.\\captions.txt')
tokenizer = tokens_feeding(dictionary)
dictionary = tokens(dictionary)
final = tokens_with_padding(dictionary, tokenizer)