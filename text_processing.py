import pandas as pd
import nltk
import re
import tensorflow as tf

nltk.download('punkt')

def tokens(caption):

    caption2 = re.sub(r'[^\w\s]', '', caption)

    caption2 = 'startseq ' + caption2 + ' endseq'

    return caption2

def text_processing(file_path):

    mapping = {}

    df = pd.read_csv(file_path, names=['image', 'caption'])

    for row in df.values:
        img_name = row[0]
        caption = row[1]

        caption = caption.lower()

        if img_name not in mapping:
            mapping[img_name] = []
        mapping[img_name].append(tokens(caption))

    return mapping
