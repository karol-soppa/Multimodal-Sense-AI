import os
import sys
import cv2

def data_import(path):
    directory = path
    img_array = []
    img_label = []

    for img in os.listdir(path):
        image_path = os.path.join(path, img)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is not None:
            new_image = cv2.resize(image, (299,299))
            img_array.append(new_image)
            img_label.append(img)

    return img_array, img_label



