from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

data_path = 'dataset/'
only_faces = [f for f in listdir(data_path) if isfile(join(data_path, f))]

training_data, labels = [], []

for i, file in enumerate(only_faces):
    image_path = join(data_path, file)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images, dtype=np.uint8))
    labels.append(i)

labels = np.asarray(labels, dtype=np.int32)

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(training_data, labels)

recognizer.save('lbph_model.yml')

print("Dataset model training completed")
