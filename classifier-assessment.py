from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import cv2
import numpy as np
import seaborn
import matplotlib.pyplot as plt

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('./lbph_classifier.yml')

paths = [os.path.join('./resources/train', f) for f in os.listdir('./resources/train')]
predictions = []
expected_exits = []

for path in paths:
    # print(path)
    #  CONVERT IMAGE TO GRAY SCALE
    image = Image.open(path).convert('L')
    np_image = np.array(image, dtype=np.uint8)
    prediction, _ = lbph_face_classifier.predict(np_image)
    expected_exit = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))

    predictions.append(prediction)
    expected_exits.append(expected_exit)

predictions = np.array(predictions)
expected_exits = np.array(expected_exits)

accuracy = accuracy_score(expected_exits, predictions)
confusion = confusion_matrix(expected_exits, predictions)

seaborn.heatmap(confusion, annot=True)

plt.show()

# test_image = './resources/train/subject01.normal.gif'
#
# image = Image.open(test_image).convert('L')
# np_image = np.array(image, dtype=np.uint8)
# prediction = lbph_face_classifier.predict(np_image)
#
# expected_exit = int(os.path.split(test_image)[1].split('.')[0].replace('subject', ''))
#
# cv2.putText(np_image, 'Pred: ' + str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
# cv2.putText(np_image, 'Exp: ' + str(expected_exit), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
# cv2.imshow('image', np_image)
#
# cv2.waitKey(0)