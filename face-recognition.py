from PIL import Image
import cv2
import numpy as np
import os


# PRÉ PROCESSAMENTO DAS IMAGENS
def get_image_data():
    paths = [os.path.join('./resources/train', f) for f in os.listdir('./resources/train')]
    faces: list[str] = []
    ids: list[int] = []
    for path in paths:
        image = Image.open(path).convert('L')
        image_np = np.array(image, 'uint8')
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
        ids.append(id)
        faces.append(image_np)
    return np.array(ids), faces


ids, faces = get_image_data()

# TREINAMENTO DO CLASSIFICADOR

lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=5, grid_x=5, grid_y=5)
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')

# RECONHECIMENTO DAS FACES

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('./lbph_classifier.yml')

test_image = './resources/train/subject01.normal.gif'

image = Image.open(test_image).convert('L')
np_image = np.array(image, dtype=np.uint8)
prediction = lbph_face_classifier.predict(np_image)

expected_exit = int(os.path.split(test_image)[1].split('.')[0].replace('subject', ''))

cv2.putText(np_image, 'Pred: ' + str(prediction[0]), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.putText(np_image, 'Exp: ' + str(expected_exit), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.imshow('image', np_image)

cv2.waitKey(0)
