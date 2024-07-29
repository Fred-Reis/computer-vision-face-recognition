import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor('./resources/shape_predictor_68_face_landmarks.dat')

image = cv2.imread('./resources/people2.jpg')
detections = face_detector(image, 1)

for face in detections:
    points = points_detector(image, face)

    for point in points.parts():
        cv2.circle(image, (point.x, point.y), 2, (0, 255, 0), 1)


    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 255), 2)

cv2.imshow('image', image)
cv2.waitKey(0)