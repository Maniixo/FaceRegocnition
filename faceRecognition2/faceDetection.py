import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Train the Model
detector = MTCNN()
def detectFacesInImage(image):
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2RGB)
    # Find faces using CNN
    facesCoordinates = detector.detect_faces(image)

    return facesCoordinates
def getCropedFaces(image, label):
    facesDetected = detectFacesInImage(image)
    exactFace = None
    for i in range(len(facesDetected)):
        column, row, width, height = facesDetected[i]['box']
        # Fixing error of negative
        column = abs(column)
        row = abs(row)

        exactFace = image[row:row + height, column:column + width]
        exactFace = cv2.resize(exactFace, (160, 160))
        exactFace = np.array(exactFace)

    faceInImage = np.array(exactFace)
    return faceInImage, label
def drawSquare(img, facesDetected):
    for i in range(len(facesDetected)):
        column, row, width, height = facesDetected[i]['box']
        column = abs(column)
        row = abs(row)
        cv2.rectangle(img, (column, row), (column + width, row + height), (255, 255, 255), 1)
    return img