import os
import cv2
import keras
import joblib
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing, svm, metrics
from dataSet import loadAllData, getEmbedding
from faceDetection import getCropedFaces
from predectFace import getCropedFace
from faceDetection import detectFacesInImage,drawSquare
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

class faceRecognition:
    def __init__(self):
        dataSet = 'dataSet'
        # All the images the CNN will train on
        trainData, trainLabel, testData, testLabel, self.uniqueNames = loadAllData(dataSet)
        print('Training data loaded successfully !')

        # Load the FaceNet model "CNN Model"
        self.embeddingModel = keras.models.load_model('facenet_keras.h5')
        print('Embedding Model Loaded successfully')
        if os.path.exists("model.joblib"):
            self.model = joblib.load('model.joblib')
            print('Trained Model Loaded successfully')
            count = 0
            dir = 'dataSet/Fun/'
            for imgFun in tqdm(os.listdir(dir)):
                count+=1
                path = dir + '/' + imgFun
                imgFun = cv2.imread(path, 1)
                faces = detectFacesInImage(imgFun)
                facesDetectedInImg = drawSquare(imgFun, faces)
                for face in faces:
                    column, row, width, height = face['box']
                    column = abs(column)
                    row = abs(row)
                    FaceName, probOfFace, uniqueNames = self.predectImg(imgFun, face)
                    if probOfFace >= 90:
                        cv2.putText(facesDetectedInImg, str(uniqueNames[FaceName]) +  str(int(round(probOfFace))) + '%', (column, row), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (200, 200, 200))
                    else:
                        cv2.putText(facesDetectedInImg, str("UNKOWN"), (column, row), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255))
                cv2.imshow('frame', facesDetectedInImg)
                k = cv2.waitKey(0)
                cv2.imwrite(str(count)+'new.jpg', facesDetectedInImg)

            print('You could now predict any person using this model')
        else:
            # Transform all pictures into 128-D Vector
            embeddedTrainData = []
            for oldFace in tqdm(trainData):
                embeddedFace = getEmbedding(self.embeddingModel, oldFace)
                embeddedTrainData.append(embeddedFace)
            embeddedTrainData = np.array(embeddedTrainData)

            embeddedTestData = []
            for oldFace in tqdm(testData):
                embeddedFace = getEmbedding(self.embeddingModel, oldFace)
                embeddedTestData.append(embeddedFace)
            embeddedTestData = np.array(embeddedTestData)
            print('All Data Embedding Successfully !')

            inputEncoder = preprocessing.Normalizer(norm='l2')
            trainX = inputEncoder.transform(embeddedTrainData)
            testX = inputEncoder.transform(embeddedTestData)
            print('Data Normalized Successfully !')

            outputEncoder = preprocessing.LabelEncoder()
            outputEncoder.fit(trainLabel)

            trainY = outputEncoder.transform(trainLabel)
            testY = outputEncoder.transform(testLabel)
            print('Labels Encoded Successfully !')

            self.model = svm.SVC(kernel='linear', probability=True)
            self.model.fit(trainX, trainY)
            print('Model Trained Successfully !')

            # summarizeq
            predectedTest = self.model.predict(testX)
            score_test = metrics.accuracy_score(testY, predectedTest)
            print('Accuracy: test=%.3fx' % (score_test * 100))



            print('You could now predict any person using this model')
    def predectImg(self, frame, face):
        cropedFace = getCropedFace(frame, face)

        EmbeddedFace = getEmbedding(self.embeddingModel, cropedFace)
        EmbeddedFace = np.array(EmbeddedFace)

        tempArray = []
        tempArray.append(EmbeddedFace)
        tempArray = np.array(tempArray)

        inputEncoder = preprocessing.Normalizer(norm='l2')
        trainX = inputEncoder.transform(tempArray)

        labelOfPerson  = self.model.predict(trainX)
        probOflabels = self.model.predict_proba(trainX)

        probOfPerson = np.amax(probOflabels) * 100

        return labelOfPerson[0], probOfPerson, self.uniqueNames
    def closing(self):
        joblib.dump(self.model,"model.joblib")