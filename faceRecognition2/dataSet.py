import os
import cv2
import keras
import random
import numpy as np
import skimage as sk
from tqdm import tqdm
from skimage import util
from faceDetection import getCropedFaces
from scipy import ndarray
from skimage import transform

def randomRotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    randomDegree = random.uniform(-25, 25)
    return np.array(sk.transform.rotate(image_array, randomDegree))
def randomNoise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)
def noisedRotation(image_array: ndarray):
    # add random noise to the rotated image
    randomDegree = random.uniform(-25, 25)
    rotatedImage = sk.transform.rotate(image_array, randomDegree)
    return sk.util.random_noise(rotatedImage)

def loadAllData(dataFolder):
    trainData, trainLabel, testData, testLabel, uniqueNames = loadData(dataFolder)

    return trainData, trainLabel, testData, testLabel, uniqueNames
def loadData(directory):
    trainData = []
    trainLabel = []
    uniqueNames = []
    testData = []
    testLabel = []

    print("LOADING TRAINING DATA FOR ALL CLASSES !")
    trainFolder = directory + '/train/'
    for personName in os.listdir(trainFolder):
        personFolder = trainFolder + personName
        i = 1
        for image in tqdm(os.listdir(personFolder)):
            path = personFolder + '/' + image
            img = cv2.imread(path, 1)
            cropedFace, cropedLabel = getCropedFaces(img, personName)
            trainData.append(cropedFace)
            trainLabel.append(personName)

            rotatedFace = randomRotation(cropedFace)
            trainData.append(rotatedFace)
            trainLabel.append(personName)

            noisedImg = randomNoise(cropedFace)
            trainData.append(noisedImg)
            trainLabel.append(personName)
            i = i + 1
        print('loaded ' + str(i - 1) +  ' training examples for class: ' + personName )
        uniqueNames.append(personName)

    print("LOADING TESTING DATA FOR ALL CLASSES !")
    testFolder = directory + '/test/'
    for personName in os.listdir(testFolder):
        personFolder = testFolder + personName
        i = 1
        for image in os.listdir(personFolder):
            path = personFolder + '/' + image
            img = cv2.imread(path, 1)
            cropedFace, cropedLabel = getCropedFaces(img, personName)
            testData.append(cropedFace)
            testLabel.append(personName)
            i = i + 1
        print('loaded ' + str(i - 1) + ' testing examples for class: ' + personName)

    trainData = np.array(trainData)
    trainLabel = np.array(trainLabel)
    testData = np.array(testData)
    testLabel = np.array(testLabel)

    return trainData, trainLabel, testData, testLabel, uniqueNames
def getEmbedding(model, face):
    face = face.astype('float32')

    # standardize pixel values across channels (global)
    mean, std = face.mean(), face.std()
    face = (face - mean) / std

    # transform face into one sample
    samples = keras.backend.expand_dims(face, axis=0)

    # make prediction to get embedding
    yhat = model.predict(samples, steps= 1)
    return yhat[0]
