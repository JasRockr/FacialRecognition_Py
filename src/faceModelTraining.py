
##* Import libraries
import cv2
import os
import numpy as np
from dotenv import load_dotenv

##* Define init vars
load_dotenv()
rootPath = os.getenv('ROOT_PATH')
createdModel = os.getenv('CREATED_MODEL_NAME')
modelPath = os.getenv('MODEL_TRAINED_PATH')
dataPath = os.getenv('DATA_PATH')
peopleList = os.listdir(rootPath + dataPath)
print('List Persons: ', peopleList)
print('Root Path: ', rootPath)
print('Model Path: ', modelPath)


labels = []
facesData = []
label = 0 

##* Iterate the folders of each person
for nameDir in peopleList:
  personPath = rootPath + dataPath + nameDir
  print('Reading images ... ')

  ##* Iterate the images of each person
  for fileName in os.listdir(personPath):
    print('Faces: ', nameDir + '--> ' + fileName)
    labels.append(label)

    ##* Load images read to the array
    ##* Conversion of images to gray scale
    facesData.append(cv2.imread(personPath + '/' +fileName, 0))
    image = cv2.imread(personPath + '/' + fileName, 0)

    ##### - Testing
    # cv2.imshow('Image', image)
    #cv2.waitKey(10)
    #####
  label += 1
# cv2.destroyAllWindows()

##### - Testing count of images
# print('Labels: ', labels)
# print('Tags with 0: ', np.count_nonzero(np.array(labels)==0))
# print('Tags with 1: ', np.count_nonzero(np.array(labels)==1))

##* Define training model
# Eigenfaces Model [*** Assumes images have the same size ***]
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# FisherFaces Model [*** Assumes images have the same size ***]
# face_recognizer = cv2.face.FisherFaceRecognizer_create() ##! Error training
# LBPH Model - (Local Binary Patterns Histograms)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

##* Start training with defined model
print('Training ... ')
  ##@params train: (set training images, image tags[array labels])
face_recognizer.train(facesData, np.array(labels))

##* Save trained model
print('Saving model... ')
  ##@params write: (model name to create **XML/YAML file**)
face_recognizer.write(rootPath + modelPath + createdModel)
print('Saved model successfully.')