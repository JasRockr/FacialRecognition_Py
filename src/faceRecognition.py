
##* Import libraries
import cv2
import os
from dotenv import load_dotenv

##* Define init vars
load_dotenv()
rootPath = os.getenv('ROOT_PATH')
createdModel = os.getenv('CREATED_MODEL_NAME')
modelPath = os.getenv('MODEL_TRAINED_PATH')
trainingModel = os.getenv('TRAINING_MODEL_NAME')
dataPath = os.getenv('DATA_PATH')
videoTestPath = os.getenv('VIDEO_TESTING_PATH')
imagePaths = os.listdir(rootPath + dataPath)
print('Image Path: ', imagePaths)

##* Define filename (Only for video file training)
srcVideo = False
#! Required if srcVideo is defined as True
fileName = 'TestVideo1.mp4'
print(rootPath + videoTestPath + fileName)


##* Define training model
# Eigenfaces Model [*** Assumes images have the same size ***]
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# FisherFaces Model [*** Assumes images have the same size ***]
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
# LBPH Model - (Local Binary Patterns Histograms)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

##* Load the training data
face_recognizer.read(rootPath + modelPath + createdModel)

##* Capture person from video or camera
if srcVideo and fileName != '':
  print('--> Starting video capture from ' + fileName + ' ...')
  cap = cv2.VideoCapture(rootPath + videoTestPath + fileName)
else: 
  print('Warning: The srcVideo variable is not defined!!')
  print('--> Starting video from available camera...')
  cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# ##* Init faces detection with haarcascade model
##* Load classifier from XML file with opencv
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + trainingModel)

while True:
  ret, frame = cap.read()
  if ret == False: 
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  auxFrame = gray.copy()

  ##* Classifier application
    # @params: image, scaleFactor, minNeighbors
  faces = faceClassif.detectMultiScale(gray, 1.3, 5)

  ##* Draw frame for person identify
  for (x,y,w,h) in faces:
    face = auxFrame[y:y + h, x:x + w]
    face = cv2.resize(face,(150,150), interpolation=cv2.INTER_CUBIC)
    ##* Predict(),  predicts the label and trust associated for each face of the input
      # @params: face to recognize
    res = face_recognizer.predict(face)
    ##* Visualization of results
    cv2.putText(frame, '{}'.format(res),(x, y-5), 1, 1.3, (255, 255, 0),1, cv2.LINE_AA)

    '''
    ##* EigenFaces (Res: Label, Trust value)
    if res[1] < 6200:
      cv2.putText(frame, '{}'.format(imagePaths[res[0]]), (x, y-25), 2, 1.1, (255, 255, 0), 1, cv2.LINE_AA)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
      cv2.putText(frame, 'No information about this person', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    '''
    ##* LBPH (Res: Label, Trust value)
    if res[1] < 68.5:
      cv2.putText(frame, '{}'.format(imagePaths[res[0]]), (x, y-25), 2, 1.1, (255, 255, 0), 1, cv2.LINE_AA)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
      cv2.putText(frame, 'No information about this person', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


  cv2.imshow('frame', frame)

  esc = cv2.waitKey(1)
  #27 = Escape keypress
  if esc == 27:
	  break

cap.release()
cv2.destroyAllWindows()