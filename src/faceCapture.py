
##* Import libraries
import cv2
import os
import imutils
from dotenv import load_dotenv

##* Define init vars
load_dotenv()
rootPath = os.getenv('ROOT_PATH')
trainingModel = os.getenv('TRAINING_MODEL_NAME')
personName = os.getenv('PERSON_NAME')
dataPath = os.getenv('DATA_PATH')
videoTrainingPath = os.getenv('VIDEO_TRAINING_PATH')

##* Define filename (Only for video file training)
srcVideo = True
#! Required if srcVideo is defined as True
fileName = 'Video1.mp4'

##* Define limits for counter
startCount = 0
endCount = 300

# TODO: Define a variable path!!
personPath = rootPath + dataPath + personName
videoPath = rootPath + videoTrainingPath + fileName
print('Path Person: ', personPath)
print('Path Video: ', videoPath)

##* Create required folder paths
if not os.path.exists(personPath):
  print('The folder has been successfully created in: ', personPath)
  os.makedirs(personPath)
else: 
	print('There is already a folder for this person!!')



##* Capture person from video or camera
if srcVideo and fileName != '':
  print('--> Starting video capture from ' + fileName + ' ...')
  cap = cv2.VideoCapture(videoPath)
else: 
  print('Warning: The srcVideo variable is not defined!!')
  print('--> Starting video from available camera...')
  cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# ##* Init faces detection with haarcascade model
##* Load classifier from XML file with opencv
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + trainingModel)

##* Define counter for while loop
count = startCount
print('Saving images of faces... ')

while True:
  ret, frame = cap.read()
  if ret == False: 
    break

  frame = imutils.resize(frame, width=640)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  auxFrame = frame.copy()

  ##* Classifier application
  # @params: image, scaleFactor, minNeighbors
  faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    # scaleFactor=1.3,
    # minNeighbors=5,
    # minSize=(30, 30),
    # maxSize=(200,200))

  ##* Draw frame for person identify
  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    face = auxFrame[y:y + h, x:x + w]
    face = cv2.resize(face,(150,150), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(personPath + '/face_{}.jpg'.format(count), face)
    count += 1
  
  cv2.imshow('frame', frame)

  esc = cv2.waitKey(1)
  #27 = Escape keypress
  if esc == 27 or count >= endCount:
	  break

print('Saved Images: ', count)
cap.release()
cv2.destroyAllWindows()
