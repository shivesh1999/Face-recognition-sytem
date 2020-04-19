# import the necessary packages
from imutils.video import VideoStream #for video stream
from imutils.video import FPS
from playsound import playsound # for playing sounds
from gtts import gTTS # for text to speech
import numpy as np 
import pyautogui
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#for text to speech
language = 'en'
mytext = 'Hello'
myobj1 = gTTS(text=mytext, lang=language, slow=False) 
myobj1.save("hello.mp3")
mytext2 = 'welcome'
myobj2 = gTTS(text=mytext2, lang=language, slow=False) 
myobj2.save("welcome.mp3")
mytext3='please wait for the entry permissions'
myobj3 = gTTS(text=mytext3, lang=language, slow=False)
myobj3.save("wait.mp3")

# load our serialized face detector from disk
print("Activating...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up 
print("Starting application...")
vs = VideoStream(src=0).start()
time.sleep(2.0) #wait for 2 secounds
# start the FPS 
fps = FPS().start()
playsound('hello.mp3')
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frame = imutils.resize(frame, width=800)
	(h, w) = frame.shape[:2]
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
				# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			percent = proba*100
			name = le.classes_[j] 
			# draw the bounding box of the face  
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			if percent > 60 :
				text2 = 'Access granted'
				cv2.putText(frame, text2, (startX, y + 350),
				 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			else:
				text3 = 'Access Denied'
				cv2.putText(frame, text3, (startX, y + 350),
				 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	# update the FPS counter
	fps.update()
	# show the output frame
	cv2.imshow("CAMERA1", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("Closing application...")
		break
# stop FPS information
fps.stop()
cv2.destroyAllWindows()
vs.stop()