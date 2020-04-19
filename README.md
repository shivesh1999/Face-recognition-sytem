# Face-recognition-sytem
This project gives you a face detection system using python and openCV.
to build face recognitio system-
1.detection
2.extract face embeddings
3.train face reciognizer on embeddings
4. recognizing python program

we are using deep learning in detecting face from video and extracting features of the face

deep learning model extract embeddings of the face that mean the qualities of a particular face

face embedding are extracted by the following process- three images are fed up by the user psitive and anchor are the face that need to be detected and the negative image is some random face. the model compare sthe weights of the image where anchor and positive are of same weight and negative is farther away from the range.

we are passing three dataset omne is me another is my brother and one unknown person  that will help us to label the people who are not in the direvtory 

now coming on the files that are needed
dataset- it is a file consisting of folder that are named to the people, consisting of 6 images 
images- test images 
face_detection_model- pretrained model to detect faces using deep learning
output- conssits of output pickle.the output files include 
embeddings.pickle-after embeddings have been computed it will be stored in this file.
le.pickle-contains the name label of the people who will be recognized 
recognizer,pickle-our support vector machine(svm) that is depp learning model to recognize faces

there five files in root directory 

extract_embeddings.py-responsible for making ambeddings using deep learning for every face in dataset

openface_nn4.small2.v1.t7- deep learning model that generates 128d embedding model

train_model.py-our svmm model will be trained by the data collected in previous scripts.

recognize_video.py- a python program that is used to integrate everything that is done and recognize a face in live video stream

step 1 : extract embeddings from face data 
arguments passed-dataset,embeddings.pickle,detector,embedding-model
step 2 : train face recognition model 
arguments passed - embeddings,recognizer,le
step 3 : recognize faces with openCV
arguments paased- detector,embeddibng-model,recognizer,le



