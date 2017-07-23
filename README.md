# JARVIS_Security_System

1. Go to facerec/ folder

2. Run "sudo python FR_train.py <name>" to add new images to training set + retrain neural net

3. run "sudo python FR_test.py" to see your predicted results

NOTE:

a. Make sure this facerec/ folder has "shape_predictor_68_face_landmarks.dat" file, otherwise download it (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

b. Make sure the caffe_root path is properly set to your own caffe root file in FR_NN_train.py and FR_NN_test.py

c. Make sure the labels.txt files in both train/ and test/ folder point to the correct absolute image path

d. These files have NOT been tested yet!

e. There is a bug where the last image is saved, but the label gets put to the labels.txt file
