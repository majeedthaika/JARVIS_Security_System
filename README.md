# JARVIS_Security_System

1. Go to facerec/ folder

2. Run "python FR_train.py <name>" to add new images to training set + retrain neural net

3. run "python FR_test.py" to see your predicted results

NOTE:

a. Make sure this facerec/ folder has "shape_predictor_68_face_landmarks.dat" file, otherwise download it

b. Make sure the caffe_root path is properly set to your own caffe root file in FR_NN_train.py and FR_NN_test.py

c. Make sure the labels.txt files in both train/ and test/ folder point to the correct absolute image path

d. These files have NOT been tested yet!
