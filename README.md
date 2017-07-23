# JARVIS_Security_System

## Main Skeleton:

1. the first phase is to detect "hello jarvis"

2. the second phase is to confirm with the user whether the face detected matches the user.
    - JARVIS says: Are you <username>?
    - user says: <yes or no>
    if yes, the user may proceed to password verification part, else, user remain here until the correct response shows

        password verification:
        - user has to draw out a sequence of number on the screen

3. the third phase is to verify user's password with JARVIS
    - <open app or google query>
      - app: calendar, photo booth, and calculator; query: any search query for google


## For Face Recognition:

1. Go to facerec/ folder

2. Run "sudo python FR_train.py <name>" to add new images to training set + retrain neural net

3. run "sudo python FR_test.py" to see your predicted results

NOTE:

a. Make sure this facerec/ folder has "shape_predictor_68_face_landmarks.dat" file, otherwise download it (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

b. Make sure the caffe_root path is properly set to your own caffe root file in FR_NN_train.py and FR_NN_test.py (and "NN_demo.py")

c. Make sure the labels.txt files in both train/ and test/ folder point to the correct absolute image path

d. These files have NOT been tested yet!

e. There is a bug where the last image isn't saved, but the label gets put to the labels.txt file

f. You can run "sudo python NN_demo.py" to see how the system should work once integrated properly.

## For Speech Detection:

1. run python speech_main.py

2. JARVIS will response start hearing the user when the user speaks or if the room is loud enough. If the room is quiet, then JARVIS will not start hearing.

    
