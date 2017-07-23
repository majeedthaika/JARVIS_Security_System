import record_detect
import transcribe_main
import time


def detect_hello():
    while True:
        """detect 'jarvis' to continue"""
        try:
            record_detect.record()
            jarvis = transcribe_main.get_jarvis()

            if jarvis:
                break

        except:
            pass

    """facial recognition"""

    ### want to confirm user


    time.sleep(3)

def confirm_user():
    print "entering phase 2"
    while True:
        """are you <user's name>? """
        try:
            record_detect.record()
            rs = transcribe_main.get_confirm()
            if rs == "Yes":
                print "yes"
                """ tw part """
                return True
            if rs == "No":
                print "no"
                """ mj part """
                return False
            else:
                print "can u repeat that please"
        except:
            pass

    time.sleep(3)

def get_command():
    print "entering phase 3"
    while True:
        """what do you want to do? <open app> or <google query>"""
        try:
            record_detect.record()
            success = transcribe_main.get_feature()

            if success:
                break

        except:
            pass

def jarvis_main():

    detect_hello()

    # mj's face

    right_user = confirm_user()
    while not (right_user):
        """call mj detect face 'r u <new username>'?
           since the last one was wrong
           mj will give name, and indian jarvis will ask"""
        right_user = confirm_user()

    """verify password from tw, if yes"""
    get_command()



for i in range(2):
    jarvis_main()
