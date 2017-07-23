import command

print "detect_jarvis"

def is_jarvis(st):
    speech = st.lower()
    print "speech", speech
    if "hello" in speech:
        print "authenticated"
        return True
    else:
        print "not"
        return False

def is_right_user(st):
    speech = st.lower()
    if "yes" in speech:
        return "Yes"
    elif "no" in speech:
        return "No"
    else:
        repeat = "could you repeat that please"
        command.say_word(repeat)
        return "I do not understand"
