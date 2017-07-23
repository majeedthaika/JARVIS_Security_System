import subprocess
import os
import webbrowser

def say_word(word):
    """for osx"""
    # os.system('say -v Veena %s'%word)
    """for linux"""
    os.system('echo %s|espeak'%word)

def perform_task(st):
    """cant open app with linux"""
    speech = st.lower()
    if 'google' in speech:
        query = ' '.join(speech.split()[1:])
        webbrowser.open('https://google.com/search?q='+query)
    elif 'open' in speech:
        task = '\ '.join(speech.split()[1:])
        os.system('open /Applications/%s.app'%task)
    else:
        say_word("Unknown command")
        return False

    return True
