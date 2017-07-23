from array import array
from struct import pack
from sys import byteorder
import copy
import pyaudio
import wave
from collections import deque
import math
import audioop
import wavrecorder

THRESHOLD = 400  # audio levels not normalised.
CHUNK_SIZE = 1024
SILENT_CHUNKS = 3 * 44100 / 1024  # about 3sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
RATE = 44100
CHANNELS = 1
TRIM_APPEND = RATE / 4
PREV_AUDIO = 0.5
done_record = False

def is_silent(data_chunk):
    """Returns 'True' if below the 'silent' threshold"""
    return max(data_chunk) < THRESHOLD

def normalize(data_all):
    """Amplify the volume out to max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r

def trim(data_all):
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break

    return copy.deepcopy(data_all[_from:(_to + 1)])

def record(threshold=THRESHOLD, num_phrases=-1):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print "* Listening mic. "
    audio2send = []
    cur_data = ''  # current chunk  of audio data
    rel = RATE/CHUNK_SIZE
    slid_win = deque(maxlen=SILENT_CHUNKS * rel)
    #Prepend audio from 0.5 seconds before noise was detected
    started = False
    n = num_phrases
    response = []
    slid_win.append(0)
    slid_win.append(0)
    slid_win.append(0)

    while (num_phrases == -1 or n > 0):
        cur_data = stream.read(CHUNK_SIZE)
        slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))
        if( (slid_win[-1]+slid_win[-2]+slid_win[-3]+slid_win[-4])/4.0 >threshold):
            # if(not started):
            # print "Starting record of phrase"
            wavrecorder.start_record()

            started = False
            slid_win = deque(maxlen=SILENT_CHUNKS * rel)
            audio2send = []
            done_record = True
            # print "break please"
            # threshold = 500
            break

    # print "* Done recording"
    stream.close()
    p.terminate()

# def record_to_file(data):
#     "Records from the microphone and outputs the resulting data"
#     sample_width = 2
#     data = ''.join(data)
#     wave_file = wave.open('demo.wav', 'wb')
#     wave_file.setnchannels(CHANNELS)
#     wave_file.setsampwidth(sample_width)
#     wave_file.setframerate(RATE)
#     wave_file.writeframes(data)
#     wave_file.close()

# print("Wait in silence to begin recording; wait in silence to terminate")
# if(not done_record):
#     record()
# print("done - result written to demo.wav")

# comment if wanna make function
# if __name__ == '__main__':
#     print("Wait in silence to begin recording; wait in silence to terminate")
#     if(not done_record):
#         record()
#     print("done - result written to demo.wav")
