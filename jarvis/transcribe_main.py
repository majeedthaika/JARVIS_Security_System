#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the REST API for batch
processing.
Example usage:
    python transcribe.py resources/audio.raw
    python transcribe.py gs://cloud-samples-tests/speech/brooklyn.flac
"""

# [START import_libraries]
# import argparse
import io
import detect_jarvis
import command
# [END import_libraries]

def transcribe_jarvis(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    speech_client = speech.Client()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
        audio_sample = speech_client.sample(
            content=content,
            source_uri=None,
            encoding='LINEAR16')

    alternatives = audio_sample.recognize('en-US')
    for alternative in alternatives:
        text = alternative.transcript
        return detect_jarvis.is_jarvis(text)

def transcribe_confirm(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    speech_client = speech.Client()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
        audio_sample = speech_client.sample(
            content=content,
            source_uri=None,
            encoding='LINEAR16')

    alternatives = audio_sample.recognize('en-US')
    for alternative in alternatives:
        text = alternative.transcript
        return detect_jarvis.is_right_user(text)

def transcribe_feature(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    speech_client = speech.Client()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
        audio_sample = speech_client.sample(
            content=content,
            source_uri=None,
            encoding='LINEAR16')

    alternatives = audio_sample.recognize('en-US')
    for alternative in alternatives:
        text = alternative.transcript
        print 'feature', text
        return command.perform_task(text)

def get_jarvis():
    voice_file = "demo.wav"
    return transcribe_jarvis(voice_file)

def get_confirm():
    voice_file = "demo.wav"
    return transcribe_confirm(voice_file)

def get_feature():
    voice_file = "demo.wav"
    return transcribe_feature(voice_file)
