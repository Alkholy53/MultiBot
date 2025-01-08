import speech_recognition as sr
import pyttsx3
import streamlit as st

# Speech recognition and synthesis
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Capture audio input
def get_audio_input():
    with sr.Microphone() as source:
        st.write("Please speak:")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"

# Text-to-speech output
def speak_text(text):
    engine.say(text)
    engine.runAndWait()