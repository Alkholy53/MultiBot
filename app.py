import streamlit as st
from src.image_caption import generate_caption
from src.audio_processing import get_audio_input, speak_text
from src.document_handler import handle_document_upload, handle_arxiv_upload, chatbot_response, clear_conversation_history

import os
import tempfile
import sqlite3
import torch
from PIL import Image
import speech_recognition as sr
import pyttsx3
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader, ArxivLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from httpx import ConnectError
import chromadb




# Initialize database connection
conn = sqlite3.connect('chatbot_memory.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY,
    role TEXT,
    message TEXT
)
''')

# Function to clear the conversation history
def clear_conversation_history():
    cursor.execute('DELETE FROM memory')
    conn.commit()
    st.write("Conversation history cleared!")

# Load the CLIP model for image captioning
device = "cpu"  # Use CPU instead of GPU

# Load the BLIP model for dynamic image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Initialize Ollama LLM
llm = OllamaLLM(model="llama2")

# Use embedding model with matching dimensionality (768)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Recreate the Chroma vector store to avoid dimension mismatch
client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("default")  # Delete existing collection to avoid dimension mismatch
except ValueError:
    pass  # Skip if collection does not exist
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

def chatbot_response(user_input):
    cursor.execute('INSERT INTO memory (role, message) VALUES (?, ?)', ('user', user_input))
    conn.commit()
    
    try:
        response = qa_chain.run(user_input)
    except ConnectError:
        st.write("Error: Cannot connect to Ollama. Please ensure the server is running.")
        response = "Sorry, I'm currently unable to process requests."
    
    cursor.execute('INSERT INTO memory (role, message) VALUES (?, ?)', ('chatbot', response))
    conn.commit()
    
    return response
# Streamlit UI
st.title("Multimodal LLM Chatbot with Image Captioning, Memory, and Audio I/O")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Chatbot", "Image Captioning", "Audio I/O", "Document Upload", "Conversation History"])

if section == "Chatbot":
    st.header("Chatbot Interaction")
    user_input = st.text_input("You: ", placeholder="Type your message here...")
    if user_input:
        response = chatbot_response(user_input)
        st.write(f"Chatbot: {response}")
        speak_text(response)

elif section == "Image Captioning":
    st.header("Image Captioning")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        caption = generate_caption(uploaded_image)
        st.write(f"Generated Caption: {caption}")

elif section == "Audio I/O":
    st.header("Audio Input/Output")
    if st.button("Speak"):
        audio_input = get_audio_input()
        if audio_input:
            response = chatbot_response(audio_input)
            st.write(f"Chatbot: {response}")
            speak_text(response)

elif section == "Document Upload":
    st.header("Document Upload and Processing")
    uploaded_file = st.file_uploader("Upload a PDF, CSV, or arXiv file", type=["pdf", "csv", "arxiv"])
    arxiv_id = st.text_input("Or enter the arXiv ID of the paper:")
    operation = st.radio("Choose an operation:", ["Question Answering", "Summarization"])
    
    if st.button("Process Upload"):
        handle_document_upload(uploaded_file, operation)
    
    if st.button("Process arXiv ID"):
        handle_arxiv_upload(arxiv_id, operation)

    if 'operation' in st.session_state and st.session_state['operation'] == "Question Answering":
        question = st.text_input("Ask a question about the uploaded file:")
        if question:
            try:
                response = chatbot_response(question)
                st.write(f"Answer: {response}")
            except Exception as e:
                st.write(f"Error: {e}")
            # Display conversation history
            st.write("Conversation History:")
            cursor.execute('SELECT role, message FROM memory')
            for row in cursor.fetchall():
                st.write(f"{row[0]}: {row[1]}")

elif section == "Conversation History":
    st.header("Conversation History")
    cursor.execute('SELECT role, message FROM memory')
    for row in cursor.fetchall():
        st.write(f"{row[0]}: {row[1]}")
    
    if st.button("Clear Conversation History"):
        clear_conversation_history()
