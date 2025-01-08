import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader, ArxivLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from sqlite3 import connect
import streamlit as st

# Database and vector store setup
conn = connect('chatbot_memory.db')
cursor = conn.cursor()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
retriever = vector_store.as_retriever()
llm = OllamaLLM(model="llama2")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Document upload and handling
def handle_document_upload(uploaded_file, operation):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    if uploaded_file.name.endswith('.pdf'):
        loader = PyMuPDFLoader(tmp_path)
    elif uploaded_file.name.endswith('.csv'):
        loader = CSVLoader(tmp_path)
    elif uploaded_file.name.endswith('.arxiv'):
        loader = ArxivLoader(tmp_path)
    
    documents = loader.load()
    vector_store.add_documents(documents)
    
    os.remove(tmp_path)
    
    if operation == "Question Answering":
        st.write("Document uploaded and indexed. Ready for question answering.")
        st.session_state['documents'] = documents
        st.session_state['operation'] = operation
    elif operation == "Summarization":
        # Assuming a summarization function summarize_document exists
        summaries = [summarize_document(doc) for doc in documents]
        st.write("Summarization completed:\n" + "\n".join(summaries))
    else:
        st.write("Invalid operation.")

def handle_arxiv_upload(arxiv_id, operation):
    loader = ArxivLoader(arxiv_id)
    documents = loader.load()
    vector_store.add_documents(documents)
    
    if operation == "Question Answering":
        st.write("arXiv paper uploaded and indexed. Ready for question answering.")
        st.session_state['documents'] = documents
        st.session_state['operation'] = operation
    elif operation == "Summarization":
        summaries = [summarize_document(doc) for doc in documents]
        st.write("Summarization completed:\n" + "\n".join(summaries))
    else:
        st.write("Invalid operation.")

def summarize_document(document):
    # Placeholder for the summarization logic
    return f"Summary of document: {document}"


# Chatbot interaction
def chatbot_response(user_input):
    cursor.execute('INSERT INTO memory (role, message) VALUES (?, ?)', ('user', user_input))
    conn.commit()
    response = qa_chain.run(user_input)
    cursor.execute('INSERT INTO memory (role, message) VALUES (?, ?)', ('chatbot', response))
    conn.commit()
    return response

# Clear conversation history
def clear_conversation_history():
    cursor.execute('DELETE FROM memory')
    conn.commit()
    st.write("Conversation history cleared!")