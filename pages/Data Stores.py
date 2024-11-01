import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings


# Load environment variables
load_dotenv()

# Set the title of the Streamlit app
st.set_page_config(layout="wide", page_title="Creating a Data Store")
st.title("Creating a Data Store")

# Define the function to handle document embeddings
def vector_embedding(uploaded_files):

    st.session_state.embeddings = OllamaEmbeddings(model='all-minilm')

    docs = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to disk
        with open(os.path.join("uploaded_docs", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Load PDF document
        loader = PyPDFLoader(os.path.join("uploaded_docs", uploaded_file.name))
        docs.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    
    # Create vector embeddings
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
    st.session_state.vectors.save_local('Database')
    
    st.session_state.final_documents = final_documents


# Allow users to upload PDF documents
uploaded_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=["pdf"])

# Button to create document embeddings
if st.button("Create Document Embeddings") and uploaded_files:
    os.makedirs("uploaded_docs", exist_ok=True)
    vector_embedding(uploaded_files)
    st.write("Vector Store DB is ready")