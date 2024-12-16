import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os

st.set_page_config(page_title="InsightBot: Conversational AI for Multi-PDF Insights")

# Download and load spaCy model
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure API key securely
os.environ['GOOGLE_API_KEY'] = "AIzaSyCTCDhiHrWjBjROAELCSiynayzvv023xW8"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def advanced_chunking(text):
    try:
        max_length = 50000
        text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        
        processed_chunks = []
        for chunk in text_chunks:
            doc = nlp(chunk)
            sentences = [sent.text for sent in doc.sents]
            ner_entities = [ent.text for ent in doc.ents]
            sentence_embeddings = sentence_model.encode(sentences)
            
            current_chunk = []
            current_chunk_embedding = None
            for i, sentence in enumerate(sentences):
                if current_chunk_embedding is None:
                    current_chunk.append(sentence)
                    current_chunk_embedding = sentence_embeddings[i]
                else:
                    similarity_score = cosine_similarity([current_chunk_embedding], [sentence_embeddings[i]])[0][0]
                    if similarity_score > 0.8:
                        current_chunk.append(sentence)
                        current_chunk_embedding = (current_chunk_embedding + sentence_embeddings[i]) / 2
                    else:
                        processed_chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_chunk_embedding = sentence_embeddings[i]

            if current_chunk:
                processed_chunks.append(" ".join(current_chunk))

        final_chunks = [chunk for chunk in processed_chunks if any(entity in chunk for entity in ner_entities)]
        return final_chunks
    except Exception as e:
        st.error("Error during text chunking: " + str(e))
        return []

# Function to generate vector embeddings and store them in FAISS
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to create the conversational chain
def get_conversational_chain(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Main function for the app
def main():
    st.header("InsightBoot: Chat with Multiple PDFs 💬")

    # Sidebar for uploading PDF documents
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = advanced_chunking(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Processing Complete!")

    # Chat functionality
    user_question = st.text_input("Ask a question regarding the PDF")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if user_question:
        if st.session_state.conversation:
            try:
                response = st.session_state.conversation({'question': user_question})
                st.write("Alina: ", response['answer'])
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload and process a document first.")

if __name__ == "__main__":
    main()
