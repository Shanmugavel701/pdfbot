# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import tempfile
import os

# Set your Google API Key
GOOGLE_API_KEY = "AIzaSyD-q5-mcoLn6Horgx-tPD_q4V5N_GV7uQE"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

st.set_page_config(page_title="RAG PDF QA App", layout="centered")
st.title("📄 RAG-powered PDF Q&A App with Gemini 2.0")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    st.success("PDF Uploaded Successfully!")

    loader = PyPDFLoader(tmp_pdf_path)
    pages = loader.load()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("Answer the question based only on the following context:\n\n{context}"),
        HumanMessagePromptTemplate.from_template("Question: {question}")
    ])

    chain = (
        {"context": retriever | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])),
         "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    user_question = st.text_input("Ask a question about your PDF")
    if user_question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(user_question)
            st.markdown("### ✨ Answer")
            st.write(answer.content)

# Cleanup
if 'tmp_pdf_path' in locals():
    os.remove(tmp_pdf_path)
