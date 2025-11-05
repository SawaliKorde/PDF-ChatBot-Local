import streamlit as st
import weaviate
import time
from typing import List

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Weaviate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- CONSTANTS  ---
WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_INDEX_NAME = "LangChain" # Default Weaviate collection name for LangChain
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:7b"

print("--- Imports and constants loaded ---")

# --- Client Connection ---
try:
    client = weaviate.Client(url=WEAVIATE_URL)
    print("--- Connected to Weaviate ---")
except Exception as e:
    print(f"--- ERROR: Could not connect to Weaviate. Is it running? ---")
    print(e)
    client = None

@st.cache_resource(show_spinner="Processing PDF...")
def setup_rag_pipeline(uploaded_file):
    """
    Sets up the RAG pipeline from an uploaded PDF file.
    (Based on Section 3.1's setup_rag_pipeline)
    """
    
    
   
    try:
        client = weaviate.Client(url=WEAVIATE_URL)
        print("--- Connected to Weaviate ---")
    except Exception as e:
        st.error("Could not connect to Weaviate. Is your Docker container running?")
        st.exception(e)
        return None  # Stop execution if we can't connect
    # --- END NEW CODE ---

   
    # PyPDFLoader needs a file path
    with open(f"./temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    file_path = f"./temp_{uploaded_file.name}"

    # 1. Load the document
    st.write("Step 1: Loading PDF document...")
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    if not docs:
        st.error("Could not load any content from the PDF.")
        return None
    st.write(f"Document loaded. Number of pages: {len(docs)}")

    # 2. Split the document into chunks
    st.write("Step 2: Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] #
    )
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        st.error("Could not split the document into chunks.")
        return None
    st.write(f"Document split into {len(chunks)} chunks.")

    # 3. Initialize embedding model
    st.write(f"Step 3: Initializing embedding model ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 4. Index chunks in Weaviate vector store
    st.write("Step 4: Indexing chunks in Weaviate vector store...")
    try:
        # Clear old data for this demo
        client.schema.delete_class(WEAVIATE_INDEX_NAME)
        print("--- Cleared old Weaviate schema ---")
    except Exception as e:
        print("--- Schema not found, creating new one ---")

    vectorstore = Weaviate.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,  # This client is now guaranteed to be valid
        index_name=WEAVIATE_INDEX_NAME,
        by_text=False #
    )
    
    st.success("PDF processing complete! Ready to answer questions.")
    return vectorstore


def format_docs(docs: List[Document]) -> str:
    """
    Formats a list of Document objects into a single string.
    (From Section 3.1)
    """
    return "\n\n".join(doc.page_content for doc in docs) # [cite: 130]

# --- STREAMLIT UI ---
print("--- Starting Streamlit App ---")

st.title("📄 Local-First PDF Chatbot")
st.markdown("Using Weaviate, Ollama (Mistral 7B), and LangChain")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
st.info("Your document is processed locally and never leaves your machine.")

if uploaded_file:
    # --- Setup RAG Pipeline ---
    # We use session_state to store the vectorstore AND the name of the file
    # to detect when a new file is uploaded.

    if "vectorstore" not in st.session_state or st.session_state.get("processed_file_name") != uploaded_file.name:
        # This is a new file, or the first file. Process it.
        vectorstore = setup_rag_pipeline(uploaded_file)
        
        # Store the vectorstore and the name of the processed file in the session
        st.session_state.vectorstore = vectorstore
        st.session_state.processed_file_name = uploaded_file.name
        
        # Clear old chat messages
        st.session_state.messages = []
    else:
        # It's the same file, just grab the vectorstore from session state
        vectorstore = st.session_state.vectorstore

    # --- RAG Q&A Chain ---
    if vectorstore:
        # Initialize the chat model (from Section 3.1)
        llm = ChatOllama(model=LLM_MODEL, temperature=0) # [cite: 134]

        # Define the retriever (from Section 3.1)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # [cite: 132-133]

        # Define the prompt template (from Section 3.1)
        template = """
        You are an expert assistant for question-answering tasks.
        Use the following retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Keep the answer concise and based only on the provided context.
        Context:
        {context}
        Question:
        {questio[n}
        Answer:
        """ # [cite: 133]
        prompt = ChatPromptTemplate.from_template(template) # [cite: 134]

        # Construct the RAG Chain (from Section 3.1)
        rag_chain = (
            RunnableParallel(
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
            )
            | prompt
            | llm
            | StrOutputParser()
        ) # [cite: 134]

        # --- Chat Interface ---
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_query := st.chat_input("Ask a question about your document"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Get model response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Invoke the chain (from Section 3.1)
                    response = rag_chain.invoke(user_query) # [cite: 135]
                    st.markdown(response)
                    # Add model response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})

else:

    st.warning("Please upload a PDF document to begin.")
