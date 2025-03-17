import os
import streamlit as st
from dotenv import load_dotenv
import io

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

load_dotenv()

# llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
llm = ChatGroq(model="llama3-70b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

template = """
    You are an AI that mimics the personality and communication style of a specific user. 
    Use the following context to answer the user's query. Maintain the tone and style that is within the context.
    
    Context:
    {context}

    Chat History:
    {chat_history}

    User Query: {query}

    Answer:
"""

PERSIST_DIRECTORY = "./chroma_langchain_db"

def init_session_state():
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def load_data(files):
    all_documents = []

    if not files:
        return all_documents

    for uploaded_file in files:
        try:
            text = uploaded_file.getvalue().decode("utf-8")
            document = Document(page_content=text, metadata={"source": uploaded_file.name})
            all_documents.append(document)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    return all_documents

    # if files:
    #     for uploaded_file in files:
    #         try:
    #             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    #                 temp_file.write(uploaded_file.getvalue())
    #                 temp_file_path = temp_file.name

    #             loader = PyPDFLoader(temp_file_path)
    #             documents = loader.load()
    #             all_documents.extend(documents)

    #             os.unlink(temp_file_path)

    #         except Exception as e:
    #             st.error(f"Error processing {uploaded_file.name}: {e}")
    # return all_documents

    



def chunk_data(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    texts = [Document(page_content=chunk.page_content) for chunk in chunks]
    return texts

def vectordb(texts):

    text_list = [doc.page_content for doc in texts]
    vectordb = FAISS.from_texts(text_list, embeddings)  
    return vectordb.as_retriever()  


def get_llm_response(retriever, query):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in st.session_state.chat_history])

    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(context=context, chat_history=chat_history_str, query=query)
    response = llm.invoke(messages).content
    return response

def main():
    st.title("Personality Modeling with RAG")
    init_session_state()

    for message in st.session_state.chat_history:
        with st.chat_message("human" if isinstance(message, HumanMessage) else "ai"):
            st.markdown(message.content)

    with st.sidebar:
        st.header("Upload Data Here")
        uploaded_files = st.file_uploader(
            "Upload documents", 
            type=["pdf", "txt", "doc", "docx"],
            accept_multiple_files=True
        )
        button = st.button("Process")

        if button:
            with st.spinner("Processing..."):
                loaded_docs = load_data(uploaded_files)
                chunks = chunk_data(loaded_docs)
                st.session_state.retriever = vectordb(chunks)
                st.success("Vector DB Updated!")

    query = st.chat_input("Enter your query:")
    
    if query:
        with st.chat_message("human"):  
            st.markdown(query)

        st.session_state.chat_history.append(HumanMessage(content=query))

        if st.session_state.retriever:
            response = get_llm_response(st.session_state.retriever, query)

            with st.chat_message("ai"):  
                st.markdown(response)

            st.session_state.chat_history.append(AIMessage(content=response))

if __name__=="__main__":
    main()
