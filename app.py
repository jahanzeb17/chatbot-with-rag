import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", streaming=False) #Disable Streaming
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

template = """
You are an AI that emulates the conversational style and personality of the user whose conversation data is provided in the context.
{persona}
Analyze the context to understand their typical word choices, sentence structure, tone, and any recurring phrases.
Pay close attention to the chat history to maintain consistency and coherence.
Respond to the user's query as if you were that person, using their unique style and voice.
Try to keep the response within the length of the users usual responses.

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
    if 'persona' not in st.session_state:
        st.session_state.persona = ""
    if 'persona_history' not in st.session_state:
        st.session_state.persona_history = []

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

def chunk_data(documents):
    semantic_splitter = SemanticChunker(embeddings)
    chunks = semantic_splitter.split_documents(documents)
    return chunks

def vectordb(texts):
    faiss_retriever = FAISS.from_documents(texts, embeddings).as_retriever()
    bm25_retriever = BM25Retriever.from_documents(texts)
    tfidf_retriever = TFIDFRetriever.from_documents(texts)
    ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever, tfidf_retriever], weights=[0.6, 0.2, 0.2])
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
    return compression_retriever

def extract_persona(documents):
    all_text = " ".join([doc.page_content for doc in documents])
    persona_prompt = f"""
    Analyze the following text and extract the key characteristics of the speaker's personality and conversational style.

    Identify and describe the following aspects in detail:

    - Typical word choices and vocabulary.
    - Tone and formality (e.g., formal, informal, humorous, sarcastic).
    - Sentence structure and length.
    - Common phrases and recurring patterns.
    - Emotional range and expression.
    - Level of empathy and social awareness.
    - Any unique quirks or idiosyncrasies.

    Text:
    {all_text}

    Persona:
    """
    persona_messages = [HumanMessage(content=persona_prompt)]
    persona = llm.invoke(persona_messages).content
    return persona

def get_llm_response(retriever, query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    chat_history_str = "\n".join([f"{message.type}: {message.content}" for message in st.session_state.chat_history])

    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(context=context, chat_history=chat_history_str, query=query, persona=st.session_state.persona)
    response = llm.invoke(messages).content

    # Post processing
    persona_check_prompt_template = PromptTemplate.from_template("""
    Does the following response match the persona?
    Persona: {persona}
    Response: {response}
    Answer yes or no.
    """)
    persona_check_chain = LLMChain(llm=llm, prompt=persona_check_prompt_template)
    persona_check = persona_check_chain.run(persona=st.session_state.persona, response=response)

    if "no" in persona_check.lower():
        style_adjust_prompt_template = PromptTemplate.from_template("""
        Adjust the following response to better match the persona.
        Persona: {persona}
        Response: {response}
        Adjusted Response:
        """)
        style_adjust_chain = LLMChain(llm=llm, prompt=style_adjust_prompt_template)
        response = style_adjust_chain.run(persona=st.session_state.persona, response=response)

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
                st.session_state.persona = extract_persona(loaded_docs)
                st.session_state.persona_history.append(st.session_state.persona)
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

if __name__ == "__main__":
    main()
