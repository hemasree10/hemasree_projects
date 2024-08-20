
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
#import pinecone
#from langchain.vectorstores import Pinecone as PC
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]="ls__ecbe0d66035a4c2ca1f21898500cde4d"
os.environ["LANGCHAIN_PROJECT"]="Sparklewords chatbot"
def main():
    st.set_page_config(page_title="Chat With files")
    st.header("Chatbot developed by [@Sparkle Words](https://github.com/sravani586/SparkeWords)")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_files_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key)

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Chat with your file")
        if user_question:
            handel_userinput(user_question)





def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    #pinecone.init(api_key="ce0aca59-f50d-4cf2-ae70-03bb069c3c51",environment="gcp-starter")
    #index_name="educationalvector"
    #index=PC.from_texts(text_chunks, embeddings, index_name=index_name)

    return knowledge_base

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    template = """you are a domain specific chatbot which gives answers only to the input document provided, if any query asked out of context just say i am supposed to answer only questions related to this document.
                     {context}
                  Question: {question}
                   Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory,
        #condense_question_prompt=QA_CHAIN_PROMPT
        combine_docs_chain_kwargs={'prompt':QA_CHAIN_PROMPT}
    )
    return conversation_chain


def handel_userinput(user_question):

    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
        st.write(f"Total Tokens: {cb.total_tokens}" f", Prompt Tokens: {cb.prompt_tokens}" f", Completion Tokens: {cb.completion_tokens}" f", Total Cost (USD): ${cb.total_cost}")




if __name__ == '__main__':
    main()
