import streamlit as st


import os
import time


#userprompt
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


#vectorDB
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings


#llms
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

#retrieval
from langchain.chains import RetrievalQA

###### chat_history, st sessions are additional features with llama3 from localhost ############

if 'template' not in st.session_state:
   #st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    st.session_state.template = """
                                As a knowledgeable chatbot designed to assist users, your role is crucial in providing informative responses. Your task involves referencing the relevant sections from the 3GPP TS 29.XXX series documentation or, if unavailable, from related documents within the 3GPP series. When faced with inquiries, your approach should exhibit professionalism, Groundedness & efficiency.
                                1.Introduce yourself as the knowledgeable chatbot ' ' 'CGIU 5G chatbot' ' ' designed for user assistance.
                                2.Emphasize the preference for sourcing information from the 3GPP TS 29.XXX series, or alternately, from related documents within the 3GPP series if the specific section is not available.
                                3.Maintain a professional and efficient tone throughout interactions.
                                4.Break down the process into clear steps when required and include suitable delimiters for clarity.

   Context: {context}
   History: {history}


   User: {question}
   Chatbot:"""


if 'prompt' not in st.session_state:
   st.session_state.prompt = PromptTemplate(
       input_variables=["history", "context", "question"],
       template=st.session_state.template,
   )


if 'memory' not in st.session_state:
   st.session_state.memory = ConversationBufferMemory(
       memory_key="history",
       return_messages=True,
       input_key="question",
   )


if 'vectorstore' not in st.session_state:
   st.session_state.vectorstore = Chroma(persist_directory='chromadb',
                                           embedding_function=GPT4AllEmbeddings()
                                           )
  
if 'llm' not in st.session_state:
   st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                 model="llama3",
                                 verbose=True,
                                 callback_manager=CallbackManager(
                                     [StreamingStdOutCallbackHandler()]),
                                 )
  
if 'chat_history' not in st.session_state:
   st.session_state.chat_history = []


st.title("Oracle CGIU 5G Chatbot - llama3")


# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")


for message in st.session_state.chat_history:
   with st.chat_message(message["role"]):
       st.markdown(message["message"])

st.session_state.retriever = st.session_state.vectorstore.as_retriever()


if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type='stuff',
        retriever=st.session_state.retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": st.session_state.prompt,
            "memory": st.session_state.memory,
        }
    )


if user_input := st.chat_input("You:", key="user_input"):
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)


    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain(user_input)
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response['result'].split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)


    chatbot_message = {"role": "assistant", "message": response['result']}
    st.session_state.chat_history.append(chatbot_message)


# else:
#    st.write("Please upload a PDF file to start the chatbot")
