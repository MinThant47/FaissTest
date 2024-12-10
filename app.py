import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

st.set_page_config(page_title="Swel Pay Lar Q&A", layout="centered")

## load the GROQ And OpenAI API KEY 
groq_api_key="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"
os.environ["GOOGLE_API_KEY"]= "AIzaSyC87rM9xeEqJ6Rt5LhguLed6QK5mzT6XBM"

conversational_memory_length = 5
memory=ConversationBufferWindowMemory(k=conversational_memory_length)

if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
else:
    for message in st.session_state.chat_history:
        memory.save_context({'input':message['human']},{'output':message['AI']})

st.title("Swel Pay Lar Q&A")
st.write("A chatbot for that provides everything you need to communicate without waiting for admin assistance.")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector google embeddings

    #     index_path = "faiss_index"
    # st.session_state.vectors.save_local("faiss_index")
        
prompt1=st.chat_input("Enter Your Question...")

vector_embedding()

# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")
if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            st.chat_message('user').markdown(msg['human'])
            st.chat_message('assistant').markdown(msg['AI'])
            #  st.write("🧑‍💻 **You:**", msg['human'])
            #  st.write("🤖 **Chatbot:**", msg['AI'])
            #  st.write("------------------------------")
# for message in st.session_state.chat_history:
#     st.chat_message(message['role']).markdown(message['content'])

if prompt1:
    st.chat_message('user').markdown(prompt1)
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.chat_message('ai').markdown(response['answer'])


    if response:
        # Save the interaction to session history
        message = {'human': prompt1, 'AI': response['answer']}
        st.session_state.chat_history.append(message)

    # With a streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant chunks
    #     for i, doc in enumerate(response["context"]):
    #         st.write(doc.page_content)
    #         st.write("--------------------------------")




