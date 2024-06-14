import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from dotenv import load_dotenv

# load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = "sk-proj-N4Eym32hD3HkVltVNWi8T3BlbkFJiq4n7XcFLK24rp6FgBgZ"

# handle streaming conversation
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Function to extract text from an PDF file
from pdfminer.high_level import extract_text

def get_pdf_text(filename):
    raw_text = extract_text(filename)
    return raw_text

# document preprocess
def process_uploaded_file(uploaded_file):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # loader
        raw_text = get_pdf_text(uploaded_file)

        # splitter
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,
        )
        all_splits = text_splitter.create_documents([raw_text])
        print("총 " + str(len(all_splits)) + "개의 passage")
        
        # storage
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

        return vectorstore, raw_text
    return None

# generate response using RAG technic
def generate_response(query_text, vectorstore, callback):

    # retriever
    docs_list = vectorstore.similarity_search(query_text, k=3)
    docs = ""
    for i, doc in enumerate(docs_list):
        docs += f"'문서{i+1}':{doc.page_content}\n"
    print(docs)
        
    # generator
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True, callbacks=[callback])
    
    # chaining
    rag_prompt = [
        SystemMessage(
            content="You are a judge specializing in traffic laws, modeled after 한문철. Based on your knowledge and the provided references, you will make judgments for various situations. Respond in Korean, offering clear and precise rulings according to the traffic laws and legal precedents. Use a tone and style similar to 한문철."
        ),
        HumanMessage(
            content=f"질문:{query_text}\n\n{docs}"
        ),
    ]

    response = llm(rag_prompt)
    
    return response.content


def generate_summarize(raw_text, callback):
    # generator 
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True, callbacks=[callback])
    
    # prompt formatting
    rag_prompt = [
        SystemMessage(
            content="다음 나올 문서를 'Notion style'로 요약해줘. 중요한 내용만."
        ),
        HumanMessage(
            content=raw_text
        ),
    ]
    
    response = llm(rag_prompt)
    return response.content


# page title
st.set_page_config(page_title='교통법률 전문가 AI문철')
st.title('교통법률 전문가 AI문철')

# api key input button 
# import os
# api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
# save_button = st.sidebar.button("Save Key")
# if save_button and len(api_key)>10:
#     os.environ["OPENAI_API_KEY"] = api_key
#     st.sidebar.success("API Key saved successfully!")

# file upload
uploaded_file = st.file_uploader('Upload an document', type=['pdf'])

# file upload logic
if uploaded_file:
    vectorstore, raw_text = process_uploaded_file(uploaded_file)
    if vectorstore:
        st.session_state['vectorstore'] = vectorstore
        st.session_state['raw_text'] = raw_text
        
# chatbot greatings
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant", content="교통사고 발생시 당황하지 말고 저를 찾아보세요."
        )
    ]

# conversation history print 
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)
    
# message interaction
if prompt := st.chat_input("어떤 난처한 상황에 처하셨나요?"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        
        if prompt == "어떤 난처한 상황에 처하셨나요?":
            response = generate_summarize(st.session_state['raw_text'],stream_handler)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
        
        else:
            response = generate_response(prompt, st.session_state['vectorstore'], stream_handler)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
