# First
import os
import openai 
import streamlit as st

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = "afbd12be650748099d22ab1791ed4193"

os.environ["BASE_URL"] = "https://evaluateopenai.openai.azure.com/"

os.environ["CHAT_DEPLOYMENT_NAME"] = "gpt-35-turbo_0301"
os.environ["EMB_DEPLOYMENT_NAME"] = "text-embedding-ada-002"

API_KEY = "afbd12be650748099d22ab1791ed4193"

BASE_URL = "https://evaluateopenai.openai.azure.com/"

CHAT_DEPLOYMENT_NAME = "gpt-35-turbo_0301"
EMB_DEPLOYMENT_NAME = "text-embedding-ada-002"

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI

import openai
openai.api_type = "azure"
openai.api_version = "2023-07-01-preview"
openai.api_base = BASE_URL  # Your Azure OpenAI resource's endpoint value.
openai.api_key = API_KEY

from langchain.embeddings import OpenAIEmbeddings

# import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",deployment="text-embedding-ada-002")

# load from disk
#db = Chroma(persist_directory="/Users/hiroyuki/Documents/python/OpenAI/chroma_db", embedding_function=OpenAIEmbeddings())
db = Chroma(persist_directory="/Users/hiroyuki/Library/CloudStorage/OneDrive-個人用/mypython/OpenAI/chroma_db", embedding_function=OpenAIEmbeddings())
_ = """

text_splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
    length_function = len,
)

# Input file path
input_file = "/Users/hiroyuki/Library/CloudStorage/OneDrive-個人用/mypython/OpenAI/long_text.txt"

# Read the input text from the input file and process each line
with open(input_file, "r", encoding="utf-8") as f_in:
    for line in f_in:
        input_text = line.strip()  # 行末の改行文字を削除
        #print(input_text)
        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        #text_splitter = CharacterTextSplitter(separator = "$",chunk_size=300, chunk_overlap=0)
        documents = text_splitter.split_text(input_text)
        #print(documents)
        #db = Chroma.from_documents(documents, OpenAIEmbeddings())
        # save to disk
        db = Chroma.from_texts(documents, OpenAIEmbeddings(), persist_directory="/Users/hiroyuki/Library/CloudStorage/OneDrive-個人用/mypython/OpenAI/chroma_db")
        #db = Chroma.from_documents(loader.load(), OpenAIEmbeddings())


"""


st.title("💬 PM-Ai Assistant Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    query=st.session_state.messages[-1]['content']
    print(query)
    # load from disk
    docs = db.similarity_search_with_score(query,k=5)
    print(docs[0])
    docs_5=""
    for i in range(5):
        print(i)
        docs_5 += str(i) + docs[i][0].page_content+"\n"
    query="類似事例>>>"+docs_5+">>>以上を踏まえ>>>以下の質問に回答してください。もし、類似事例を求めている場合は、文書ID:,工場名,改善内容要約というフォーマットで類似事例を示してください。>>>"+query

    messages=[]
    messages.append({"role": "user", "content": query})
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo_0301",
        messages=messages,
    )

    #response = response['choices'][0]['message']['content']

    #response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)

    print(msg)