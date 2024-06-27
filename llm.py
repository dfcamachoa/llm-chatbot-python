import streamlit as st
# from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# llm = ChatOpenAI(
#     openai_api_key=st.secrets["OPENAI_API_KEY"],
#     model=st.secrets["OPENAI_MODEL"]
# )

llm = ChatOllama(
    model = "llama3"
)

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

# llm = Ollama(
#     model="mistral"
# )

# embeddings = OllamaEmbeddings(
#     model="mistral"
# )