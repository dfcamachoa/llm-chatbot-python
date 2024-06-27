import streamlit as st
import os
import openai
from neo4j import GraphDatabase, Result
import pandas as pd
from openai import OpenAI, APIError
from time import sleep
from langchain_community.llms import Ollama

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def generate_embeddings(file_name, limit=None):
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    driver = GraphDatabase.driver(
        st.secrets["NEO4J_URI"],
        auth=(st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])
    )

    driver.verify_connectivity()

    query = """MATCH (l:Lesson) WHERE l.what IS NOT NULL
    RETURN l.id AS lessonId, l.title AS title, l.what AS what"""

    if limit is not None:
        query += f" LIMIT {limit}"

    lessons = driver.execute_query(
        query,
        result_transformer_=Result.to_df
    )

    print(len(lessons))
    
    embeddings = []

    for _, n in lessons.iterrows():
        
        successful_call = False
        while not successful_call:
            try:
                # Combine 'title' and 'what' for the input text
                input_text = f"{n['title']}: {n['what']}"
                # Get the embedding
                embedding = get_embedding(input_text)
                successful_call = True
            except APIError as e:
                print(e)
                print("Retrying in 5 seconds...")
                sleep(5)

        print(n['title'])

        embeddings.append({
            "lessonId": n['lessonId'],
            "embedding": embedding
        })

    embedding_df = pd.DataFrame(embeddings)
    embedding_df.head()
    embedding_df.to_csv(file_name, index=False)

generate_embeddings('data\openai-embeddings.csv',limit=1000)
generate_embeddings('data\openai-embeddings-full.csv',limit=None)