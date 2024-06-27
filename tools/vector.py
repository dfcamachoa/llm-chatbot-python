import streamlit as st
from llm import llm, embeddings
from graph import graph

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.chains import RetrievalQA

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="lessonInfo",                 # (5)
    node_label="Lesson",                     # (6)
    text_node_property="what",               # (7)
    embedding_node_property="embedding",     # (8)
    retrieval_query="""
RETURN
    node.what AS text,
    score,
    {
        title: node.title,
        problem: [ (node)-[:IS_CAUSED_BY]->(cause)-[:HAS_ROOT]->(problem) | problem.description ],
        reccomendation: [ (node)-[:GENERATES]->(reccomendation) | reccomendation.description ],
        action: [ (node)-[:RESULTS_IN]->(action) | action.description ],
        engineering_phase: [ (node)<-[:ENACTS]-(feedback) | feedback.detected ]
    } AS metadata
"""
)
                         
retriever = neo4jvector.as_retriever()

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
    )

prompt = ChatPromptTemplate.from 

kg_qa = RetrievalQA.from_chain_type(
    llm,                  # (1)
    chain_type="stuff",   # (2)
    retriever=retriever,  # (3)
)