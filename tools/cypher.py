from langchain.chains import GraphCypherQAChain

from llm import llm
from graph import graph

cypher_qa = GraphCypherQAChain.from_llm(
    llm,          # (1)
    graph=graph,  # (2)
    verbose=True
)