import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
# from langchain import hub
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from llm import llm
from langchain.prompts import PromptTemplate
from tools.vector import kg_qa
from tools.cypher import cypher_qa
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
    )

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=False
    ),
    Tool.from_function(
        name="Vector Search Index",  # (1)
        description="Provides information about lessons learned during projects in Technip Energies using Vector Search", # (2)
        func = kg_qa, # (3)
        return_direct=False
    ),
    # Tool.from_function(
    #     name="Graph Cypher QA Chain",  # (1)
    #     description="Provides information about lessons including their causes, reccomendations and root Problems", # (2)
    #     func = cypher_qa, # (3)
    #     return_direct=False
    # ),
]        

# agent_prompt = hub.pull("hwchase17/react-chat")
agent_prompt = PromptTemplate.from_template("""
You are a Quality Assurance expert providing information about quality and lessons learned in Technip Energies, an EPC engineering company.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to lessons learned, causes, projects, users, impacts.
Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

def generate_response(prompt):
    st_callback = StreamlitCallbackHandler(st.container()) 
    response = agent_executor.invoke({"input":prompt}, {"callbacks": [st_callback]})
    
    return response['output']