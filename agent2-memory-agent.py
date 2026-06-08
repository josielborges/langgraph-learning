from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-5.4-nano", reasoning_effort="low")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""

    messages = state.get("messages", [])
    response = llm.invoke(messages)
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print(f"\nCURRENT STATE: {state["messages"]}")
        
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history=[]

user_input = input("Enter: ")
while (user_input != "exit"):
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]

    user_input = input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("Your conversation Log:\n")

    for messages in conversation_history:
        if isinstance(messages, HumanMessage):
            file.write(f"User: {messages.content}\n")
        elif isinstance(messages, AIMessage):
            file.write(f"AI: {messages.content}\n")
    file.write("End of conversation.\n")

print("Conversation saved into logging.txt")