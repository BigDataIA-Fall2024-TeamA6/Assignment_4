import os
import time
from semantic_router.encoders import OpenAIEncoder
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from tools import run_oracle,run_tool,tools
from state import AgentState

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY") 

encoder = OpenAIEncoder(name="text-embedding-ada-002")
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(
    cloud="aws", region="us-east-1")


pdf_index_name = "pdf-364ff30954"

def get_index(index_name):
    if index_name not in pc.list_indexes().names():
    # if does not exist, create index
        new_index = pc.create_index(
            index_name,
            dimension=768,  # dimensionality of embed 3
            metric='dotproduct',
            spec=spec  )
    # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        return new_index
    else:
        index = pc.Index(index_name)
        print("Index fetch success")
        return index

index = get_index(pdf_index_name)


def router(state: list):
    # return the tool name to use
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
        # if we output bad format go to final answer
        print("Router invalid format")
        return "final_answer"


from langgraph.graph import StateGraph, END

def get_graph():
    graph = StateGraph(AgentState)

    graph.add_node("oracle", run_oracle)
    graph.add_node("rag_search_filter", run_tool)
    graph.add_node("rag_search", run_tool)
    graph.add_node("fetch_arxiv", run_tool)
    graph.add_node("web_search", run_tool)
    graph.add_node("final_answer", run_tool)

    graph.set_entry_point("oracle")

    graph.add_conditional_edges(
        source="oracle",  # where in graph to start
        path=router,  # function to determine which node is called
    )

    # create edges from each tool back to the oracle
    for tool_obj in tools:
        if tool_obj.name != "final_answer":
            graph.add_edge(tool_obj.name, "oracle")

    # if anything goes to final answer, it must then move to END
    graph.add_edge("final_answer", END)

    ai_graph = graph.compile()
    
    return ai_graph


runnable = get_graph()

"""## Building Reports

Let's test our research agent. First, I want to try on something simple (although not within the intended use-case of our agent):
"""

# out = runnable.invoke({
#     "input": "tell me something interesting about dogs",
#     "chat_history": [],
# })

"""Let's create a function to consume the agent output and format it into our report:"""


# def build_report(output: dict):
#     research_steps = output["research_steps"]
#     if type(research_steps) is list:
#         research_steps = "\n".join([f"- {r}" for r in research_steps])
#     sources = output["sources"]
#     if type(sources) is list:
#         sources = "\n".join([f"- {s}" for s in sources])
#     return f"""
# INTRODUCTION
# ------------
# {output["introduction"]}

# RESEARCH STEPS
# --------------
# {research_steps}

# REPORT
# ------
# {output["main_body"]}

# CONCLUSION
# ----------
# {output["conclusion"]}

# SOURCES
# -------
# {sources}
# """

# print(build_report(
#     output=out["intermediate_steps"][-1].tool_input
# ))

# """Now let's try with an on-topic question on AI."""

# out = runnable.invoke({
#     "input": "tell me about AI",
#     "chat_history": []
# })

# print(build_report(
#     output=out["intermediate_steps"][-1].tool_input
# ))

# """Let's ask about RAG specifically."""

# out = runnable.invoke({
#     "input": "what is retrieval augmented generation?",
#     "chat_history": []
# })

# print(build_report(
#     output=out["intermediate_steps"][-1].tool_input
# ))
