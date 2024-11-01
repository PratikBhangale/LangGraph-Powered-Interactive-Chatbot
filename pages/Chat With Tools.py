import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import ollama
from langchain_core.output_parsers import StrOutputParser
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


from langchain_ollama import ChatOllama
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List, Annotated

# Creating an Array for storing the chat history for the model.
context = []


# Set the title of the Streamlit app
st.set_page_config(layout="wide")#, page_title="Llama 3 Model Document Q&A"
st.title("LTIMindtree Basis Chat With LLM")

# Creating a Session State array to store and show a copy of the conversation to the user.
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Create the Sidebar
sidebar = st.sidebar

# Create the reset button for the chats
clear_chat = sidebar.button("Clear Chat")
if clear_chat:
    context = []
    st.session_state.messages =[]



# Tavily Search Tool
web_search_tool = TavilySearchResults(k=3)


# Defining out LLM Model
local_llm = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")




# Prompt for ROuting to rag or websearch
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to Pratik Bhangale and his resume.
                                    
Use the vectorstore for questions on those topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""





# Prompt for generating an answer
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""



# Retrieval Grader Prompts

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains information related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains information that is relevant to the question."""




# Retriver tool
vector_db=FAISS.load_local('Database', OllamaEmbeddings(model='all-minilm'), allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(k=3)


# Post-processing for rertriver tool
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



### Initialising the variables for the graph

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    documents: List[str]  # List of retrieved documents



### Creating the Nodes


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    st.write("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    st.write("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    correct_docs = 0
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            st.write("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            correct_docs+=1
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            st.write("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            # web_search = "Yes"
            continue
    
    if correct_docs<1: web_search = "Yes"
    return {"documents": filtered_docs, "web_search": web_search}



def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    st.write("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    st.write("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation}





### Creating the Edges


def route_question(state):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    st.write("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        st.write("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        st.write("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):

    print("---ASSESS GRADED DOCUMENTS---")
    st.write("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"




### Creating the Graph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate


# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)


workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

workflow.add_edge("generate", END)





# Compile
graph = workflow.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))








# Create the function to stream output from llm
def get_response(question):

    quest = {"question": question}

    for event in graph.stream(quest, stream_mode="values"):
        if 'generation' in event:
            return event['generation'].content, event
        







# ------------------------------------------------------------------------------------------------------------------------------

def start_app():

        try:
            OLLAMA_MODELS = ollama.list()["models"]
        except Exception as e:
            st.warning("Please make sure Ollama is installed and running first. See https://ollama.ai for more details.")
            st.stop()

        question = st.chat_input("Ask Anything.", key=1)

        if question:
            
            # with st.chat_message("Human"):
            #     st.write(question)

            st.session_state.messages.append({"role": "user", "content": question})
            context.append(HumanMessage(content=question))



            response, event = get_response(question)
            # with st.chat_message("AI"):
            #     st.write(response)
            # with st.chat_message("AI"):
            #     # response = st.write(get_response(question))
                            
            st.session_state.messages.append({"role": "assistant", "content": response})



            context.append(AIMessage(content=str(response)))


            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])



if __name__ == "__main__":
    start_app()