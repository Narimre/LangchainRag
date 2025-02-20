import bs4
import os
import getpass
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
import gradio as gr  # For the interactive chat window
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.agents import AgentType
import requests
#import langchain_unstructured
from langchain_community.document_loaders import UnstructuredFileLoader

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
# Load, chunk and index the contents of the blog.
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     #web_paths=("https://en.wikipedia.org/wiki/Wikipedia:Very_short_featured_articles",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )


file_path = 'Book.txt'
loader = UnstructuredFileLoader(file_path)

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
# Create a vector store with embeddings
#embeddings = OpenAIEmbeddings()
embeddings = OllamaEmbeddings(model="all-minilm:latest",)

vector_store = FAISS.from_documents(docs, embeddings)# Initialize a retriever for querying the vector store
retriever = vector_store.as_retriever(search_type="similarity", search_k=3)
# Retrieve and generate using the relevant snippets of the blog.
#retriever = vectorstore.as_retriever()



#  Define your custom prompt template
# custom_prompt_template = """
# {context}

# Question: {question}
# """

# # Create a PromptTemplate instance with your custom template
# custom_prompt = PromptTemplate(
#     template=custom_prompt_template,
#     input_variables=["context", "question"],
# )

# # Use your custom prompt when creating the ConversationalRetrievalChain
# qa = ConversationalRetrievalChain.from_llm(
#     cfg.llm,
#     verbose=True,
#     retriever=retriever,
#     memory=memory,
#     combine_docs_chain_kwargs={"prompt": custom_prompt},
# ) 



# Initialize the LLM
# llm = ChatOpenAI(model="gpt-4o")
llm = OllamaLLM(model="llama2-uncensored:latest")

# Example: Weather API function
def get_weather(location: str):
    return f"The current weather in {location} is sunny and 25°C."


# 2. Define the API tool (same as before)
def get_parameters_from_api(product_id):
    """Fetches parameters for a product from an API."""
    api_url = f"YOUR_API_ENDPOINT/{product_id}"  # Replace with your API endpoint
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        parameters = response.json()
        return parameters
    except requests.exceptions.RequestException as e:
        return f"Error fetching parameters: {e}"
    
def add_btn_to_page(text):
    print("text,color"+ str(text))
    return f"Success"

def add_btn_array_to_page(button_array):
    return 0

def clear_page():
    return 0

def verify_page():
    return f"Success"
def VerifyBtnOnPage():
    return f'missing'

def GetButtonArrayTemplate(button_array_name):
    ret_str= '{"buttons": [{Button1 with parameters}, {Button2 with parameters}, {Button3 with parameters}, {Button4 with parameters}]}'
    return ret_str

# Create the Retrieval QA chain
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Combine tools and retrieval chain
tools = [
    Tool(
        name="Document Retrieval",
        func=lambda q: retrieval_qa_chain({"query": q})["result"],
        description="Retrieve detailed Button configuration knowledge from the document database."
    ),
    Tool(
        name="AddButton",
        func=add_btn_to_page,
        description="Useful for adding a configured button to the webpage without checking.  Inputs should be two Button configured parameters.",
    ),
    Tool(
        name="ClearPage",
        func=clear_page,
        description="Useful for clearing the webpage.",
    ),
    Tool(
        name="AddRuttonArray",
        func=add_btn_array_to_page,
        description="Useful for adding multiple configured buttons the webpage using only info from the tools here. Inputs should be a json formated text with exactly the same format and labels as described the template Array and formated as descripted in the documents. "
    ),
    Tool(
        name="VerifyBtnOnPage",
        func=verify_page,
        description="Useful for verifying the webpage.",
    ),
    Tool(
        name="GetButtonArrayTemplate",
        func=GetButtonArrayTemplate,
        description="Useful for retrieving the button array template for correct format. Input should be the Button number",
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# to customize agent prompt
# sys_message = "YOUR NEW PROMPT IS HERE"
# zero_shot_agent.agent.llm_chain.prompt.template = sys_message


def response(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = agent.run(history_langchain_format)
    return gpt_response

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(placeholder="<strong>Button configurator</strong><br>I will modify the webpage for you")
    chatbot.like(vote, None, None)
    #gr.ChatInterface(fn=response, type="messages")
    gr.ChatInterface(fn=response, type="messages")
    
demo.launch()