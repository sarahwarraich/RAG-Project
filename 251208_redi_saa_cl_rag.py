###############################################################################
#
# 251208_redi_saa_cl_rag.py
#
# Purpose   : Final project for ReDI Machine Learning & AI Course Fall 2025
# Authors   : Sarah Akbar Ali, Chris Lai
#
# Before running this script:
# - ensure models availablility in your Ollama by the command: ollama list
# - set environment variable OLLAMA_CONTEXT_LENGTH at operating system level
# - ensure Ollama server is running by the command: ollama serve
#
###############################################################################

###############################################################################
#
# import libraries
#
###############################################################################

import os
import shutil
import time
import pprint
from decimal import Decimal
from dotenv import load_dotenv
from distutils.util import strtobool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

###############################################################################
#
# set global variables, referencing config.env in the same folder as this .py
#
###############################################################################

# folder where this script is located 
SCRIPT_PATH = os.path.dirname(__file__)  

# path to raw source data
DATA_PATH = os.path.join(SCRIPT_PATH, 'data')

# path to vector database chroma for storing embeddings
CHROMA_PATH = os.path.join(SCRIPT_PATH, 'chroma_db') 

# path to environment variable config file
ENV_PATH = os.path.join(SCRIPT_PATH, "config.env")

# load environment variables
load_dotenv(dotenv_path=ENV_PATH)

# models and their parameters
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL")
LLM_MODEL           = os.getenv("LLM_MODEL")
CONTEXT_CAPACITY    = int(os.getenv("CONTEXT_CAPACITY"))

# index documents or not
INDEX_DOCUMENTS = bool(strtobool(os.getenv("INDEX_DOCUMENTS", "False")))
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP"))
TEMPERATURE     = Decimal(os.getenv("TEMPERATURE"))


###############################################################################
#
# define functions
#
###############################################################################

# loads documents from the specified global data path
def load_documents():

    documents = [] # Main list of sll documents
    doc_info = []   # To store start/end page details
    page_index_counter = 0

    for filename in os.listdir(DATA_PATH):

        if filename.endswith(".pdf"):

            pdf_path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()  # loads each page as a Document object

            # Add filename into metadata
            for idx, d in enumerate(docs):
                d.metadata["document_name"] = filename
                d.metadata["page"] = idx + 1

            # record start and end page index for the document
            start = page_index_counter
            end = page_index_counter + len(docs) - 1

            # feeding Payload
            doc_info.append({
                "document": filename,
                "start_page_index": start,
                "end_page_index": end,
                "num_pages": len(docs)
            })

            # Update counter for next PDF
            page_index_counter += len(docs)
            documents.extend(docs)

            print(f"Loaded {len(docs)} pages from {filename}")

    print("Total Loaded", len(documents), "pages successfully!")

    return documents, doc_info


# splits documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size      =   CHUNK_SIZE,
        chunk_overlap   =   CHUNK_OVERLAP,
        length_function =   len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks")
    return all_splits


# initializes the Ollama embedding function
def get_embedding_function(model_name=EMBEDDING_MODEL):
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Initialized embeddings model: {model_name}")
    return embeddings


# initializes or loads the Chroma vector store
def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    vectorstore = Chroma(
        persist_directory   =   persist_directory,
        embedding_function  =   embedding_function
    )
    print(f"Connected to vector store at {persist_directory}")
    return vectorstore


# indexes document chunks into the Chroma vector store
def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    print(f"Indexing {len(chunks)} chunks...")
    # Use from_documents for initial creation.
    # This will overwrite existing data if the directory exists but isn't a valid Chroma DB.
    # For incremental updates, initialize Chroma first and use vectorstore.add_documents().
    vectorstore = Chroma.from_documents(
        documents           = chunks,
        embedding           = embedding_function,
        persist_directory   = persist_directory
    )
    #vectorstore.persist() # Ensure data is saved , but with latest version not require to call explicitly.
    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vectorstore


# Context cleaner: replaces Document objects with readable text
def clean_context(documents):
    clean = ""
    for d in documents:
        name = d.metadata.get("document_name", "Unknown Document")
        page = d.metadata.get("page", "N/A")
        clean += f"\n[Source: {name} | Page: {page}]\n{d.page_content}\n"
    return clean

# initialize rag settings
def create_rag_chain(vector_store, llm_model_name=LLM_MODEL, context_window=CONTEXT_CAPACITY):
    
    # Initialize the LLM
    llm = ChatOllama(
        model       =   llm_model_name,
        temperature =   TEMPERATURE,        # Lower temperature for more factual RAG answers
        num_ctx     =   context_window      # IMPORTANT: set context window size in system env variable too
    )
    print(f"Connected to LLM model: {llm_model_name}")

    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type     =   "mmr",   # or "similarity"
        search_kwargs   =   {"k": 3, "lambda_mult": 0.25}        # Retrieve top 3 relevant chunks, lambda_mult is 1 for minimum diversity and 0 for maximum. (Default: 0.5)
        
    )
    print("Retriever initialized.")

    # define prompt templates
    system_prompt = """
        You are a helpful AI assistant. 
        Use the provided Context in your Answer.
    """

    # user prompts are best followed as a part of the "question"
    human_prompt = """
        Use the provided Context in your Answer.

        Context: 
        {context}
        
        Question: 
        {question}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user",  human_prompt)
    ])
    print("Prompt templates created.")

    # Define the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | clean_context, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    print("RAG chain created.")
    
    return rag_chain


# process questions and answers
def process_chat(message, history):
    
    # avoid an empty prompt
    if not message:
        yield "You did not type anything."
        return

    # handle a prompt entered in multiple lines
    oneliner = ""
    for i in range(len(message)):
        oneliner = message[: i + 1]

    # instructions are best followed when they are part of the question
    oneliner = oneliner + """
 
        
        Format your Answer as follows:
        - Summary
        - Key Evidence from Context with Document name

    """

    print("\nNew question:", oneliner)

    yield "Thinking..."

    # start measuring llm response time
    start = time.time()
    print(f'New question from user.  Start time: {start}')

    # ask llm the question
    response = rag_chain.invoke(oneliner)

    # end measuring llm response time
    end = time.time()
    print(f'Elapsed: {end - start:.2f} seconds')

    # return response from llm to web UI
    yield str(response)

    # update history with latest response
    # history = history + [(message, response)]
    # print(history)
    
    # clear input; event completes
    return history, gr.update(value="")


# initiate chat with selected functionalities
# remove cached responses by deleting folders under .gradio\cached_examples
chat = gr.ChatInterface(
    process_chat,
    title               = os.getenv("TITLE"),
    chatbot             = gr.Chatbot(height=400),
    flagging_mode       = "manual",
    flagging_options    = ["Like"],
    save_history        = True,
    examples            = [
        "What is the German government's number one AI goal in its high tech agenda?.Please answer in English", 
        "What are Gemany's weaknesses in AI?.Please answer in English", 
        "What are Gemany's strengths in AI?.Please answer in English", 
        "What are the German government's other goals in its high-tech agenda?.Please answer in English",
        "Was ist das Ziel Nummer 1 der deutschen Regierung im Bereich der KI?.Please answer in German", 
        "Wo liegen die Schwächen Deutschlands im Bereich KI?.Please answer in German", 
        "Wo liegen die Stärken Deutschlands im Bereich KI?.Please answer in German", 
        "Was sind die weiteren Ziele der deutschen Regierung im Rahmen ihrer Hightech-Agenda?.Please answer in German"
    ],
    cache_examples      = True,
    submit_btn          = True
)


###############################################################################
#
# conditional execution:
# - the followings will be executed if this script is being run directly
# - the followings will NOT run if this script is being imported by another
#
###############################################################################
if __name__ == "__main__":

    print("\n")
    print("\n")
    print("Welcome to our RAG!")
    print("\n")

    print("Embedding model  :", EMBEDDING_MODEL)
    print("Chunk size       :", CHUNK_SIZE)
    print("Chunk overlap    :", CHUNK_OVERLAP)
    print("Index documents? :", INDEX_DOCUMENTS)
    index_exists = os.path.exists(CHROMA_PATH)
    print("Indices exist?   :", index_exists)
    
    print("LLM model        :", LLM_MODEL)
    print("Context window   :", CONTEXT_CAPACITY)
    print("Temperature      :", TEMPERATURE)
    print("\n")

    # needed for both
    # - one-off indexing of source PDFs, and
    # - conversion of user questions to embeddings for similarity search
    embedding_function = get_embedding_function()

    # create new indices if either:
    # - config.env says so, or 
    # - indices do not already exist
    if INDEX_DOCUMENTS or (index_exists == False):

        print("\nCreate new document indices")

        # start measuring indexing time
        start = time.time()
        print(f'Start time: {start}')

        # remove existing indices, if any, before creating new ones
        if index_exists:
            print(f"Delete existing '{CHROMA_PATH}'.")
            shutil.rmtree(CHROMA_PATH)

        print("\nLoading documents")
        docs, doc_info = load_documents()

        print("\nPayload meta data:")
        pprint.pprint(doc_info, sort_dicts=False)

        print("\nSpliting documents")
        chunks = split_documents(docs)

        # index documents and store embeddings in chromadb vector store
        print("\nIndexing documents")
        vector_store = index_documents(chunks, embedding_function)

        # end measuring indexing time
        end = time.time()
        print(f'Elapsed: {end - start:.2f} seconds')

    else:

        # connect to the existing chromadb vector store
        vector_store = get_vector_store(embedding_function)

    print("\nGetting ready to chat...")

    # initialize connection to llm and prompt template etc.
    rag_chain = create_rag_chain(vector_store, llm_model_name=LLM_MODEL) 

    print("Ready to chat!\n")

    # start ui to take questions
    chat.launch()
