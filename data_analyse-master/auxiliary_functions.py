from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

## For chroma
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings

## Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


# configure chroma db 
client = chromadb.PersistentClient(path="./chromadb")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="test",metadata={"hnsw:space": "cosine"}) 


# configure Pinecone client
use_serverless = os.environ.get("USE_SERVERLESS", "False").lower() == "true"
pineconne_api = os.getenv('PINECONE_API_KEY')
environment = "gcp-starter"
index_name="ai-assistant"
pc = Pinecone(api_key=pineconne_api)

if index_name not in pc.list_indexes().names():
      pc.create_index(
          name=index_name, 
          dimension=384, 
          metric='cosine',
          spec=PodSpec
      )
      
## accesing Index
index = pc.Index(index_name)

## Define LLm
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo-16k",
    temperature=0
)   

def _parse(text):
    return text.strip("**")


def find_match_chroma(input):
    print("find_match_chroma")
    
    result = collection.query(query_texts=input, n_results=2)
    print(result)
    return result['documents'][0]

    
    
def find_match(input):
    print("find_match")
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, includeMetadata=True)
    print(result)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']


# Parser to remove the `**`
def _parse(text):
    return text.strip("**")

def query_refiner(conversation, query):

    rewrite_template = """Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.
    CONVERSATION LOG: {conversation}
    Query: {query}
    Refined Query:
    """

    rewrite_prompt = PromptTemplate.from_template(rewrite_template)
    rewriter = rewrite_prompt | llm | StrOutputParser() | _parse
    
    return rewriter.invoke({"conversation": conversation,"query": query})
    

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

