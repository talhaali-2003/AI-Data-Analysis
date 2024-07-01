import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import os

os.environ["OPENAI_API_VERSION"]="2023-03-15-preview"
os.environ["AZURE_OPENAI_ENDPOINT"]="https://guicusaest.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "42ca3bc0eb984fd9839e607d42a891cd"

# Parser to remove the `**`
def _parse(text):
    return text.strip("**")

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def statistic(col):
    nb = col.count()
    print(f"there are {nb} Items in the collection")
    
    return
    
    
def find_match_chroma(input):
    print("find_match_chroma")
    result = collection.query(query_texts=input, n_results=2)
    print(result)
    return result['documents'][0]
    
    
def query_refiner(conversation, query):

    rewrite_template = """Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.
    CONVERSATION LOG: {conversation}
    Query: {query}
    Refined Query:
    """

    rewrite_prompt = PromptTemplate.from_template(rewrite_template)
    rewriter = rewrite_prompt | llm | StrOutputParser() | _parse
    
    return rewriter.invoke({"conversation": conversation,"query": query})


## Define LLm
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo-16k",
    temperature=0
)   

####### MAIN #########

client = chromadb.PersistentClient(path="./chromadb")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

hb = client.heartbeat()
collection = client.get_or_create_collection(name="test",metadata={"hnsw:space": "cosine"}) 

#print("Statistic before")
#statistic(collection)
#print("")

#find_match_chroma("what is osp")


stuff_prompt_template = """
You're an IT communication expert, write a professional concise summary of the following text.Get to the point.
Try to extract the start date and time of the incident. 
Try to find the end date and time of the incident otherwise say ongoing.
Use the following template to answer and you will find below an example.

'{text}'

Template#

Start:
End:
Duration:


IMPACT:
ROOT CAUSE:
SUMMARY: 
RESOLUTION:

Example #1

Start : 2024/03/04 13:12
End : 2024/03/04 14:40
Duration : (00d) 01:27
Calls generated : 0
Users involved : 0


IMPACT
- Problem description :  Accessing the application is impossible using https://refvr.inetpsa.com/rvr/accueil.action.

************************
- Users impact :   Users can not provide any finance calculation for all brands (New and Used vehicles).

- Business / Job impact:  Major


ROOT CAUSE
- Change or incident potentially linked?  INCI12156102, INCI12155405
- Previous automatic incident / lack of monitoring?  NO
- Root Cause:   Indus launched a synchronisation on the two DB slave servers in the same time.


SUMMARY
- On  Friday that was asked indus to synchronize the 2  slave Database to master Database.(INCI12156102)
- He did that actions on the two servers in same time that impact the proper functioning of the application today
- This refresh action can take 2 hours.

The envisaged solution consists of pointing webservice requests only to the master database while the two slave database servers are resynchronized.

SOLUTION
Service restored when the applicative expert redirected all web-services request to the master database.

"""

conversation = ""
refined_query = query_refiner(conversation,stuff_prompt_template)

print(refined_query)