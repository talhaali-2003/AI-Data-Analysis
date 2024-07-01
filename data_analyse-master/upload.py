import os
from langchain_community.document_loaders import DirectoryLoader
from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
 
load_dotenv()

directory = 'data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print(len(documents))

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

from langchain_community.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

os.getenv('PINECONE_API_KEY', '')
os.getenv('PINECONE_INDEX_NAME', '')



index_name = "ai-assistant"

docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)



pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))

def get_similiar_docs(query,k=2,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs
  
  
query = "Who took over Twitter"
similar_docs = get_similiar_docs(query)
print(similar_docs)