import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma



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


####### MAIN #########

client = chromadb.PersistentClient(path="./chromadb")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

hb = client.heartbeat()
collection = client.get_or_create_collection(name="test",metadata={"hnsw:space": "cosine"}) 

print("Statistic before")
statistic(collection)
print("")

directory = './data'
print(f"Loading documents from {directory}")
documents = load_docs(directory)
nb_doc = len(documents)
print(f"{nb_doc} documents in the folder")
print("")

print("Spliting documents")
docs = split_docs(documents)
nb_splitted_doc = len(docs)
print(f"We now have {nb_splitted_doc} documents after splitting" )
print("")

print("Adding documents to the vectoreBase")
for i, doc in enumerate(docs):
    # Generate a unique ID for each chunk; adjust this according to your needs
    doc_id = f"doc_{i}"
   
    # Add the document chunk to Chroma
    collection.add( documents=doc.page_content, ids=doc_id, metadatas=doc.metadata)

print("")
print("Statistic after")
statistic(collection)
print("")

