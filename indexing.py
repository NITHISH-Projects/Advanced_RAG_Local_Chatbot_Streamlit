from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings



########################################## Indexing ####################################################
pdf_loader = PyPDFDirectoryLoader("./docs")
loaders= [pdf_loader]

documents = []
for loader in loaders :
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total no of documents:{len(all_documents)}")

#batch size - total no of docs accepted by embedding in single run
batch_size = 4

num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)
print(num_batches)
db = Chroma(embedding_function=GPT4AllEmbeddings(), persist_directory="./chromadb")


retv = db.as_retriever()
print("loading started")

for batch_num in range(num_batches) :
    start_indx = batch_num * batch_size
    end_idx = (batch_num + 1) * batch_size
    batch_docs = all_documents[start_indx:end_idx]
    print("load about to start")
    retv.add_documents(batch_docs)
    print(start_indx, end_idx)
db.persist()
