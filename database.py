# database.py


from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

data = PyPDFLoader("document_loaders/Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.pdf")
docs = data.load()

splitters = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunk = splitters.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store =  Chroma.from_documents(
    documents= chunk,
    embedding= embedding_model,
    persist_directory= "db-chroma"
)




