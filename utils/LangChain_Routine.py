from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader

def load_doc(data_dir):
    '''Load PDF documents from the specified directory.'''
    loader = DirectoryLoader(path=data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def split_doc(documents):
    '''Split documents into chunks for processing.'''
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splited_documents = splitter.split_documents(documents)
    return splited_documents


def store_doc(data_dir, embedding_model):
    documents = load_doc(data_dir)
    texts = split_doc(documents)
    db = FAISS.from_documents(texts, embedding_model)
    retriever = db.as_retriever(search_kwargs={'k': 2})
    return retriever

def llm():
    llm= LlamaCpp(
        model_path = "e:/models/llama/llama-2-7b-chat.Q6_K.gguf",
        n_gpu_layers=40,
        n_batch=512,  # Batch size for model processing
    )
    return llm

