from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
model_path="e:/models/llama/llama-2-7b-chat.Q6_K.gguf"

def d1_load_documents(data_dir):
    '''Load PDF documents from the specified directory.'''
    loader = DirectoryLoader(path=data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def d2_split_documents(documents):
    '''Split documents into chunks for processing.'''
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splited_documents = splitter.split_documents(documents)
    return splited_documents


def d3_embedding_vector(data_dir):
    documents = d1_load_documents(data_dir)
    texts = d2_split_documents(documents)
    db = FAISS.from_documents(texts, embedding_model)
    retriever = db.as_retriever(search_kwargs={'k': 2})
    return retriever


def prompt():
    # prepare the template we will use when prompting the AI
    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".

    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])
    return prompt


def create_chain(data_dir):
    chain  = RetrievalQA.from_chain_type(chain_type ='stuff',
                                        llm         = LlamaCpp(model_path=model_path,  n_gpu_layers=40, n_batch=512),
                                        retriever   = d3_embedding_vector(data_dir=data_dir),
                                        return_source_documents=False,
                                        chain_type_kwargs={'prompt': prompt()})
    return chain