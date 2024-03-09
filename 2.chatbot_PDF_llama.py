from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

def create_vector_store(data_dir):
    '''Create a vector store from PDF files'''
    # define what documents to load
    loader = DirectoryLoader(path=data_dir, glob="*.pdf", loader_cls=PyPDFLoader)

    # interpret information in the documents
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    # create the vector store database
    db = FAISS.from_documents(texts, embeddings)
    return db

def load_llm():
    # Adjust GPU usage based on your hardware
    llm = LlamaCpp(
        model_path = "e:/models/llama/llama-2-7b-chat.Q6_K.gguf",
        n_gpu_layers=40,  # Number of GPU layers (adjust based on available GPUs)
        n_batch=512,  # Batch size for model processing
    )
    return llm


def create_prompt_template():
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

def create_chain():
    db = create_vector_store(data_dir='data')
    llm = load_llm()
    prompt = create_prompt_template()
    retriever = db.as_retriever(search_kwargs={'k': 2})
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=False,
                                        chain_type_kwargs={'prompt': prompt})
    return chain

def query_doc(chain, question):
    return chain({'query':question})['result']


def main():
  chain = create_chain()

  while True:
      print("Chatbot for PDF files initialized, ready to query...")
      question = input("> ")
      answer = query_doc(chain, question)
      print(': ', answer, '\n')

main()