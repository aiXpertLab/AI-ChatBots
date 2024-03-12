from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

from utils.LangChain_Routine import llm
from utils.LangChain_Prompt  import GeneralPromptTemplate

# 1. Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = llm()

# 2. Define the prompt template with a placeholder for the question
# prompt = prompt()
prompt = GeneralPromptTemplate.create_prompt()


# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Chatbot initialized, ready to chat...")
question = ""
while question != "q":
    question = input(">>> ")
    answer = llm_chain.run(question)
    print(answer, '\n')