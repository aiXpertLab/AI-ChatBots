"""
    Refactoring
    The ChainFactory class is responsible for creating the core query processing chain
    used by the chatbot. This chain combines document retrieval from a vector store
    and interaction with a large language model (LLM).

    Here's a breakdown of its functionality:

    1. **Vector Store Creation:**
    - Utilizes the `VectorStoreFactory` to create a vector store from PDF documents
        located in the specified `data_dir`. This involves loading documents, splitting
        them into text chunks, and generating embeddings for efficient retrieval.

    2. **LLM Loading:**
    - Employs the `LLMFactory` to load the Llama-2 large language model. You might need
        to adjust the `model_path` to point to the location where you have downloaded
        the model.

    3. **Prompt Template Creation:**
    - Leverages the `PromptTemplateFactory` to construct a prompt template that
        guides the LLM on how to answer user questions using the retrieved context.

    4. **Chain Assembly:**
    - Combines the retrieved documents (through the vector store), the loaded LLM,
        and the prompt template to construct a `RetrievalQA` chain. This chain
        orchestrates the retrieval of relevant documents based on user queries and
        feeds them along with the user question to the LLM for answer generation.

    5. **Return Value:**
    - Returns the constructed `RetrievalQA` chain, which is the core component
        used by the chatbot to process user queries.

    **Example Usage:**

    ```python
    chain = ChainFactory.create_chain(data_dir='path/to/your/pdfs')
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader


class VectorStoreFactory:
    @staticmethod
    def create_vector_store(data_dir):
        '''1. Load - Split - Embedd - FAISS DB. Create a vector store from PDF files'''
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


class LLMFactory:
    @staticmethod
    def load_llm():
        llm = LlamaCpp(
            model_path="e:/models/llama/llama-2-7b-chat.Q6_K.gguf",
            n_gpu_layers=40,
            n_batch=512,
        )
        return llm


class PromptTemplateFactory:
    @staticmethod
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

class ChainRetrievalQAFactory:
    @staticmethod
    def create_chain(data_dir):
        db = VectorStoreFactory.create_vector_store(data_dir)
        llm = LLMFactory.load_llm()
        prompt = PromptTemplateFactory.create_prompt_template()
        retriever = db.as_retriever(search_kwargs={'k': 2})
        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type='stuff',
                                            retriever=retriever,
                                            return_source_documents=False,
                                            chain_type_kwargs={'prompt': prompt})
        return chain