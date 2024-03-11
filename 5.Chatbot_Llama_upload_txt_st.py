import streamlit as st 

from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from langchain.prompts import PromptTemplate
from langchain.chains  import LLMChain
from langchain.text_splitter import CharacterTextSplitter

# Customize the layout
st.set_page_config(page_title="Learning Godel", page_icon="🤖", layout="wide", )     
st.markdown(f"""<style>.stApp {{background-image: url("https://aixpertlab.netlify.app/images/background.png"); background-attachment: fixed;background-size: cover}}style>""", unsafe_allow_html=True)

# function for writing uploaded file in temp
def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False

# set prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# initialize hte LLM & Embeddings
llm = LlamaCpp(
    model_path = "e:/models/llama/llama-2-7b.Q2_K.gguf",
    n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
    # verbose=False,  # Enable detailed logging for debugging
)

embeddings = LlamaCppEmbeddings(model_path="e:/models/llama/llama-2-7b.Q2_K.gguf",)
llm_chain  = LLMChain(llm=llm, prompt=prompt)

st.title("📄 Document Conversation 🤖")
uploaded_file = st.file_uploader("Upload an article", type="txt")

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    # st.write(content)
    file_path = "temp/chatbot.txt"
    write_text_file(content, file_path)   
    loader = TextLoader(file_path)
    print(content)
    print(file_path)
    docs = loader.load()    
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=1)
    texts = text_splitter.split_documents(docs)
    db = Chroma.from_documents(texts, embeddings)    
    st.success("File Loaded Successfully!!")
    
    # Query through LLM    
    question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?", disabled=not uploaded_file,)    
    if question:
        similar_doc = db.similarity_search(question, k=1)
        context = similar_doc[0].page_content
        query_llm = LLMChain(llm=llm, prompt=prompt)
        response = query_llm.run({"context": context, "question": question})        
        st.write(response)