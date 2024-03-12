from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings

from utils.LangChain_Routine import load_doc, split_doc, store_doc
from utils.LangChain_Prompt  import MainPromptCreator, IceCreamPromptCreator
from conf import load_env
load_env()

# model_path ="e:/models/llama/llama-2-7b-chat.Q6_K.gguf"
model_path = "e:/models/llama/llama-2-7b.Q2_K.gguf"
data_dir   = "./data"
chain_type = 'stuff'

def embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
    return embedding_model

def create_chain():
    chain  = RetrievalQA.from_chain_type(chain_type = chain_type,
                                        llm         = LlamaCpp(model_path=model_path,  n_gpu_layers=40, n_batch=512),
                                        retriever   = store_doc(data_dir=data_dir, embedding_model=embedding_model()),
                                        return_source_documents=False,  chain_type_kwargs={'prompt': MainPromptCreator.create_prompt()})

    return chain
#---------------------------------------------------------------------------------
class Chatbot:
    def __init__(self):
        self.chain = create_chain()

    def query_doc(self, question):
        return self.chain({'query': question})['result']

    def run(self):
        print("Chatbot 1 for PDF files initialized, ready to query...")
        while True:
            question = input("------>> ")
            answer = self.query_doc(question)
            print(': ', answer, '\n')

def main():
    chatbot = Chatbot()
    chatbot.run()

if __name__ == "__main__":
    main()