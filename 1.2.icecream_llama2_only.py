from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
# from prompts import ice_cream_assistant_prompt_template
from utils.LangChain_Prompt  import IceCreamPromptCreator


llm = LlamaCpp(
    model_path = "e:/models/llama/llama-2-7b-chat.Q6_K.gguf",
    n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
)

# prompt = ice_cream_assistant_prompt_template
prompt = IceCreamPromptCreator.create_prompt()
llm_chain = LLMChain(llm=llm, prompt=prompt)

def query_llm(question):
    print(llm_chain.invoke({'question': question})['text'])

class Chatbot:
    def __init__(self):
        self.chain = llm_chain

    def query_doc(self, question):
        return self.chain.invoke({'query': question})['text']

    def run(self):
        print("Chatbot 1 for PDF files initialized, ready to query...")
        while True:
            question = input("--> ")
            answer = query_llm(question)
            print(': ', answer, '\n')


def main():
    chatbot = Chatbot()
    chatbot.run()

if __name__ == "__main__":
    main()