"""
1. Command Line Chatbot
This Python script implements a basic chatbot capable of answering questions about PDF documents from command line.

**Features:**

* Retrieves relevant PDF documents using a vector store (e.g., FAISS) for efficient information retrieval.
* Queries a large language model (LLM) like Llama-2 to generate informative responses based on the retrieved PDFs.
* Provides a user-friendly command-line interface to interact with the chatbot.

**Structure:**

* `utils.factories.ChainFactory`: Creates the core query processing chain, combining document retrieval and LLM interaction.
* `Chatbot`: Manages the chatbot's functionality, including user input processing, query execution, and answer presentation.
* `main`: The entry point of the program, instantiates the chatbot and initiates the chatbot loop.

**Example Usage:**

1. Run the script: `python your_script_name.py`
2. The chatbot will prompt you with "Chatbot for PDF files initialized, ready to query...".
3. Enter your question about the PDF content, and the chatbot will retrieve, analyze, and respond based on the relevant documents.

**Note:**

* This is a foundational implementation and can be extended with features like:
    * Configuration options for model paths, document directories, and retrieval parameters.
    * More sophisticated user interface (e.g., web-based).
    * Error handling and logging for robustness.

**Enjoy exploring and potentially customizing this chatbot for your specific needs!**
"""
from conf import load_env
load_env()

from utils.factories_Llama2 import ChainRetrievalQAFactory

class Chatbot:
    def __init__(self):
        self.chain = ChainRetrievalQAFactory.create_chain('./data')


    def query_doc(self, question):
        return self.chain({'query': question})['result']


    def run(self):
        print("Chatbot 1 for PDF files initialized, ready to query...")
        while True:
            question = input("--> ")
            answer = self.query_doc(question)
            print(': ', answer, '\n')


def main():
    chatbot = Chatbot()
    chatbot.run()

if __name__ == "__main__":
    main()