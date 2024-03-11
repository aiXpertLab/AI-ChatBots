from utils.LangChain_Routine import create_chain
from conf import load_env
load_env()

class Chatbot:
    def __init__(self):
        self.chain = create_chain('./data')

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