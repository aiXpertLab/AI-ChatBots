def main():

    print("Chatbot for PDF files initialized, ready to query...")
    while True:
        print("1 initialized, ready to query...")
        question = input("> ")
        answer = "query_doc(chain, question)"
        print(': ', answer, '\n')

main()