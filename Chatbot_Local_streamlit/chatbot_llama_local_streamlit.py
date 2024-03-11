import streamlit as st

from langchain_community.llms import LlamaCpp

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

def init_page():
    st.set_page_config(
        page_title="Personal ChatGPT"
    )
    st.header("Personal ChatGPT")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Respond your answer in mardkown format.")
        ]
        st.session_state.costs = []
        

# def select_model():
#     model_name = st.sidebar.radio("Choose LLM:",
#                                   ("gpt-3.5-turbo-0613", "gpt-4"))
#     temperature = st.sidebar.slider("Temperature:", min_value=0.0,
#                                     max_value=1.0, value=0.0, step=0.01)
#     return ChatOpenAI(temperature=temperature, model_name=model_name)

def llm_select_model():
    model_path = st.sidebar.radio("Choose LLM:",
                                  ("e:/models/llama/llama-2-7b-chat.Q6_K.gguf", "gpt-4"))
    llm = LlamaCpp(model_path=model_path, n_gpu_layers=40, n_batch=512)
    return llm

def get_answer(llm, messages):
    llm_chain = LLMChain(prompt=messages, llm=llm)
    answer = llm_chain.run(messages)
    return answer.content


# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

st.title("Chatbot Interface")

question = st.text_input("You:", "")

if st.button("Ask"):
    if question:
        answer = llm_chain.run(question)
        st.text_area("Bot:", value=answer, height=200)
    else:
        st.warning("Please enter a question.")


def main():
    _ = load_dotenv(find_dotenv())

    init_page()
    llm = llm_select_model()
    init_messages()

    # Supervise user input
    
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            answer = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


if __name__ == "__main__":
    main()