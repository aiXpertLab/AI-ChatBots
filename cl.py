from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain

import chainlit as cl

from utils.LangChain_Prompt  import IceCreamPromptCreatorMemory
from utils.LangChain_Routine import llm



llm = llm()
prompt = IceCreamPromptCreatorMemory.create_prompt()
conversation_memory = ConversationBufferMemory(memory_key="chat_history",max_len=50,return_messages=True,)

@cl.on_chat_start
def quey_llm():
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=conversation_memory)
    cl.user_session.set("llm_chain", llm_chain)
    # print(llm_chain.invoke({'question': question})['text'])

@cl.on_message
async def query_llm(question):
    lm_chain = cl.user_session.get("llm_chain")
    
    respons = await lm_chain.acall(question.content, callbacks = [cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(respons["text"]).send()
