# Chat-AI

1. chatbot_llama2.py:

This code demonstrates a basic chatbot built with LangChain and Llama-2.  LangChain is a framework for building creative AI workflows, while Llama-2 is a powerful large language model (LLM) capable of generating text. Here's a breakdown of the steps involved:

Import Libraries: We start by importing the necessary libraries:
LlamaCpp from langchain_community.llms: This provides access to the Llama-2 LLM through a C++ interface.
PromptTemplate and LLMChain from langchain.prompts and langchain.chains respectively: These components help define the interaction flow with the LLM using prompts.
Load Llama-2 Model: The LlamaCpp object is created, specifying the path to the pre-trained Llama-2 model (models/llama-2-7b-chat.Q4_0.gguf) and adjusting the number of GPU layers (n_gpu_layers) and batch size (n_batch) based on your hardware capabilities.
Define Prompt Template: A PromptTemplate object is created. This template defines the structure of the prompt presented to the LLM, including placeholders for user input. Here, the template uses "Question:" followed by a placeholder {question} and an "Answer:" section.
Create LLMChain: An LLMChain object is created. This chain connects the prompt template and the Llama-2 model, essentially defining the workflow of how user questions will be processed by the LLM and how the answers will be generated.
Chatbot Loop: The code enters a loop where it:
Prompts the user for a question with >.
Uses the LLMChain.run(question) method to pass the user's question to the LLM chain.
Prints the generated answer from the LLM.
This basic structure allows users to interact with the Llama-2 model through a question-and-answer format. You can further customize this code to explore different functionalities of LangChain and Llama-2.

