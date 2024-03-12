from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

class PromptCreator(ABC):
    # Factory/Creator: In this code, the factory is represented by the PromptCreator abstract base class (ABC) and its concrete subclasses (MainPromptCreator and IceCreamPromptCreator). 
    # The factory is responsible for creating instances of PromptTemplate objects based on certain conditions.
    @abstractmethod
    def create_prompt(self, context):pass

# class PromptTemplate:
#     #Product/ConcreteProduct: The product is represented by the PromptTemplate class. It defines the interface for creating prompt templates. 
#     # Instances of PromptTemplate are the products created by the factory.
#     def __init__(self, template, input_variables):
#         self.template = template
#         self.input_variables = input_variables

#     def create_prompt(self, context):
#         return self.template.format(context=context, question="What is your question?")

class MainPromptCreator(PromptCreator):
    #Concrete Creator: MainPromptCreator and IceCreamPromptCreator are concrete creators. 
    # They subclass PromptCreator and override the create_prompt() method to create instances of PromptTemplate with different templates and input variables, depending on the context.
    @staticmethod
    def create_prompt():
        main_template = """Use the provided context to answer the user's question. If you don't know the answer, respond with "I do not know".

        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(
            template=main_template,
            input_variables=['context', 'question'])
        return prompt


class IceCreamPromptCreator():
    #Concrete Creator: MainPromptCreator and IceCreamPromptCreator are concrete creators. 
    # They subclass PromptCreator and override the create_prompt() method to create instances of PromptTemplate with different templates and input variables, depending on the context.
    @staticmethod
    def create_prompt():
        ice_cream_assistant_template = """
        Question: {question} 
        Answer:
        """

        prompt = PromptTemplate(
            template=ice_cream_assistant_template,
            input_variables=["question"])
        
        return prompt  
    
class IceCreamPromptCreatorMemory():
    #Concrete Creator: MainPromptCreator and IceCreamPromptCreator are concrete creators. 
    # They subclass PromptCreator and override the create_prompt() method to create instances of PromptTemplate with different templates and input variables, depending on the context.
    @staticmethod
    def create_prompt():
        ice_cream_assistant_template = """
        Question: {question} 
        Answer:
        """

        prompt = PromptTemplate(
            template=ice_cream_assistant_template,
            input_variables=["chat_history", "question"])
        
        return prompt  
    

class GeneralPromptTemplate(PromptCreator):
    @staticmethod
    def create_prompt():
        template = """
        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        return prompt   
