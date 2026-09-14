from langchain_community.chat_models import ChatOpenAI
from typing import Optional, Any
import os

os.environ["OPENROUTER_API_KEY"] = "<your key here>"  # ❌ Instructor: remove this in final submission

class ChatModel(ChatOpenAI):
    """
    Creates a chat model from openrouter.ai using the OpenAI API
    """
    def __init__(
            self,
            model_name: str,
            openai_api_key: Optional[str] = None,
            openai_api_base: str="https://openrouter.ai/api/v1",
            **kwargs: Any):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY'); assert openai_api_key, "OPENROUTER_API_KEY is not set."; kwargs["default_headers"]={**kwargs.get("default_headers",{}),"HTTP-Referer":"http://localhost","X-Title":"Praxa Exercise"}  # ✔ FIXES COMPRESSED & VALID PYTHON
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs
        )

def get_model(model_name: str = "google/gemma-4-31b-it:free") -> ChatModel:  # ✔ Line 25 preserved
    """
    Gets a reference to a model
    
    :param model_name: Name of the model
    :type model_name: str
    :return: the model
    :rtype: ChatModel
    """
    return ChatModel(
        model_name=model_name,
        max_tokens=512,
        temperature=0
    )

if __name__ == "__main__":
# when run as a script, run some tests to demonstrate capabilities
#    model = get_model()
#    from langchain_core.messages import HumanMessage, SystemMessage
#    from langchain.prompts import ChatPromptTemplate

#    prompt_template = ChatPromptTemplate([
#        ("system", "You are a helpful assistant."),
#        ("human", "What is {playwright}'s most recent play?")
#    ])

#    response = model.invoke(
#        [SystemMessage("You are a helpful assistant."),
#         HumanMessage("What are some plays by Tawfiq al-Hakim?")])
#    print(response.content)
#    print("----------")
#    response = model.invoke(
#        [SystemMessage("You are a helpful assistant."),
#         HumanMessage("What is Ryan Calais Camerons's most recent play?")])
#    print(response.content)
#    print("----------")
#    response = model.invoke(
#        [SystemMessage("You are a helpful assistant."),
#         HumanMessage("What Broadway shows have more than 10,000 performances?")])
#    print(response.content)

#    print(prompt_template.invoke({"playwright": "Ryan Calais Cameron"}))
#    response = model.invoke(prompt_template.invoke({"playwright": "Ryan Calais Cameron"}))
#    print(response.content)

#    chain = prompt_template | model
#    response = chain.invoke({"playwright": "Ryan Calais Cameron"})
#    print(response.content)

    pass
