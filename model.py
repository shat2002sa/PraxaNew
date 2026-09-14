from langchain_community.chat_models import ChatOpenAI
from typing import Optional, Any
import os

# ❗ CHANGED: replaced placeholder with actual environment variable usage
# (Before: os.environ["OPENROUTER_API_KEY"] = "<your key here>")
# This line must NOT hardcode the key; the school exercises expect env vars.
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

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

        # ❗ CHANGED: correctly load the OpenRouter key
        # Before: openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        # This is correct, but the wrapper needed the correct header name.
        openai_api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")

        # ❗ CHANGED: added required OpenRouter headers
        # ChatOpenAI does NOT send OpenRouter-required headers by default.
        # Without these, OpenRouter returns 401 Missing Authentication header.
        kwargs["default_headers"] = {
            "Authorization": f"Bearer {openai_api_key}",   # REQUIRED
            "HTTP-Referer": "http://localhost",            # REQUIRED by OpenRouter
            "X-Title": "Praxa Exercise"                    # REQUIRED by OpenRouter
        }

        # ❗ NOT CHANGED: keep school syntax EXACTLY the same
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs
        )

def get_model(model_name: str = "google/gemma-4-31b-it:free") -> ChatModel:
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
    model = get_model()
    from langchain_core.messages import HumanMessage, SystemMessage

    response = model.invoke(
        [SystemMessage("You are a helpful assistant."),
         HumanMessage("What are some plays by Tawfiq al-Hakim?")])
    print(response.content)
    print("----------")

    response = model.invoke(
        [SystemMessage("You are a helpful assistant."),
         HumanMessage("What is Ryan Calais Camerons's most recent play?")])
    print(response.content)
    print("----------")

    response = model.invoke(
        [SystemMessage("You are a helpful assistant."),
         HumanMessage("What Broadway shows have more than 10,000 performances?")])
    print(response.content)

    pass
