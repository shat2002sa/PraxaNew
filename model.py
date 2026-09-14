from langchain_community.chat_models import ChatOpenAI
from typing import Optional, Any
import os

# ❌ REMOVED (per instructor): do NOT set env vars inside code
# os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

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

        # ✅ FIX #1: Load the key normally, but do NOT assign it back into os.environ
        openai_api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")

        # ✅ FIX #3: Fail early if the key is missing
        if not openai_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Export it in your terminal before running: "
                "export OPENROUTER_API_KEY='your_key_here'"
            )

        # ✅ FIX #2: Add ONLY the extra OpenRouter headers (do NOT override Authorization)
        # We MERGE headers instead of replacing them.
        extra_headers = {
            "HTTP-Referer": "http://localhost",
            "X-Title": "Praxa Exercise"
        }

        # Merge with any existing headers passed in kwargs
        existing_headers = kwargs.get("default_headers", {})
        existing_headers.update(extra_headers)
        kwargs["default_headers"] = existing_headers

        # ❗ School syntax preserved exactly
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs
        )

def get_model(model_name: str = "openrouter/free") -> ChatModel:
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
