from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
import context, model

prompt_template = ChatPromptTemplate([
    ("system", "You are an assistant providing answers to questions about the theater. In addition to your training data, use the additional context provided below to provide up-to-date information."),
    ("human", "Question: {question}\nContext: {context}\nAnswer:")
])

retriever = context.get_vector_store().as_retriever()

question_and_docs = RunnableParallel(
    { "question": RunnablePassthrough(),
      "context_docs": retriever }
)

def make_context_string(dict_with_docs: dict[str, Document]) -> str:
    """
    Takes the contents of each Document object in a dictionary and joins them
    in one string, separated by two newlines
    
    :param dict_with_docs: The dictionary with the context docs under the key
                           "context_docs"
    :type dict_with_docs: dict[str, Document]
    :returns: The combined string
    :rtype: str
    """
    return "\n\n".join(doc.page_content for doc in dict_with_docs["context_docs"])

context = RunnablePassthrough.assign(context=make_context_string)
model = model.get_model()
answer_chain = context | prompt_template | model
chain_with_sources = question_and_docs.assign(answer=answer_chain)

def answer_and_sources(question: str) -> dict[str, str]:
    """
    Invokes the model with the given question.
    
    :param question: The question to ask.
    :returns: Dictionary with the answer and supporting sources
    """
    result = chain_with_sources.invoke(question)
    response_text = result["answer"].content
    sources = "\n\n".join(f"{doc.metadata['source']}, page {doc.metadata['page']}" for doc in result["context_docs"])
    return {"answer": response_text,
            "sources": sources}

if __name__ == "__main__":
# when run as a script, run some tests to demonstrate capabilities
#    docs = retriever.invoke("What is Ryan Calais Cameron's most recent play?")
#    print(f"Found {len(docs)} documents:")

#    for doc in docs:
#        print("-----")
#        print(doc)

#    print(question_and_docs.invoke("What is Ryan Calais Cameron's most recent play?"))

#    my_dict = {
#        "question": "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
#        "answer": "All the wood that a woodchuck could chuck if a woodchuck could chuck wood."
#    }

#    add_length = RunnablePassthrough.assign(length=len)
#    print(type(add_length))
#    print(add_length.invoke(my_dict))

#    complete_prompt_chain = question_and_docs | context | prompt_template
#    result = complete_prompt_chain.invoke("What is Ryan Calais Cameron's most recent play?")
#    print(type(result))
#    print(result)

#    chain = question_and_docs | context | prompt_template | model
#    result = chain.invoke("What is Ryan Calais Cameron's most recent play?")
#    print(result.content)

#    result = chain_with_sources.invoke("What Broadway shows have had more than 10,000 performances?")
#    print("The docs used in this answer:")
#    print("\n".join(doc.metadata.__repr__() for doc in result["context_docs"]))
#    print("-----")
#    print("The answer:")
#    print(result["answer"].content)

#    print(answer_and_sources("What is Ryan Calais Cameron's most recent play?"))

    pass