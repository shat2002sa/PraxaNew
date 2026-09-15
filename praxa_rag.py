from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
import context as context_module, model as model_module

prompt_template = ChatPromptTemplate([
    ("system", "You are an assistant providing answers to questions about the theater. In addition to your training data, use the additional context provided below to provide up-to-date information."),
    ("human", "Question: {question}\nContext: {context}\nAnswer:")
])

retriever = context_module.get_vector_store().as_retriever()

question_and_docs = RunnableParallel(
    { "question": RunnablePassthrough(),
      "context_docs": retriever }
)

def make_context_string(dict_with_docs: dict[str, Document]) -> str:
    return "\n\n".join(doc.page_content for doc in dict_with_docs["context_docs"])

context_runnable = RunnablePassthrough.assign(context=make_context_string)
llm = model_module.get_model()
answer_chain = context_runnable | prompt_template | llm
chain_with_sources = question_and_docs.assign(answer=answer_chain)

def answer_and_sources(question: str) -> dict[str, str]:
    result = chain_with_sources.invoke(question)
    response_text = result["answer"].content
    sources = "\n\n".join(f"{doc.metadata['source']}, page {doc.metadata['page']}" for doc in result["context_docs"])
    return {"answer": response_text, "sources": sources}

if __name__ == "__main__":
    result = chain_with_sources.invoke("What Broadway shows have had more than 10,000 performances?")
    print("The docs used in this answer:")
    print("\n".join(doc.metadata.__repr__() for doc in result["context_docs"]))
    print("-----")
    print("The answer:")
    print(result["answer"].content)

    pass
