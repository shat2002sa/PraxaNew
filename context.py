import gdown
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Iterable

def download_context_data(pdfs: Iterable[dict[str, str]], path: str = "./context_data") -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        gdown.download(pdf["url"], f"{path}/{pdf['filename']}", quiet=True)

def load_context_data(path: str = "./context_data") -> list[Document]:
    loader = PyPDFDirectoryLoader(path)
    return loader.load()

def chunk_context_data(context_data: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(context_data)

def get_embedding_model(model_name: str = "intfloat/e5-small-v2") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vector_store(chunks: list[Document], embedding_model: Embeddings = get_embedding_model(), path: str = "./chromadb") -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=path
    )
    
def get_vector_store(embedding_model: Embeddings = get_embedding_model(), path: str = "./chromadb") -> Chroma:
    return Chroma(
        persist_directory=path,
        embedding_function=embedding_model
    )

if __name__ == "__main__":
     context_data = load_context_data()
     chunks = chunk_context_data(context_data)
     embedding_model = get_embedding_model()
     vector_store = create_vector_store(chunks, embedding_model)

     retrieved_chunks = vector_store.similarity_search("A play written by Ryan Calais Cameron.")
     print(f"Query retrieved {len(retrieved_chunks)} chunks.")

     for chunk in retrieved_chunks:
         print(f"Chunk content: {chunk.page_content}")
         print(f"Chunk metadata: {chunk.metadata}")
         print("-----")

pass
