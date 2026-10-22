import gdown
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Iterable, Optional

def download_context_data(pdfs: Iterable[dict[str, str]], path: str = "./context_data") -> None:
    """
    Downloads PDFs and stores them in local storage.
    
    :param pdfs: an iterable of dictionaries. each dictionary must have
                 keys of "url" with the URL of the PDF and "filename"
                 with the name to store the file as.
    :type pdfs: Iterable[dict[str,str]]
    :param path: location to store the files (default is "./context_data")
    :type path: str
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        gdown.download(pdf["url"], f"{path}/{pdf['filename']}", quiet=True)

def load_context_data(path: str = "./context_data") -> list[Document]:
    """
    Loads multiple PDFs into LangChain Document objects.
    
    :param path: location of the files (default: "./context_data")
    :type path: str
    :return: list of Document objects
    :rtype: list[Document]
    """
    loader = PyPDFDirectoryLoader(path)
    return loader.load()

def chunk_context_data(context_data: list[Document]) -> list[Document]:
    """
    Split the context data into overlapping chunks
    
    :param context_data: the Documents to split into chunks
    :type context_data: list[Document]
    :return: the chunked Documents
    :rtype: list[Document]
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    return text_splitter.split_documents(context_data)

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Gets an embedding model for vectorizing the context data.

    :param model_name: The name of the embedding model to get (default is sentence-transformers/all-MiniLM-L6-v2)
    :type model_name: str
    :return: The embedding model.
    :rtype: HuggingFaceEmbeddings
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vector_store(chunks: list[Document], embedding_model: Optional[Embeddings] = None, path: str = "./chromadb") -> Chroma:
    """
    Create a persistent vector store from a list of chunked documents.
    
    :param chunks: The context data. If not specified, create an empty vector store.
    :type chunks: list[Document]
    :param embedding_model: The embedding model to use (default: get_embedding_model(), loaded when this function is called)
    :type embedding_model: Optional[Embeddings]
    :param path: Path to the vector store
    :type path: str
    :return: The vector store
    :rtype: Chroma
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()  # loaded on call, not when context.py is imported
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=path
    )
    
def get_vector_store(embedding_model: Optional[Embeddings] = None, path: str = "./chromadb") -> Chroma:
    """
    Gets a persistent vector store.
    
    :param embedding_model: The embedding model to use (default: get_embedding_model(), loaded when this function is called)
    :type embedding_model: Optional[Embeddings]
    :param path: path to the vector store
    :type path: str
    :return: The vector store
    :rtype: Chroma
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()  # loaded on call, not when context.py is imported
    return Chroma(
        persist_directory=path,
        embedding_function=embedding_model
    )

if __name__ == "__main__":
    # when run as a script, run some tests to demonstrate capabilities
    pdfs = (
        { "url": "https://quanticedu.github.io/PraxaNew/Longest Running Shows on Broadway 2025.pdf",
          "filename": "Longest Running Shows on Broadway.pdf" },
        { "url": "https://quanticedu.github.io/PraxaNew/Every play and musical coming to the West End in 2025.pdf",
          "filename": "Every play and musical coming to the West End in 2025.pdf" }
    )
    download_context_data(pdfs)
    context_data = load_context_data()
#    chunks = chunk_context_data(context_data)
#    embedding_model = get_embedding_model()
#    vector_store = create_vector_store(chunks, embedding_model)

    for page in context_data:
        print(page)

#    for num, chunk in enumerate(chunks):
#        print("-----")
#        print(f"Chunk {num}:")
#        print(f"Length: {len(chunk.page_content)}")
#        print(f"Metadata: {chunk.metadata}")
#        print(f"Content: {chunk.page_content}")

#    embedding = embedding_model.embed_query("This is a test sentence.")
#    print(f"Embedding length: {len(embedding)}")
#    embedding = embedding_model.embed_query("This is a longer test sentence.")
#    print(f"Embedding length: {len(embedding)}")
    
#    retrieved_chunks = vector_store.similarity_search("A play written by Ryan Calais Cameron.")
#    print(f"Query retrieved {len(retrieved_chunks)} chunks.")

#    for chunk in retrieved_chunks:
#        print(f"Chunk content: {chunk.page_content}")
#        print(f"Chunk metadata: {chunk.metadata}")
#        print("-----")

    pass
