from langchain_community.document_loaders import PyPDFLoader
from utils import (
    get_text_splitter,
    create_collection_if_not_exists,
    get_vector_store,
    get_collection_name,
)


# ingest the pdf
def ingest_pdf(pdf_path: str):
    # 1. load the pdf
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. split the docs into chunks
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_documents(docs)

    # 3. create a collection if it doesn't exist
    collection_name = get_collection_name(pdf_path)
    create_collection_if_not_exists(collection_name=collection_name)

    # 4. get qdrant vector store
    # this will create a vector store & assign the OpenAI embeddings to it
    vector_store = get_vector_store(collection_name=collection_name)

    # 5. add the chunks to the vector store
    # this will generate the embeddings for the chunks & add them to the vector store
    vector_store.add_documents(documents=chunks)

    print(f"Ingested {len(chunks)} chunks from {pdf_path} into {collection_name}")
