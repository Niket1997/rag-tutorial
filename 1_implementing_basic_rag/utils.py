import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import Distance, VectorParams

# load the environment variables
load_dotenv()


### openai related utils
# initialize the embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# create LLM
llm = ChatOpenAI(
    model="gpt-4.1",
)


### qdrant related utils
# create qrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
)


# create a collection if it doesn't exist
def create_collection_if_not_exists(collection_name: str):
    # check if collection exists
    if not qdrant_client.collection_exists(collection_name):
        # create the collection if it doesn't exist
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"Collection {collection_name} created")
    else:
        print(f"Collection {collection_name} already exists")


# check if collection exists
def collection_exists(collection_name: str):
    return qdrant_client.collection_exists(collection_name)


# get the qdrant vector store for collection
def get_vector_store(collection_name: str):
    return QdrantVectorStore(
        collection_name=collection_name,
        client=qdrant_client,
        embedding=embeddings,
    )


### langchain related utils
# get the text splitter
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )


### general utils
# get the collection name
def get_collection_name(file_name: str):
    return f"rag_collection_{file_name.split('/')[-1].split('.')[0]}"
