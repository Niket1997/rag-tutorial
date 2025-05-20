import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import Distance, VectorParams
from models import MultipleQueries

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

# llm for query generation
llm_for_query_gen = llm.with_structured_output(MultipleQueries)


### qdrant related utils
# create qrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
)


# create a collection if it doesn't exist
def create_collection_if_not_exists(collection_name: str):
    # check if collection exists
    if not collection_exists(collection_name):
        # create the collection if it doesn't exist
        # Note, here the dimensions 1536 is corresponding to the embedding model we chose
        # which is text-embedding-3-small
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
