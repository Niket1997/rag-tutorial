from utils import llm_for_query_gen, llm, get_vector_store, collection_exists
from models import MultipleQueries
from constants import (
    SYSTEM_PROMPT_QUERY_GEN,
    COLLECTION_NAME,
    SIMILARITY_THRESHOLD,
    SYSTEM_PROMPT_ANSWER_GEN,
)
from langchain_core.documents import Document


# retrieve answer for user's query
def retrieve_answer(query: str) -> str:
    # 1. use LLM to generate 3 queries similar to the user's query
    queries = generate_queries(query)

    # 2. fetch the relevant documents for all the queries from vector store
    docs = aggregate_relevant_documents(queries + [query])

    # 3. use LLM to generate the answer for the user's query based on the relevant documents
    answer = generate_answer(query, docs)
    return answer


# generate 3 queries similar to the user's query
def generate_queries(query: str) -> list[str]:
    # 1. use LLM to generate 3 queries similar to the user's query
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_QUERY_GEN},
        {"role": "user", "content": query},
    ]

    response = llm_for_query_gen.invoke(messages)
    if isinstance(response, MultipleQueries):
        result = response.queries
        print(f"🌀🌀🌀 Generated {len(result)} queries")
        for i, query in enumerate(result):
            print(f"🌀🌀🌀 {i+1}. {query}")
        return result
    else:
        raise ValueError("Invalid response from LLM")


# aggregate the relevant documents
def aggregate_relevant_documents(queries: list[str]) -> list[Document]:
    # 1. fetch the relevant documents for each query
    docs = [fetch_relevant_documents_for_query(query) for query in queries]

    # 2. flatten the list of lists and get unique documents
    flattened_docs = [doc for sublist in docs for doc in sublist]
    unique_docs = list({doc.page_content: doc for doc in flattened_docs}.values())

    print(f"🌀🌀🌀 Found {len(unique_docs)} unique documents across all the queries")

    return unique_docs


# fetch the relevant documents for the query
def fetch_relevant_documents_for_query(query: str) -> list[Document]:
    # 1. check if collection exists
    if not collection_exists(COLLECTION_NAME):
        raise ValueError("Collection does not exist")

    # 2. fetch the relevant documents
    vector_store = get_vector_store(COLLECTION_NAME)

    # 3. fetch the relevant documents
    docs = vector_store.similarity_search_with_score(query, k=5)

    # 4. filter the documents based on the similarity threshold
    filtered_docs = [doc for doc, score in docs if score >= SIMILARITY_THRESHOLD]

    print(f"🌀🌀🌀 QUERY: {query}. FOUND: {len(filtered_docs)} documents")

    return filtered_docs


# generate the answer for the user's query
def generate_answer(query: str, docs: list[Document]) -> str:
    # 1. use LLM to generate the answer for the user's query based on the relevant documents
    system_prompt = SYSTEM_PROMPT_ANSWER_GEN
    for doc in docs:
        system_prompt += f"""
        Document: {doc.page_content}
        """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    response = llm.invoke(messages)
    return response.content
