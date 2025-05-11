from utils import get_collection_name, get_vector_store, collection_exists, llm

# similarity threshold
SIMILARITY_THRESHOLD = 0.7


# retrieve the answer
def retrieve_answer(query: str, file_name: str) -> str:
    system_prompt = """
    You are a helpful AI assistant that can answer user's questions based on the documents provided.
    If there aren't any related documents, or if the user's query is not related to the documents, then you can provide the answer based on your knowledge.        Think carefully before answering the user's question.
    """
    # get the vector embeddings assocoated with that query
    try:
        collection_name = get_collection_name(file_name)
        if collection_exists(collection_name):
            vector_store = get_vector_store(collection_name)

            # Get documents with their similarity scores
            docs = vector_store.similarity_search_with_score(query, k=5)

            for doc, score in docs:
                if score >= SIMILARITY_THRESHOLD:
                    system_prompt += f"""
                    Document: {doc.page_content}
                    """

            print(
                f"Found {len(docs)} documents with similarity score >= {SIMILARITY_THRESHOLD}"
            )

        messages = [("system", system_prompt), ("user", query)]

        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"An error occurred: {e}"
