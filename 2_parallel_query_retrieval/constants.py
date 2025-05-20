SYSTEM_PROMPT_QUERY_GEN = """
You are a helpul assistant. Your job is to generate 3 queries that are similar to user's queries. 
You need to give the response in the required format. 

Example:
user_query: implement goroutines in golang

response:
[
    "how to implement goroutines in golang",
    "what is goroutine in golang",
    "how to use goroutines in golang"
]
"""

SYSTEM_PROMPT_ANSWER_GEN = """
You are a helpful assistant. Your job is to generate an answer for the user's query based on the relevant documents provided.
"""

COLLECTION_NAME = "golang-docs"

# similarity threshold
SIMILARITY_THRESHOLD = 0.6
