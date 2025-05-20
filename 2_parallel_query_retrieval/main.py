from ingestion import ingest_pdf
from query import retrieve_answer

# pdf path
pdf_path = "/Users/aniket.mahangare/myProjects/rag-tutorial/2_parallel_query_retrieval/go-tutorial.pdf"


# ingest the pdf
def ingest():
    print("Ingesting the pdf about Go Tutorial...")
    ingest_pdf(pdf_path=pdf_path)


# query the pdf
def query_cli():
    print("\n\n🤖: Welcome to the RAG CLI! Type 'exit' to quit.\n")
    while True:
        query = input("> ")
        if query.lower() == "exit":
            print("🤖: Exiting the RAG CLI...")
            break
        answer = retrieve_answer(query=query)
        print(f"🤖: {answer}")


if __name__ == "__main__":
    # ingest()
    query_cli()
