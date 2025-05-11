from ingestion import ingest_pdf
from query import retrieve_answer

# pdf path
pdf_path = "/Users/aniket.mahangare/myProjects/rag-tutorial/1_implementing_basic_rag/alices-adventures-in-wonderland.pdf"


# ingest the pdf
def ingest():
    print("Ingesting the pdf about Alice's Adventures in Wonderland...")
    ingest_pdf(pdf_path=pdf_path)


# query the pdf
def query_cli():
    print("\n\n🤖: Welcome to the RAG CLI! Type 'exit' to quit.\n")
    while True:
        query = input("> ")
        if query.lower() == "exit":
            print("🤖: Exiting the RAG CLI...")
            break
        answer = retrieve_answer(query=query, file_name=pdf_path)
        print(f"🤖: {answer}")


if __name__ == "__main__":
    ingest()
    query_cli()
