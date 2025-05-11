from ingestion import ingest_pdf
from query import retrieve_answer

# pdf path
pdf_path = "/Users/aniket.mahangare/myProjects/rag-tutorial/1_implementing_basic_rag/alices-adventures-in-wonderland.pdf"

# ingest the pdf
def ingest():
    print("Ingesting the pdf...")
    ingest_pdf(pdf_path=pdf_path)

def query_cli():
    while True:
        query = input("> ")
        answer = retrieve_answer(query=query, file_name=pdf_path)
        print(answer)


if __name__ == "__main__":
    # ingest()
    query_cli()