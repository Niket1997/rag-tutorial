from pydantic import BaseModel


# model for multiple queries
class MultipleQueries(BaseModel):
    queries: list[str]
