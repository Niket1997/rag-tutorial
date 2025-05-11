# The code in this folder implements the most basic RAG. 

### You can find the tutorial in [this](https://blogs.niket.pro/implementing-rag) article. 

## Set Up 

### Python

Make sure you have Python installed locally, preferably the latest version.

### OpenAI

You need to create an account in OpenAI & generate an API key for testing. We will be storing this API key in .env file to be used in the code. You can refer to this short YouTube video to know how to generate OpenAI API key.

### Clone GitHub Repository

GitHub Repository: https://github.com/Niket1997/rag-tutorial

### Install Dependencies

You also need to install the required dependencies. Open the cloned repository in the IDE of your choice & run the following command to install dependencies.

#### installing uv on mac
```bash
brew install uv 
```

#### install dependencies
```bash 
uv pip install . 
# or alternatively, uv pip install -r pyproject.toml
```

### Install Docker

We will be using Docker to set up the vector database qdrant locally, hence you need to install Docker in your machine. Just Google it.

### Run qdrant locally using Docker
```bash
docker compose up -d -f docker-compose.yml
```


### Create .env file

Create a new file in the cloned repository with the name .env & and following contents to it.
```bash
OPENAI_API_KEY="<your-openai-api-key>"
QDRANT_URL="http://localhost:6333"
```
As mentioned in the previous article, a RAG system has two phases, ingestion phase & query phase. Let’s code them one by one.

💡
We will be using LangChain framework in this tutorial to build our basic RAG. LangChain is widely used open source framework for building applications on top of Large Language Models (LLMs). You can read more about LangChain here.