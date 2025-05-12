from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG


# Demo database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")

driver = GraphDatabase.driver(URI, auth=AUTH)
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
retriever = HybridRetriever(
    driver=driver,
    vector_index_name="moviePlotsEmbedding",
    fulltext_index_name="movieFulltext",
    embedder=embedder,
    return_properties=["title", "plot"],
)
query_text = "What is the name of the movie set in 1375 in Imperial China?"

# W/O. LLM
# retriever_result = retriever.search(query_text=query_text, top_k=3)
# print(retriever_result)

# W/. LLM
# Note: the OPENAI_API_KEY must be in the env vars
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# Initialize the RAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Query the graph
query_text = "What is the name of the movie set in 1375 in Imperial China?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 3})
print(response.answer)
