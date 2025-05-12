from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

# Demo database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
retriever = VectorRetriever(
    driver,
    index_name="moviePlotsEmbedding",
    embedder=embedder,
    return_properties=["title", "plot"],
)


query_text = "A movie about the famous sinking of the Titanic"
retriever_result = retriever.search(query_text=query_text, top_k=5)
# print(retriever_result)
items = retriever_result.items
retriever_result = [item.content for item in items]
print("Retriever Result: ", retriever_result)