from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# Demo database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
retrieval_query = """
MATCH
(actor:Actor)-[:ACTED_IN]->(node)
RETURN
node.title AS movie_title,
node.plot AS movie_plot,
collect(actor.name) AS actors;
"""
vc_retriever = VectorCypherRetriever(
    driver,
    index_name="moviePlotsEmbedding",
    embedder=embedder,
    # return_properties=["title", "plot"],
    retrieval_query=retrieval_query,
)

# LLM
# Note: the OPENAI_API_KEY must be in the env vars
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# Initialize the RAG pipeline
rag = GraphRAG(retriever=vc_retriever, llm=llm)

# Query the graph
query_text = "who directed a movie set in 1375 in Imperial China?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 3})
print(response.answer)
