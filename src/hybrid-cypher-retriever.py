from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j+s://demo.neo4jlabs.com", auth=("recommendations", "recommendations"))
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
retrieval_query = """
MATCH
(actor:Actor)-[:ACTED_IN]->(node)
RETURN
node.title AS movie_title,
node.plot AS movie_plot, 
collect(actor.name) AS actors;
"""
retriever = HybridCypherRetriever(
    driver,
    vector_index_name="moviePlotsEmbedding",
    fulltext_index_name="movieFulltext",
    retrieval_query=retrieval_query,
    embedder=embedder,
)
query_text = "What are the names of the actors in the movie set in 1375 in Imperial China?"
retriever_result = retriever.search(query_text=query_text, top_k=3)
print(retriever_result)