from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

# embdedder = OpenAIEmbeddings(model="text-embedding-ada-002")
driver = GraphDatabase.driver("neo4j+s://demo.neo4jlabs.com", auth=("recommendations", "recommendations"))

# Create LLM object
t2c_llm = OpenAILLM(model_name="gpt-3.5-turbo")

# (Optional) Specify your own Neo4j schema
neo4j_schema = """
  Node properties:
  Person {name: STRING, born: INTEGER}
  Movie {tagline: STRING, title: STRING, released: INTEGER}
  Relationship properties:
  ACTED_IN {roles: LIST}
  REVIEWED {summary: STRING, rating: INTEGER}
  The relationships:
  (:Person)-[:ACTED_IN]->(:Movie)
  (:Person)-[:DIRECTED]->(:Movie)
  (:Person)-[:PRODUCED]->(:Movie)
  (:Person)-[:WROTE]->(:Movie)
  (:Person)-[:FOLLOWS]->(:Person)
  (:Person)-[:REVIEWED]->(:Movie)
"""

# (Optional) Provide user input/query pairs for the LLM to use as examples
examples = [
  "USER INPUT: 'Which actors starred in the Matrix?' QUERY: MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title = 'The Matrix' RETURN p.name"
]

# Initialize the retriever
retriever = Text2CypherRetriever(
  driver=driver,
  llm=t2c_llm,
  neo4j_schema=neo4j_schema,
  examples=examples,
)

query_text = "Which movies did Hugo Weaving star in?"
print(retriever.search(query_text=query_text))
