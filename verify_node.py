
import asyncio
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

async def verify_node():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        query = """
        MATCH (n:Entity)
        WHERE n.id CONTAINS 'simple_to_implement' OR n.name CONTAINS 'simple_to_implement'
        RETURN n
        """
        print(f"Running query on {URI}...")
        records, summary, keys = driver.execute_query(query)
        
        if records:
            print(f"Found {len(records)} node(s):")
            for record in records:
                node = record["n"]
                print(f" - ID: {node.get('id')}")
                print(f" - Name: {node.get('name')}")
                print(f" - Labels: {node.labels}")
                print(f" - Properties: {dict(node)}")
        else:
            print("No node found matching 'simple_to_implement'")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    asyncio.run(verify_node())
