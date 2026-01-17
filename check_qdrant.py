
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

client = QdrantClient(":memory:")
client.create_collection("test", vectors_config=VectorParams(size=2, distance="Cosine"))
client.upsert("test", points=[
    {"id": 1, "vector": [0.1, 0.1], "payload": {"foo": "bar"}}
])

res = client.query_points("test", query=[0.1, 0.1], limit=1)
print(f"Type: {type(res)}")
print(f"Dir: {dir(res)}")
if hasattr(res, 'points'):
    print(f"Points type: {type(res.points)}")
    if res.points:
        print(f"Point 0: {res.points[0]}")
