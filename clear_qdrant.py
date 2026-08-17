import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = "http://localhost:6335"
COLLECTION_NAME = "accurate_docs"

client = QdrantClient(url=QDRANT_URL)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' successfully deleted!")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")

