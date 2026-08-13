import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = "http://localhost:6335"
COLLECTION_NAME = "accurate_docs"

def get_vectorstore():
    client = QdrantClient(url=QDRANT_URL)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Ensure collection exists
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vector_store

def ingest_parsed_pages(parsed_pages):
    """
    Takes a list of dictionaries with 'page' and 'text', chunks the text,
    and stores it in the Qdrant vector database.
    """
    from tenacity import retry, wait_exponential, stop_after_attempt

    vector_store = get_vectorstore()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    docs = []
    for item in parsed_pages:
        page_num = item["page"]
        text = item["text"]
        
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"page": page_num}))
            
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def add_docs_with_retry(vs, documents):
        vs.add_documents(documents)

    if docs:
        add_docs_with_retry(vector_store, docs)
        
    return len(docs)
