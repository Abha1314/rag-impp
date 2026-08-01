print("APP FILE IS RUNNING")

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

# Load documents
docs = load_all_documents("data")

# Create vector store
vector_store = FaissVectorStore()

# Build FAISS index
vector_store.build_from_documents(docs)

# Save
vector_store.save()

# Load
vector_store.load()

# Query
result = vector_store.query(
    query="What is Machine Learning?",
    top_k=3
)

print(result["context"])

for doc in result["documents"]:
    print("=" * 80)
    print("Rank:", doc["rank"])
    print("Similarity:", round(doc["similarity_score"], 4))
    print("Source:", doc["metadata"])
    print(doc["content"][:300])
