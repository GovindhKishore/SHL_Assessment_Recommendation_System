import chromadb
from chromadb.utils import embedding_functions
import os

# Path to the ChromaDB database and collection name
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings", "chroma_db")
COLLECTION_NAME = "shl_assessments"

class Retriever:
    """
    Retriever class for semantic search of SHL assessments.

    Uses ChromaDB and sentence transformers to find relevant assessments
    based on semantic similarity to the query.
    """
    def __init__(self):
        """Initialize the retriever with ChromaDB and embedding function."""
        try:
            if not os.path.exists(DB_PATH):
                raise FileNotFoundError(f"ChromaDB path not found at {DB_PATH}.")

            # Connect to the persistent ChromaDB
            self.client = chromadb.PersistentClient(DB_PATH)

            # Initialize the embedding function with a sentence transformer model
            self.embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # Get the collection with the embedding function
            try:
                self.collection = self.client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embed_func
                )
            except Exception as e:
                raise RuntimeError(f"Failed to get collection: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize retriever: {str(e)}")

    def search(self, query, n_results=15):
        """
        Search for assessments matching the query.

        Args:
            query (str): The search query
            n_results (int): Number of results to return

        Returns:
            list: Ranked list of assessment dictionaries
        """
        try:
            # Query the vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            # Format the results into a clean list of dictionaries
            cleaned_results = []
            for i in range(len(results['ids'][0])):
                item = {
                    "id": results['ids'][0][i],
                    "score": results['distances'][0][i],  # Lower distance = better match
                    "document": results['documents'][0][i],
                    **results['metadatas'][0][i]  # Include all metadata fields
                }
                cleaned_results.append(item)

            return cleaned_results

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


if __name__ == "__main__":
    # Test the retriever with a sample query
    retriever = Retriever()
    test_query = "I need a Java developer who is good at teamwork"
    print(f"\n🔍 Testing Query: '{test_query}'\n")

    matches = retriever.search(test_query)

    for m in matches:
        print(f" - [{m['score']:.4f}] {m['name']} ({m['test_type']})")
