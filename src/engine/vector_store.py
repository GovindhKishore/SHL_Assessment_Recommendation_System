import pandas as pd
import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions
import os

# Paths to data files and database
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "shl_assessments.csv")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings", "chroma_db")
COLLECTION_NAME = "shl_assessments"


def initialize_vector_store():
    """
    Initialize the vector database with assessment data.

    Loads assessment data from CSV, creates embeddings using sentence transformers,
    and stores them in a ChromaDB collection for semantic search.
    """
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"file not found at {DATA_PATH}")
        return

    # Load and clean data
    df = pd.read_csv(DATA_PATH).fillna("")
    print(f"Loaded {len(df)} assessments from CSV. (Preparing to index...)")

    # Create focused text representation for better semantic search
    def create_focused_text(row):
        """
        Create a focused text representation of an assessment for embedding.

        Structures the text to emphasize name and type while including a 
        shortened description for better semantic matching.
        """
        name = str(row['name'])
        test_type = str(','.join(row['test_type']))
        desc = str(row['description'])

        # Use only the first part of the description to avoid noise
        short_desc = desc[:200]

        # Format the text with clear structure
        return f"Name: {name}. Type: {test_type}. Description: {short_desc}"

    # Apply the text transformation to create combined text field
    df["combined_text"] = df.apply(create_focused_text, axis=1)

    # Initialize ChromaDB client and embedding function
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if it exists to ensure clean data
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted old collection to ensure clean rebuild.")
    except NotFoundError:
        pass

    # Create a new collection with the embedding function
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_func
    )

    # Prepare data for indexing
    ids = [str(i) for i in range(len(df))]
    documents = df['combined_text'].tolist()

    # Prepare metadata for each assessment
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "name": str(row['name']),
            "url": str(row['url']),
            "description": str(row['description']),  # Store full description in metadata
            "duration": str(row['duration']),
            "test_type": str(row['test_type']),
            "remote_support": str(row['remote_support']),
            "adaptive_support": str(row['adaptive_support'])
        })

    # Generate embeddings and add to collection
    print("Generating embeddings and indexing... (This may take a while)")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Success! Indexed {collection.count()} items in {DB_PATH}")


if __name__ == "__main__":
    initialize_vector_store()
