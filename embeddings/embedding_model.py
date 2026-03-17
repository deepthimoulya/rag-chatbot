from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print(" Embedding model loaded!")
    return embeddings