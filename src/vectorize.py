import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

"""
Vectorize documents into the Chromadb Docker image with the help of the HTTPClient. 
Using a client (pulled docker image) doesn't need to persist the database locally. 
"""

DATA_PATH = "data/"
# DB_FAISS_LOCAL_PATH = "../vectorstores/db_faiss"
# DB_CHROMA_LOCAL_PATH = "../vectorstores/db_chroma"

def create_documents(raw_documents_path: str = "../data/"):
    # Create vector database
    loader = DirectoryLoader(raw_documents_path, glob='*.pdf', loader_cls=PyPDFLoader)
    file = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(file)
    return documents


def vectorize_documents(chroma_client, embedding_function, raw_documents_path:str = "../data/", collection_name:str = 'medical_collection', distance_function:str = 'l2'):

    documents = create_documents(raw_documents_path)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function = embedding_function,    # Chroma will use sentence transformer as a default.
        metadata={"hnsw:space": distance_function}  # Valid options for hnsw:space are "l2", "ip, "or "cosine"
    )

    # db.add_documents(documents) -> This is not working with my embedding function (https://github.com/langchain-ai/langchain/commit/564871cded9f1f371f0f266815840326c7e0087d)

    for doc in documents:   # we can't add Document object directly yet
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
        )

    return collection