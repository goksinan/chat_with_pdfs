from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import os
import shutil


def example_embeddings():
    # Create embedding object
    embedding = OpenAIEmbeddings()
    # Three sentences to compare
    sentence1 = "i like dogs"
    sentence2 = "i like cats"
    sentence3 = "it is very cold outside"
    # Create embeddings
    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)
    # Check
    print("Similarity 1-2:", np.dot(embedding1, embedding2))
    print("Similarity 1-3:", np.dot(embedding1, embedding3))
    print("Similarity 2-3:", np.dot(embedding2, embedding3))


def example_vector_database():
    sources = [
        "docs/cs229_lectures/MachineLearning-Lecture01.pdf",
        "docs/cs229_lectures/MachineLearning-Lecture01.pdf",
        "docs/cs229_lectures/MachineLearning-Lecture02.pdf",
        "docs/cs229_lectures/MachineLearning-Lecture03.pdf",
    ]
    docs = []
    for source in sources:
        loader = PyPDFLoader(source)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    print("Len. splits:", len(splits))
    embedding = OpenAIEmbeddings()
    persist_directory = "docs/chroma"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)
    try:
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory,
        )
        count = vectordb._collection.count()
        print(f"Successfully created vector database with {count} documents")
    except Exception as e:
        print(f"Error using Chroma DB: {str(e)}")
    # Retrieve chunks
    question = "is there an email i can ask for help"
    out = vectordb.similarity_search(question, k=3)
    print(out[0].page_content)
    # Failure mode (duplicate data)
    question = "what did they say about matlab?"
    out = vectordb.similarity_search(question, k=3)
    print("\n")
    print(out[0].page_content[:100])
    print(out[1].page_content[:100])
    # Failure mode (disregarding metadata)
    question = "what did they say about regression in the third lecture?"
    out = vectordb.similarity_search(question, k=5)
    print("\n")
    for item in out:
        print(item.metadata)


def load_existing_database():
    embedding = OpenAIEmbeddings()
    persist_directory = "docs/chroma"
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    count = vectordb._collection.count()
    print(f"Successfully loaded vector database with {count} documents")


if __name__ == '__main__':
    # example_embeddings()
    example_vector_database()
    # load_existing_database()