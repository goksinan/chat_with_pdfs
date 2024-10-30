from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def example_simple_retrieval():
    texts = [
        """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
        """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
        """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    ]
    embedding = OpenAIEmbeddings()
    smalldb = Chroma.from_texts(texts, embedding=embedding)
    question = "Tell me about all-white mushrooms with large fruiting bodies"
    # Semantic search doesn't care about diversity. Just cares about similarity
    out = smalldb.similarity_search(question, k=2)
    print(out)
    # MMR also cares about diversity
    out = smalldb.max_marginal_relevance_search(question, k=2, fetch_k=3)
    print(out)


def example_retrieval():
    vectordb = load_existing_database()
    question = "what did they say about matlab?"
    out = vectordb.similarity_search(question, k=3)
    print("\nSimilarity search:")  # most similar
    print(out[0].page_content[:100])
    print(out[1].page_content[:100])
    out = vectordb.max_marginal_relevance_search(question, k=3)
    print("\nRelevancy search:")  # more diversity
    print(out[0].page_content[:100])
    print(out[1].page_content[:100])


def example_self_query_retrieval():
    vectordb = load_existing_database()
    question = "what did they say about regression in the third lecture?"
    # We have to manually provide a description to filter relevant docs based on metadata 
    out = vectordb.similarity_search(
        question,
        k=3,
        filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
    )
    print("\nManual:")
    for item in out:
        print(item.metadata)
    # However, we can't manually enter descriptions everytime we have a question.
    # We need to infer the metadata from the query itself.
    # We can use LLMs to automatically create 1. the query 2. the filter description so that we can pass them to the retriever.
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the lecture",
            type="integer",
        ),
    ]
    document_content_description = "Lecture notes"
    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True
    )
    out = retriever.invoke(question)
    print("\nAutomatic:")
    for item in out:
        print(item.metadata)


def example_compression_retrieval():
    def pretty_print_docs(docs):
        print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

    vectordb = load_existing_database()
    # Wrap our vectorstore
    llm = llm = ChatOpenAI(model='gpt-4o', temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever(search_type="mmr")
    )
    question = "what did they say about matlab?"
    compressed_docs = compression_retriever.invoke(question, k=5)
    pretty_print_docs(compressed_docs)


def load_existing_database():
    embedding = OpenAIEmbeddings()
    persist_directory = "docs/chroma"
    try:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
        count = vectordb._collection.count()
        print(f"Successfully loaded vector database with {count} documents")
        return vectordb
    except Exception as e:
        print(f"Error using Chroma DB: {str(e)}")


if __name__ == '__main__':
    # example_simple_retrieval()
    # example_retrieval()
    # example_self_query_retrieval()
    example_compression_retrieval()