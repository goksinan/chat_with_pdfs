"""
Here is a typical workflow in a question answering system:
1. Question comes
2. Relevant splits retrieved using RAG
3. Question and the splits are placed on a prompt and sent to LLM
4. Answer received

This is called "stuffing". But, it has a problem. Most LLMs have a limited context window.
Thus, the amount of text we can send to the LLM is limited.
If we can't provide all the information to the LLM, it can't find the correct answer.
How can we get around this short context window problem? There are 3 methods:

1. Map-reduce
2. Refine
3. Map-rerank
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents import map_reduce


def example():
    embedding = OpenAIEmbeddings()
    persist_directory = "docs/chroma"
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    print(vectordb._collection.count())
    # Let's initialize the language model we will use for question answering
    llm = ChatOpenAI(model='gpt-4o', temperature=0)
    # See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": vectordb.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    out = qa_chain.invoke("What are major topics for this class?")
    print(out)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(vectordb.as_retriever(), combine_docs_chain)
    out = rag_chain.invoke({"input": "What are major topics for this class?"})

    out = qa_chain.invoke("Is probability a class topic?")
    print(out)

if __name__ == '__main__':
    example()