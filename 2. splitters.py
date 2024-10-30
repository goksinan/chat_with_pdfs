from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader


some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""


def example_character_split():
    splitter = CharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separator=" "
    )
    print(splitter.split_text(some_text))


def example_recursive_split():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\.)", " ", ""]
    )
    print(splitter.split_text(some_text))


def example_token_split():
    splitter = TokenTextSplitter(
        chunk_size=10,
        chunk_overlap=0
    )
    print(splitter.split_text(some_text))


def example_pdf_doc():
    loader = PyPDFLoader("./What Every Programmer Should Know About Memory.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(pages)
    print("Length pages:", len(pages))
    print("Length splits:", len(docs))
    # Note that chunks carry over the document metada
    print("Page metada:", pages[0].metadata)
    print("Split metada:", docs[0].metadata)


if __name__ == '__main__':
    # example_character_split()
    # example_recursive_split()
    # example_token_split()
    example_pdf_doc()