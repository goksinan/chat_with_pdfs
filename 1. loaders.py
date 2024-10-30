from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders import WebBaseLoader


def example_pdf_loader():
    # Create loader object
    loader = PyPDFLoader("What Every Programmer Should Know About Memory.pdf")
    # Load document
    pages = loader.load()
    # See how many pages the document has
    print("Number of pages:", len(pages))
    # Print first 1000 characters from the first page
    page = pages[0]
    print(page.page_content[:1000])
    # Print the metadata of the first page
    print(page.metadata)


def example_youtube_loader():
    """
    YoutubeAudioLoader: loads the audio file from a youtube video
    OpenAIWhisperParser: uses OpenAI's Whisper model (speech detection) to convert audio into text
    """
    url = "https://youtu.be/HOAzOKTHZFg?si=rLBLZcZikTi0EsDF"
    save_dir = "docs/youtube"
    loader = GenericLoader(
        YoutubeAudioLoader(urls=[url], save_dir=save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    print(docs[0].page_content[:1000])


def example_web_loader():
    url = "https://www.anthropic.com/news/claude-3-5-sonnet"
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(docs[0].page_content[:1000])


if __name__ == '__main__':
    # example_pdf_loader()
    # example_youtube_loader()
    example_web_loader()