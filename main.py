import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders import PyPDFLoader

doc_pth = r"C:\Users\goksi\Dendron\vault\assets\pdfs\next\Face\FAS\patches\A simple and effective patch-Based method for frame-level face anti-spoofing.pdf"
loader = PyPDFLoader(doc_pth)
pages = loader.load()
print(len(pages))
page = pages[0]
print(page.page_content[:500])
print(page.metadata)
