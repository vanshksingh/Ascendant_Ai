import getpass
import os
from langchain_google_genai import GoogleGenerativeAI
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import re
import os


api_key = "AIzaSyAIsE4C0ZjwCuO0A6S7IEjszpY9MBjAgWE"
def get_task_decomposition(placeroute , query):
    # Set up the LLM which will power our application.
    llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)

    def differentiate_input(input_string):
        # Regular expression for URL detection
        url_regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        # Check if the input is a URL
        if re.match(url_regex, input_string):
            return "URL"

        # Check if the input is a file path
        elif os.path.exists(input_string):
            if os.path.isfile(input_string):
                file_extension = os.path.splitext(input_string)[1].lower()
                if file_extension == '.pdf':
                    return "PDF File"
                elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    return "Image File"
                else:
                    return "Unknown File Type"
            else:
                return "Not a File"

        # If it's neither a URL nor a valid file path
        else:
            return "Invalid Input"

    type = differentiate_input(placeroute)


    if type  == "URL":
        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(web_paths=(placeroute,))
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
    elif type == "PDF File":
        # Generate embeddings for the chunks.
        loader = PyPDFLoader(placeroute)
        splits = loader.load_and_split()
    else:
        return str("No valid input provided")


    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="mxbai-embed-large"))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)

# Example usage
if __name__ == "__main__":

    url = "https://en.wikipedia.org/wiki/Wikipedia"
    print(get_task_decomposition(url , "What is wikipedia"))
