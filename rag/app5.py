from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import lark
import getpass
import config
import warnings

# Disabling warnings:
warnings.filterwarnings("ignore")

docs = [
    Document(
        page_content="A tale about a young wizard and his journey in a magical school.",
        metadata={
            "title": "Harry Potter and the Philosopher's Stone",
            "author": "J.K. Rowling",
            "year_published": 1997,
            "genre": "Fiction",
            "isbn": "978-0747532699",
            "publisher": "Bloomsbury",
            "language": "English",
            "page_count": 223,
            "summary": "The first book in the Harry Potter series where Harry discovers his magical heritage.",
            "rating": 4.8,
        },
    ),
    Document(
        page_content="An epic tale of power, betrayal and love set in a fantastical world.",
        metadata={
            "title": "A Game of Thrones",
            "author": "George R.R. Martin",
            "year_published": 1996,
            "genre": "Fantasy",
            "isbn": "978-0553103540",
            "publisher": "Bantam",
            "language": "English",
            "page_count": 694,
            "summary": "The first book in A Song of Ice and Fire series, introducing the intricate world of Westeros.",
            "rating": 4.6,
        },
    ),
    Document(
        page_content="A futuristic society where firemen burn books to maintain order.",
        metadata={
            "title": "Fahrenheit 451",
            "author": "Ray Bradbury",
            "year_published": 1953,
            "genre": "Science Fiction",
            "isbn": "978-1451673319",
            "publisher": "Simon & Schuster",
            "language": "English",
            "page_count": 249,
            "summary": "In a future society, books are banned and firemen are tasked to burn any they find, leading one fireman to question his role.",
            "rating": 4.4,
        },
    ),
    Document(
        page_content="A young woman's life in the South during the Civil War and Reconstruction.",
        metadata={
            "title": "Gone with the Wind",
            "author": "Margaret Mitchell",
            "year_published": 1936,
            "genre": "Historical Fiction",
            "isbn": "978-0684830681",
            "publisher": "Macmillan",
            "language": "English",
            "page_count": 1037,
            "summary": "The tale of Scarlett O'Hara and her love affair with Rhett Butler, set against the backdrop of the American Civil War.",
            "rating": 4.3,
        },
    ),
    Document(
        page_content="A story about a hobbit's journey to destroy a powerful ring.",
        metadata={
            "title": "The Lord of the Rings",
            "author": "J.R.R. Tolkien",
            "year_published": 1954,
            "genre": "Fantasy",
            "isbn": "978-0618640157",
            "publisher": "Houghton Mifflin",
            "language": "English",
            "page_count": 1216,
            "summary": "The epic tale of Frodo Baggins and his quest to destroy the One Ring, accompanied by a group of diverse companions.",
            "rating": 4.7,
        },
    ),
]

embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY))

# Basic Info
basic_info = [
    AttributeInfo(name="title", description="The title of the book", type="string"),
    AttributeInfo(name="author", description="The author of the book", type="string"),
    AttributeInfo(
        name="year_published",
        description="The year the book was published",
        type="integer",
    ),
]

# Detailed Info
detailed_info = [
    AttributeInfo(
        name="genre", description="The genre of the book", type="string or list[string]"
    ),
    AttributeInfo(
        name="isbn",
        description="The International Standard Book Number for the book",
        type="string",
    ),
    AttributeInfo(
        name="publisher",
        description="The publishing house that published the book",
        type="string",
    ),
    AttributeInfo(
        name="language",
        description="The primary language the book is written in",
        type="string",
    ),
    AttributeInfo(
        name="page_count", description="Number of pages in the book", type="integer"
    ),
]

# Analysis
analysis = [
    AttributeInfo(
        name="summary",
        description="A brief summary or description of the book",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="An average rating for the book (from reviews), ranging from 1-5",
        type="float",
    ),
]

# Combining all lists into metadata_field_info
metadata_field_info = basic_info + detailed_info + analysis

document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(temperature=0,openai_api_key=config.OPENAI_API_KEY)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info
)

# Looking for sci-fi books
result = retriever.invoke("What are some sci-fi books?")
# print(result)

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
)

# result = retriever.invoke("Return 1 Fantasy book")
# result = retriever.invoke("I want some books that are published by the author J.K. Rowling for Harry Potter.")[0]
# result = retriever.invoke("Provide books with a rating over 4.0.")
print(result)
