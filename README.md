# Step 0  - Set up the virtual environment and activate the same within your assigned folder

Python -m venv venv
.\venv\Scripts\activate

# Step 1 - Install all the packages from the requirements.txt

langchain - langchain is a framework for building applications with LLMs. It provides tools for chaining together LLM calls, 
managing prompts, and more.

langchain-core - The langchain-core package contains the core functionality of the LangChain framework, including the base classes and utilities for building applications with LLMs.

langchain-community - The langchain-community package contains community-contributed components for the LangChain framework,
such as additional integrations and utilities.

langchain-openai - The langchain-openai package provides integration with OpenAI's language models, allowing you to easily use OpenAI's models within the LangChain framework.

langchain-anthropic - The langchain-anthropic package provides integration with Anthropic's language models, allowing you to easily use Anthropic's models within the LangChain framework.

openai or anthropic - This package is the official Python client library for the OpenAI API / Anthropic API, which allows you to interact with the language models and other services.

chromadb - ChromaDB is a vector database that allows you to store and query high-dimensional vectors, which is useful for applications like semantic search and similarity matching.

pandas - Pandas is a powerful data manipulation and analysis library for Python, providing data structures and functions for working with structured data.

tiktoken - Tiktoken is a library for tokenizing text, which is essential for working with language models, as it allows you to convert text into tokens that can be processed by the models.

python-dotenv - Python-dotenv is a library for loading environment variables from a .env file, which is useful for managing configuration settings in Python applications.

pydantic - Pydantic is a data validation and settings management library for Python, which allows you to define data models and validate data against those models.

numpy - numpy is a fundamental package for scientific computing in Python, providing support for arrays, matrices, and a wide range of mathematical functions.

sentence-transformers - The sentence-transformers library provides pre-trained models for generating sentence embeddings,
which can be used for tasks like semantic search, clustering, and more.


# Step 2 - Going through the python scripts. Start with vector.py and then move to the main calling script main.py

Vector.py

# This code reads a CSV file, creates documents from its rows, 
# generates embeddings using the OpenAI / Anthropic model, 
# and stores them in a Chroma vector database for retrieval.

# importing the os module, which provides a way of 
# using operating system dependent functionality, 
# such as checking if a directory exists or 
# creating a new one.
import os

# importing the pandas library, which is used for data manipulation and analysis, 
# particularly for working with structured data like CSV files.
import pandas as pd

# importing the Document class from langchain_core.documents, 
#which is used to create document objects that can be stored 
# in the vector database.
from langchain_core.documents import Document
# importing the Chroma class from langchain_community.vectorstores, 
# which is used to create and manage a vector database for 
# storing document embeddings.
from langchain_community.vectorstores import Chroma
# importing the OpenAIEmbeddings class from langchain_openai, 
# which is used to generate embeddings for the documents using 
# OpenAI's embedding models.
from langchain_openai import OpenAIEmbeddings
# importing the load_dotenv function from the dotenv library, 
# which is used to load environment variables from a .env file,
# allowing the code to access API keys and other 
# configuration settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the directory where the Chroma vector database 
# will be stored.
Chroma_Dir = "./chroma_db"

# Read the CSV file containing the AML CMT records 
# into a DataFrame.
df = pd.read_csv("AML_CMT_2025.csv")

# Create a list of Document objects from the DataFrame rows,
# where each document's content is a string representation 
# of the row's data.
docs = []

# Iterate over each row in the DataFrame and 
# create a Document, where the page content is a string 
# that concatenates and formats the column names and 
# their corresponding values, and the metadata indicates 
# the source of the document as "csv".

for _, row in df.iterrows():
    content = "\n".join(
        [f"{col}: {row[col]}" for col in df.columns]
    )

    docs.append(
        Document(
            page_content=content,
            metadata={"source": "csv"}
        )
    )

# Initialize the OpenAIEmbeddings 
# with the desired embedding model.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Create a Chroma vector database from the documents 
# and their embeddings, and specify a directory 
# to persist the database.

if os.path.exists(Chroma_Dir):
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=Chroma_Dir
    )

else:

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

# Create a retriever from the vector database that can be used to get relevant information based on a query, 
# specifying the number of top results to return.
retriever = vectordb.as_retriever(search_kwargs={"k": 500})

# results = retriever.invoke(
# "Which case_ID has highest transaction amount?"
# )

# for r in results:
# print(r.page_content)


main.py

# This code sets up a simple question-answering system 
# using a language model and a retriever to 
# fetch relevant information based on user queries.

# It uses the `dotenv` library to load environment variables
from dotenv import load_dotenv

# It imports the `ChatOpenAI` class from the `langchain_openai` library
from langchain_openai import ChatOpenAI

# It imports the `ChatPromptTemplate` class from 
# the `langchain_core.prompts` module
from langchain_core.prompts import ChatPromptTemplate

# It imports the `retriever` function from the `vector` module, 
# which is responsible for fetching relevant information based 
# on the user's question.
from vector import retriever

# Load environment variables from a .env file, which may include
# API keys or other configuration settings needed for the model 
# and retriever to function properly.
load_dotenv()

# Initialize the ChatOpenAI model with specific parameters:
# - `model`: specifies the model to use (in this case, "gpt-4o-mini").
# - `temperature`: controls the randomness of the model's output 
# (0.2 means less random and more focused responses).
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Define a prompt template that instructs the model on 
# how to use the retrieved information to answer questions.
template = """

You are a helpful assistant with a decade of experience
answering questions about AML CMT records.

Use the following retrieved information:

{CaRE_Reviews}

Question:
{question}

Your answer should:
- Be short
- Be concise
- Use bullet points

If you don't know the answer, say you don't know.

"""

# Create a ChatPromptTemplate from the defined template.
# This template will be used to format the input for the model,
# combining the retrieved information and the user's question.
prompt = ChatPromptTemplate.from_template(template)

# Combine the prompt and the model into a chain 
# that can be invoked with the retrieved information 
# and the user's question.
chain = prompt | model

# Start a loop to continuously ask the user 
# for questions until they choose to quit.
while True:
    print("\n------------------------------")
    question = input("Ask your question about the AML CMT records (q to quit): ")
    print("------------------------------\n")

    if question.lower() == "q":
        break

# Use the retriever to get relevant information 
# based on the user's question from the AML CMT records and 
# store it in the variable `CaRE_Reviews`.
    CaRE_Reviews = retriever.invoke(question)

# Invoke the chain with the retrieved information 
# and the user's question to get the answer relevant 
# to the AML CMT records, and print the result.
    Result = chain.invoke({"CaRE_Reviews": CaRE_Reviews, "question": question})
    print(Result.content)






Which analyst handled the highest number of cases?
Which analysts handled more than 50 cases?
Which cases are still open for investigation?
Which cases have no investigation end date?
What types of suspicious activities are captured in the dataset?
What are common AML investigation notes captured?
Who is the customer in Case AML0001?
What was the alert trigger for AML0001?
What was the outcome of AML0001?
Which analyst was assigned to AML0001?
What suspicious activity was identified in AML0001?
What percentage of cases are still under investigation?
What percentage of cases were escalated?
What percentage of cases are closed?
Which alert categories are most associated with escalated investigations?
Which transaction patterns are considered suspicious in the dataset?
Which AML typologies can learners identify from this dataset?
Create a cross sectional table of different suspicious transaction patterns in the dataset along with the frequency of cases along each cohort? Explain your observations.
