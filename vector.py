# This code reads a CSV file, creates documents from its rows, 
# generates embeddings using the OpenAI model model, 
# and stores them in a Chroma vector database for retrieval.
import os

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

Chroma_Dir = "./chroma_db"

# Read the CSV file containing the AML CMT records 
# into a DataFrame.
df = pd.read_csv("AML_CMT_2025.csv")

# Create a list of Document objects from the DataFrame rows,
# where each document's content is a string representation 
# of the row's data.
docs = []

# Iterate over each row in the DataFrame and create a Document
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

#results = retriever.invoke(
#   "Which case_ID has highest transaction amount?"
#)

#for r in results:
#    print(r.page_content)