from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from vector import retriever

load_dotenv()

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
prompt = ChatPromptTemplate.from_template(template)

# Combine the prompt and the model into a chain that can be invoked with the 
# retrieved information and the user's question.
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
    # based on the user's question.
    CaRE_Reviews = retriever.invoke(question)
    # Invoke the chain with the retrieved information 
    # and the user's question to get the answer.
    Result = chain.invoke({"CaRE_Reviews": CaRE_Reviews, "question": question})
    print(Result.content)


