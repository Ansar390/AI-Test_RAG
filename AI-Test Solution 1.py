
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import pinecone
# from langchain_community.vectorstores import Pinecone
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_pinecone import PineconeVectorStore

import api_keys

pc = Pinecone(api_key=api_keys.os.environ["PINECONE_API_KEY"])

index=pc.Index("ai-test")
index_name = "ai-test"

embeddings = HuggingFaceEmbeddings()





"""**LLAMA 3.1 8B LLM with Vecor DB (RAG)**"""

docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
llm_grok = ChatGroq(

            groq_api_key=api_keys.os.environ["GROQ_API_KEY"],

            model_name='llama-3.1-8b-instant', temperature=0.0

    )
# llama3-70b-8192

template ="""You are intelligent chatbot designed for users to ask questions about a conversation between a doctor and a patient. You will answer the relevant information from the provided transcript and answer user queries.



{context}
Question: {question}

"""
retriever=docsearch.as_retriever()

prompt = ChatPromptTemplate.from_template(template)

output_parser= StrOutputParser()
chain= ({
    "context":retriever,
    "question": RunnablePassthrough()

}
| prompt
| llm_grok
| output_parser
)


while True:
    query=input("Enter your query: ")
    if query=="":
        break
    else:

        for chunk in chain.stream(query):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
            else:
                print(chunk.content, end="", flush=True)

# - What doctor diagnosed?
# - What medicine doctor mentioned?
# - Duration of medicine?
# - Precautions if any?
# - Activity if any?


