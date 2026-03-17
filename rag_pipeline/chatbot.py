from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

def create_rag_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question using ONLY the context below.
Always mention which document the information comes from using the source metadata.
If the answer is not in the context, say 'I could not find that in the uploaded documents.'

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[From: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
            "chat_history": lambda x: []
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(" RAG chain created!")
    return chain

def ask_question(chain, question):
    answer = chain.invoke(question)
    return answer, []