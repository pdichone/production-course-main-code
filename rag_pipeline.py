"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""
# NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
llm = init_chat_model(model="gpt-6-luna", reasoning_effort="none", temperature=0.2)


def create_kb():
    """Create a vector store from knowledge base."""

    # split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )

    chunks = splitter.split_documents([doc])

    # create a vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store


def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know."""
    )

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Rag chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test the RAG chain
    # Test
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_rag_with_sources():

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the context below. Include which sources you used.

Context:
{context}

Question: {question}

Answer (include sources):"""
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with Sources:\n")
    answer = rag_chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")


def demo_rag_with_fallback():

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based ONLY on the following context.
If the answer is not in the context, respond with: "I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with Fallback:\n")

    questions = [
        "What is the pricing for LangSmith?",  # In knowledge base
        "What is the stock price of OpenAI?",  # Not in knowledge base
        "How do I deploy LangChain to AWS?",  # Not in knowledge base
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_structured_rag():
    """RAG with structured output."""

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    class RAGResponse(BaseModel):
        """Structured RAG response."""

        answer: str = Field(description="The answer to the question")
        confidence: str = Field(description="high, medium, or low")
        sources_used: List[str] = Field(description="List of sources referenced")
        follow_up: str = Field(description="Suggested follow-up question")

    structured_llm = llm.with_structured_output(RAGResponse)

    prompt = ChatPromptTemplate.from_template(
        """
Based on the context below, answer the question.

Context:
{context}

Question: {question}

Provide a structured response."""
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )
    print("Structured RAG Demo:\n")
    result = rag_chain.invoke("What is LangGraph?")

    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")
    print(f"Follow-up: {result.follow_up}")


# Exercise: Build a document Q&A system
def exercise_document_qa():
    """
    EXERCISE: Build a complete document Q&A system that:
    1. Takes a text document as input
    2. Splits and embeds it
    3. Allows multiple questions
    4. Returns answers with confidence scores
    """

    class DocumentQA:
        def __init__(self, document: str, source_name: str = "document"):
            # Split document
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            doc = Document(page_content=document, metadata={"source": source_name})
            chunks = splitter.split_documents([doc])

            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

            # Create chain
            self.llm = init_chat_model(model="gpt-6-luna", reasoning_effort="none", temperature=0.2)

            self.prompt = ChatPromptTemplate.from_template(
                """
Answer based on the context. Rate your confidence (high/medium/low).

Context: {context}
Question: {question}

Format: [Confidence: X] Answer"""
            )

            def format_docs(docs):
                return "\n".join(d.page_content for d in docs)

            self.chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

        def ask(self, question: str) -> str:
            return self.chain.invoke(question)

    # Test
    test_doc = """
    The Python programming language was created by Guido van Rossum.
    First released in 1991, Python emphasizes code readability.
    Python 3.12 was released in October 2023 with improved error messages.
    The language is named after Monty Python, not the snake.
    """

    qa = DocumentQA(test_doc, "python_facts")

    print("Document Q&A System:\n")
    questions = [
        "Who created Python?",
        "When was Python 3.12 released?",
        "Why is Python named Python?",
    ]

    for q in questions:
        answer = qa.ask(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    # demo_basic_rag()
    # demo_rag_with_sources()
    # demo_rag_with_fallback()
    # demo_structured_rag()
    exercise_document_qa()
