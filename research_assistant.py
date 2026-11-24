"""
Section 2 Project: AI Research Assistant
Complete RAG system with conversation memory
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import json


load_dotenv()


# ============================================================
# Data Models
# ============================================================
class ResearchResponse(BaseModel):
    """Structured response from the research assistant."""

    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low based on source quality")
    sources: List[str] = Field(description="List of source documents used")
    key_quotes: List[str] = Field(
        description="Relevant quotes from sources", default=[]
    )
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")


# ============================================================
# Research Assistant Class
# ============================================================


class AIResearchAssistant:
    """AI Research Assistant with document ingestion and retrieval."""

    def __init__(
        self,
        persist_directory: str = "./research_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_directory = persist_directory

        # 1. Embeddings - turns text into vectors
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # NOTE (2026-11): gpt-4o-mini/gpt-4o shut down 2027-01-14 -> gpt-6-luna. reasoning_effort="none" lets it accept temperature + tools.
        self.llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

        # 2. Splitter - breaks big docs into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # 3. Vector store - stores and searches embeddings
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="research_docs",
        )

        self.session_store: Dict[str, InMemoryChatMessageHistory] = {}

        print(f"Research Assistant initialized")
        print(f"  Vector store: {persist_directory}")
        print(f"  Documents indexed: {self.vectorstore._collection.count()}")

    def add_documents(
        self,
        documents: List[Document],
        source_name: Optional[str] = None,
    ) -> int:
        """Add documents to the research database."""

        # Tag with source name
        if source_name:
            for doc in documents:
                doc.metadata["source"] = source_name

        # Split into chunks
        chunks = self.splitter.split_documents(documents)

        # Timestamp each chunk
        for chunk in chunks:
            chunk.metadata["indexed_at"] = datetime.now().isoformat()

        # Store in vector DB
        self.vectorstore.add_documents(chunks)

        print(f"Added {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks)

    def add_text(self, text: str, source: str, metadata: dict = None) -> int:
        """Add a single text string as a document."""
        doc = Document(
            page_content=text, metadata={"source": source, **(metadata or {})}
        )
        return self.add_documents([doc])

    def add_texts(self, texts: List[str], source: str) -> int:
        """Add multiple text strings from the same source."""
        docs = [Document(page_content=t, metadata={"source": source}) for t in texts]
        return self.add_documents(docs)

    def get_document_count(self) -> int:
        """Get total number of indexed chunks."""
        return self.vectorstore._collection.count()

    def list_sources(self) -> List[str]:
        """List all unique sources in the database."""
        results = self.vectorstore._collection.get()
        sources = set()
        for metadata in results.get("metadatas", []):
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
        return sorted(list(sources))

    def _build_retriever(self, use_advanced: bool = False):
        """Build retriever -- basic or advanced"""

        # Base: simple similarity search
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        if not use_advanced:
            return base_retriever

        # Multi-query: LLM generates multiple search queries
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
        )

        return multi_retriever

    def _format_docs_for_context(self, docs) -> str:
        """Format retrieved documents into a string for the prompt."""
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session history."""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def ask_structured(
        self,
        question: str,
        session_id: str = "default",
        use_advanced: bool = True,
    ) -> ResearchResponse:
        """Ask a question and get a structured response."""

        # LLM that returns a Pydantic object instead of a string
        structured_llm = self.llm.with_structured_output(ResearchResponse)

        # Get memory
        history = self._get_session_history(session_id)

        # Retrieve
        retriever = self._build_retriever(use_advanced=use_advanced)
        docs = retriever.invoke(question)
        context = self._format_docs_for_context(docs)
        sources = list(set(d.metadata.get("source", "Unknown") for d in docs))

        # Prompt -- tell the LLM about available sources
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Research Assistant. Analyze the provided documents 
    and return a structured response.

    Rules:
    1. ONLY use information from the provided context
    2. If the context doesn't have the answer, say so in the answer field
    3. Set confidence: "high" if directly stated, "medium" if inferred, "low" if partial
    4. Include the source filenames you actually used
    5. Extract key quotes word-for-word from the context
    6. Suggest 2-3 follow-up questions the user might want to ask

    Use conversation history to understand follow-up questions.""",
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    """Context documents:

    {context}

    Available sources: {sources}

    Question: {question}""",
                ),
            ]
        )

        chain = prompt | structured_llm

        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "sources": ", ".join(sources),
                "history": (
                    history.messages[-10:]
                    if hasattr(history, "messages")
                    else history[-10:]
                ),
            }
        )

        # Save to memory (store just the answer text)
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response.answer))

        return response

    def ask(
        self, question: str, session_id: str = "default", use_advanced: bool = True
    ) -> str:
        """Ask a question against the research documents."""

        history = self._get_session_history(session_id)

        # Use basic or advanced retriever
        retriever = self._build_retriever(use_advanced=use_advanced)
        docs = retriever.invoke(question)
        context = self._format_docs_for_context(docs)

        # Step 3: Build the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Research Assistant. Answer questions
    based ONLY on the provided context documents.

    Rules:
    1. Only use information from the context below
    2. If the context doesn't have the answer, say so
    3. Cite which sources you used (e.g. "According to Source 1...")
    4. Rate your confidence: high, medium, or low""",
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    """Context documents:

    {context}

    Question: {question}

    Provide a clear answer with source citations.""",
                ),
            ]
        )

        # Step 4: Build and run the chain
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "history": history.messages[-10:],  # Last 10 messages for context
            }
        )

        # save this Q&A to history
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response))

        return response

    def clear_session(self, session_id: str):
        if session_id in self.session_store:
            self.session_store[session_id].clear()
            print(f"Cleared session: {session_id}")

    def get_session_messages(self, session_id: str) -> list:
        """Get conversation history as readable dicts."""
        if session_id not in self.session_store:
            return []
        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in self.session_store[session_id].messages
        ]

    def compare_retrievers(self, question: str):
        """Show basic vs advanced retrieval side by side."""

        print(f'Question: "{question}"\n')

        # --- Basic ---
        basic = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        basic_docs = basic.invoke(question)

        print("=" * 60)
        print(f"BASIC RETRIEVER: {len(basic_docs)} chunks")
        print("=" * 60)

        basic_total_chars = 0
        for i, doc in enumerate(basic_docs):
            source = doc.metadata.get("source", "Unknown")
            basic_total_chars += len(doc.page_content)
            print(f"\n  Chunk {i+1} [{source}] ({len(doc.page_content)} chars):")
            print(f"  {doc.page_content[:150]}...")

        print(f"\n  Total text sent to LLM: {basic_total_chars} chars")

        # --- Advanced ---
        advanced = self._build_retriever(use_advanced=True)
        advanced_docs = advanced.invoke(question)

        print("\n" + "=" * 60)
        print(f"ADVANCED RETRIEVER: {len(advanced_docs)} chunks")
        print("=" * 60)

        advanced_total_chars = 0
        for i, doc in enumerate(advanced_docs):
            source = doc.metadata.get("source", "Unknown")
            advanced_total_chars += len(doc.page_content)
            print(f"\n  Chunk {i+1} [{source}] ({len(doc.page_content)} chars):")
            print(f"  {doc.page_content[:150]}...")

        print(f"\n  Total text sent to LLM: {advanced_total_chars} chars")

        # --- Summary ---
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"  Basic:    {len(basic_docs)} chunks, {basic_total_chars} chars")
        print(f"  Advanced: {len(advanced_docs)} chunks, {advanced_total_chars} chars")

        if advanced_total_chars < basic_total_chars:
            reduction = round((1 - advanced_total_chars / basic_total_chars) * 100)
            print(f"  Compression saved {reduction}% of tokens!")
        else:
            print(f"  Advanced found more targeted content")


def print_research_response(question: str, response: ResearchResponse):
    """Pretty print a structured research response."""

    print(f"\nQ: {question}")
    print(f"\n  Answer: {response.answer}")
    print(f"\n  Confidence: {response.confidence}")
    print(f"  Sources: {', '.join(response.sources)}")

    if response.key_quotes:
        print(f"\n  Key Quotes:")
        for q in response.key_quotes:
            print(f'    - "{q}"')

    print(f"\n  Follow-up Questions:")
    for fq in response.follow_up_questions:
        print(f"    - {fq}")


if __name__ == "__main__":
    import shutil

    shutil.rmtree("./research_db", ignore_errors=True)
    assistant = AIResearchAssistant()

    # Add research docs
    assistant.add_text(
        """
        Attention Mechanisms in Neural Networks

        The attention mechanism was introduced in "Attention Is All You Need"
        by Vaswani et al. (2017). It allows models to focus on relevant parts
        of the input when generating output.

        Key concepts:
        - Query, Key, Value (QKV) triplets
        - Scaled dot-product attention
        - Multi-head attention for parallel processing

        The transformer architecture has become the foundation for modern NLP
        models including BERT, GPT, and T5.
        """,
        source="attention_mechanisms.pdf",
    )

    assistant.add_text(
        """
        Retrieval-Augmented Generation (RAG)

        RAG combines retrieval systems with generative models. First introduced
        by Lewis et al. (2020), RAG addresses the limitation of LLMs being
        limited to their training data.

        Components of a RAG system:
        1. Document store with vector embeddings
        2. Retriever to find relevant documents
        3. Generator (LLM) to produce responses

        Benefits include reduced hallucination, up-to-date information,
        and source attribution.
        """,
        source="rag_survey.pdf",
    )

    assistant.add_text(
        """
        LangChain and LangGraph Framework Overview

        LangChain is an open-source framework for building LLM applications.
        Key features include modular components, integration with 50+ LLM
        providers, and built-in RAG utilities.

        LangGraph extends LangChain for stateful applications with
        graph-based state management, support for cycles and loops,
        and human-in-the-loop workflows.
        """,
        source="langchain_docs.md",
    )

    print(f"\nIndexed: {assistant.get_document_count()} chunks")

    session = "structured_demo"

    # --- Step 1: String vs Structured comparison ---
    print("\n" + "=" * 60)
    print("STEP 1: String response vs Structured response")
    print("=" * 60)

    question = "What is RAG and what are its benefits?"

    print("\n--- String response (ask) ---")
    string_response = assistant.ask(question, "string_test")
    print(f"Type: {type(string_response)}")
    print(f"Response: {string_response[:200]}...")

    print("\n--- Structured response (ask_structured) ---")
    structured_response = assistant.ask_structured(question, "struct_test")
    print(f"Type: {type(structured_response)}")
    print(f"answer:             {structured_response.answer[:100]}...")
    print(f"confidence:         {structured_response.confidence}")
    print(f"sources:            {structured_response.sources}")
    print(f"key_quotes:         {structured_response.key_quotes[:2]}")
    print(f"follow_up_questions: {structured_response.follow_up_questions}")

    # --- Step 2: Access fields directly ---
    print("\n" + "=" * 60)
    print("STEP 2: Use fields in your code")
    print("=" * 60)

    r = assistant.ask_structured("What is the attention mechanism?", session)

    # This is what your app code looks like
    if r.confidence == "high":
        print(f"\n  Confident answer from: {', '.join(r.sources)}")
    else:
        print(f"\n  Low confidence -- may need more sources")

    print(f"\n  Answer: {r.answer[:200]}")

    print(f"\n  Suggested follow-ups:")
    for fq in r.follow_up_questions:
        print(f"    -> {fq}")

    # --- Step 3: Multi-turn with structured output ---
    print("\n" + "=" * 60)
    print("STEP 3: Memory works with structured output too")
    print("=" * 60)

    q1 = "What are the components of RAG?"
    print(f"\nUser: {q1}")
    r1 = assistant.ask_structured(q1, session)
    print_research_response(q1, r1)

    q2 = "How does the second component work?"
    print(f"\n{'- '*30}")
    print(f"\nUser: {q2}")
    r2 = assistant.ask_structured(q2, session)
    print_research_response(q2, r2)

    q3 = "Connect everything we discussed to LangChain."
    print(f"\n{'- '*30}")
    print(f"\nUser: {q3}")
    r3 = assistant.ask_structured(q3, session)
    print_research_response(q3, r3)

    # --- Step 4: Final stats ---
    print("\n" + "=" * 60)
    print("FINAL: What we built across 5 videos")
    print("=" * 60)

    history = assistant._get_session_history(session)
    msg_count = len(history.messages) if hasattr(history, "messages") else len(history)

    print(
        f"""
  Document ingestion    -> {assistant.get_document_count()} chunks indexed
  Sources tracked       -> {assistant.list_sources()}
  Basic retrieval       -> similarity search
  Advanced retrieval    -> multi-query + compression
  Conversation memory   -> {msg_count} messages in session '{session}'
  Structured output     -> ResearchResponse with {len(ResearchResponse.model_fields)} fields

  From raw text to a production-ready research assistant.
  That's the full RAG pipeline.
    """
    )

    # Cleanup
    shutil.rmtree("./research_db", ignore_errors=True)
