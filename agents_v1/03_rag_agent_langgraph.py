"""
RAG AGENT - LangGraph 1.0 з Checkpointing
На базі документації: LangGraph Agentic RAG (2025)

Implements:
- Agentic RAG pattern з dynamic retrieval
- LangGraph StateGraph для orchestration
- Checkpointing з MemorySaver для persistence
- Multi-step reasoning: retrieve → grade → generate
- LangSmith automatic tracing

Pattern:
User Query → Retrieve Docs → Grade Relevance → Generate Answer
          ↓ (if irrelevant)
       Rewrite Query → Retrieve Again
"""

import os
from typing import TypedDict, Annotated, List, Dict
from operator import add
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# LANGSMITH SETUP
# ============================================================================

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("OK LangSmith трейсинг активний для RAG Agent")
    print(f"Stats: Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print("🔍 All RAG operations will be traced: retrieve → grade → generate\n")
else:
    print("WARNING  LangSmith не ввімкнений\n")


GLOBAL_RETRIEVER = None  # initialized in test_rag_agent()

# ============================================================================
# KNOWLEDGE BASE - Створюємо векторну базу для RAG
# ============================================================================

def create_knowledge_base():
    """
    Створює векторну базу знань з документами

    Returns:
        FAISS retriever for similarity search
    """

    documents = [
        Document(
            page_content="""
            LangChain 1.0 Release (October 2025):
            LangChain 1.0 introduces stable APIs with create_agent function.
            No breaking changes until 2.0. Key features: middleware architecture,
            improved observability, production-ready patterns.
            """,
            metadata={"source": "release_notes", "topic": "langchain", "date": "2025-10"}
        ),
        Document(
            page_content="""
            LangGraph 1.0 Checkpointing:
            LangGraph provides checkpointing using MemorySaver for development
            and PostgresSaver for production. Checkpoints allow agents to pause,
            resume, and time-travel through execution states. Thread_id enables
            separate conversation sessions.
            """,
            metadata={"source": "docs", "topic": "langgraph", "feature": "checkpointing"}
        ),
        Document(
            page_content="""
            Agent Middleware (New in 2025):
            LangChain 1.0 middleware provides three hooks:
            - before_model: Pre-processing before LLM calls
            - after_model: Post-processing after LLM responds
            - modify_model_request: Modify tools, prompts, messages
            Built-in middlewares: Human-in-the-loop, Summarization, PII redaction.
            """,
            metadata={"source": "docs", "topic": "middleware", "version": "1.0"}
        ),
        Document(
            page_content="""
            Agentic RAG Pattern:
            Agentic RAG uses AI agents to dynamically manage retrieval strategies.
            Pattern: Query → Retrieve → Grade → (if bad) Rewrite → Retrieve Again.
            Agents can iteratively refine context, route to web search if needed,
            and orchestrate multiple retrieval sources.
            """,
            metadata={"source": "patterns", "topic": "rag", "type": "agentic"}
        ),
        Document(
            page_content="""
            LangSmith Integration:
            Set LANGCHAIN_TRACING_V2=true to enable automatic tracing.
            LangSmith provides observability for agents: trace visualization,
            cost tracking, latency analysis, and A/B testing capabilities.
            Works with LangChain, LangGraph, and custom agents.
            """,
            metadata={"source": "docs", "topic": "observability", "tool": "langsmith"}
        ),
        Document(
            page_content="""
            Production Best Practices:
            - Use create_agent instead of deprecated AgentExecutor
            - Implement error handling and max_iterations
            - Add middleware for security and cost control
            - Use checkpointing for long-running agents
            - Monitor with LangSmith in production
            - Pin dependency versions for stability
            """,
            metadata={"source": "best_practices", "topic": "production"}
        )
    ]

    print("KB: Creating knowledge base with documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc.metadata.get('topic', 'general')}: {doc.page_content[:50]}...")

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Return top 2 most relevant docs
    )

    print(f"OK Knowledge base created with {len(documents)} documents\n")

    return retriever


# ============================================================================
# AGENTIC RAG STATE
# ============================================================================

class RAGState(TypedDict):
    """
    State для Agentic RAG workflow

    Flow:
    question → retrieved_docs → relevance_grade → answer
    """
    question: str                          # User's question
    retrieved_docs: List[Document]         # Retrieved documents
    relevance_grade: str                   # "relevant" or "irrelevant"
    rewrite_count: int                     # How many times we rewrote query
    answer: str                            # Final answer
    reasoning: Annotated[List[str], add]  # Reasoning steps (accumulated)


# ============================================================================
# AGENTIC RAG NODES
# ============================================================================

def retrieve_documents(state: RAGState) -> RAGState:
    """
    Node 1: Retrieve documents based on question
    """

    print(f"\n{'='*60}")
    print("KB: NODE: Retrieve Documents")
    print(f"{'='*60}")

    question = state["question"]
    print(f"Question: {question}")

    # Get retriever from global scope (in production pass via state or config)
    docs = GLOBAL_RETRIEVER.invoke(question)

    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"     Content: {doc.page_content[:80]}...")

    reasoning = [f"Retrieved {len(docs)} documents for query"]

    return {
        **state,
        "retrieved_docs": docs,
        "reasoning": reasoning
    }


def grade_documents(state: RAGState) -> RAGState:
    """
    Node 2: Grade relevance of retrieved documents
    Uses LLM to determine if docs are relevant to question
    """

    print(f"\n{'='*60}")
    print("⚖️  NODE: Grade Document Relevance")
    print(f"{'='*60}")

    question = state["question"]
    docs = state["retrieved_docs"]

    # Structured output для grading
    class GradeOutput(BaseModel):
        """Grade output schema"""
        relevance: str = Field(description="'relevant' or 'irrelevant'")
        reasoning: str = Field(description="Why this grade")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(GradeOutput)

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader. Assess if retrieved documents are relevant to the question."),
        ("human", """Question: {question}

Retrieved Documents:
{documents}

Are these documents relevant to answer the question?
Respond with 'relevant' or 'irrelevant' and explain why.""")
    ])

    # Format documents
    docs_text = "\n\n".join([
        f"Doc {i+1}: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    # Grade
    chain = grade_prompt | structured_llm
    grade_result = chain.invoke({
        "question": question,
        "documents": docs_text
    })

    print(f"\nGrade: {grade_result.relevance.upper()}")
    print(f"Reasoning: {grade_result.reasoning}")

    reasoning = [f"Graded documents as {grade_result.relevance}: {grade_result.reasoning}"]

    return {
        **state,
        "relevance_grade": grade_result.relevance,
        "reasoning": reasoning
    }


def rewrite_query(state: RAGState) -> RAGState:
    """
    Node 3: Rewrite query if documents were irrelevant
    Uses LLM to reformulate question for better retrieval
    """

    print(f"\n{'='*60}")
    print("✍️  NODE: Rewrite Query")
    print(f"{'='*60}")

    original_question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    print(f"Original question: {original_question}")
    print(f"Rewrite attempt: {rewrite_count + 1}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query rewriter. Improve the question to get better search results."),
        ("human", """Original question: {question}

The retrieved documents were not relevant. Rewrite this question to:
1. Be more specific
2. Include key technical terms
3. Focus on the core information need

Return ONLY the rewritten question, nothing else.""")
    ])

    chain = rewrite_prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": original_question})

    print(f"Rewritten question: {new_question}\n")

    reasoning = [f"Rewrote query (attempt {rewrite_count + 1}): '{new_question}'"]

    return {
        **state,
        "question": new_question,
        "rewrite_count": rewrite_count + 1,
        "reasoning": reasoning
    }


def generate_answer(state: RAGState) -> RAGState:
    """
    Node 4: Generate final answer using retrieved documents
    """

    print(f"\n{'='*60}")
    print("TIP: NODE: Generate Answer")
    print(f"{'='*60}")

    question = state["question"]
    docs = state["retrieved_docs"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Answer questions based on the provided context.

Context:
{context}

Instructions:
- Use ONLY information from the context
- Be concise and accurate
- If context doesn't contain the answer, say so
- Include relevant details and examples"""),
        ("human", "{question}"),
    ])

    # Format context
    context = "\n\n".join([
        f"Source {i+1} ({doc.metadata.get('topic', 'N/A')}):\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print(f"Generated answer ({len(answer)} chars)")

    reasoning = [f"Generated final answer using {len(docs)} documents"]

    return {
        **state,
        "answer": answer,
        "reasoning": reasoning
    }


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def decide_next_step(state: RAGState) -> str:
    """
    Conditional edge: Decide whether to generate or rewrite

    Logic:
    - If docs are relevant → generate answer
    - If docs are irrelevant AND we haven't rewritten too many times → rewrite
    - If we've rewritten too much → generate anyway (best effort)
    """

    relevance = state.get("relevance_grade", "irrelevant")
    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = 2

    print(f"\n🤔 ROUTING: relevance={relevance}, rewrites={rewrite_count}")

    if relevance == "relevant":
        print("   → Going to: generate_answer (docs are good)")
        return "generate"

    elif rewrite_count < max_rewrites:
        print(f"   → Going to: rewrite_query (try {rewrite_count + 1}/{max_rewrites})")
        return "rewrite"

    else:
        print("   → Going to: generate_answer (max rewrites reached)")
        return "generate"


# ============================================================================
# BUILD LANGGRAPH
# ============================================================================

def create_rag_agent():
    """
    Створює Agentic RAG workflow з LangGraph

    Graph structure:
    START → retrieve → grade → [relevant? → generate | irrelevant? → rewrite → retrieve]
    """

    print("=" * 70)
    print("Retry: BUILDING AGENTIC RAG GRAPH")
    print("=" * 70 + "\n")

    # Create graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_answer)

    # Build graph structure
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")

    # Conditional routing from grade
    workflow.add_conditional_edges(
        "grade",
        decide_next_step,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    # After rewrite, go back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Generate is end
    workflow.add_edge("generate", END)

    # Compile with checkpointer for persistence
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    print("Graph structure:")
    print("  START → retrieve → grade → [decide]")
    print("                      ↓")
    print("             if relevant: generate → END")
    print("             if irrelevant: rewrite → retrieve (loop)")
    print()
    print("OK RAG Agent compiled with MemorySaver checkpointer\n")

    return app


# ============================================================================
# TESTING
# ============================================================================

def test_rag_agent():
    """Test Agentic RAG with different queries"""

    global GLOBAL_RETRIEVER
    GLOBAL_RETRIEVER = create_knowledge_base()

    agent = create_rag_agent()

    test_queries = [
        {
            "question": "What are the new middleware hooks in LangChain 1.0?",
            "expected": "Should find info about before_model, after_model, modify_model_request"
        },
        {
            "question": "How does checkpointing work in LangGraph?",
            "expected": "Should find info about MemorySaver and thread_id"
        },
        {
            "question": "What is the capital of France?",
            "expected": "Should say not in context (trigger rewrite?)"
        }
    ]

    for i, test in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {test['question']}")
        print(f"Expected: {test['expected']}")
        print("=" * 70)

        # Use unique thread_id for each query to maintain separate sessions
        config = {"configurable": {"thread_id": f"test_query_{i}"}}

        initial_state = {
            "question": test["question"],
            "retrieved_docs": [],
            "relevance_grade": "",
            "rewrite_count": 0,
            "answer": "",
            "reasoning": []
        }

        try:
            result = agent.invoke(initial_state, config)

            print(f"\n{'='*70}")
            print("FINAL RESULT:")
            print(f"{'='*70}")
            print(f"\nLOG Question: {result['question']}")
            print(f"\nTIP: Answer:\n{result['answer']}")
            print(f"\n🔍 Reasoning steps:")
            for j, step in enumerate(result['reasoning'], 1):
                print(f"  {j}. {step}")
            print(f"\nStats: Stats:")
            print(f"  - Query rewrites: {result['rewrite_count']}")
            print(f"  - Documents used: {len(result['retrieved_docs'])}")
            print(f"  - Final grade: {result['relevance_grade']}")

        except Exception as e:
            print(f"\nERROR: Error: {e}")
            import traceback
            traceback.print_exc()

        if i < len(test_queries):
            input("\nPAUSE  Press Enter for next test...\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\nTARGET LangGraph 1.0 - Agentic RAG with Checkpointing")
    print("=" * 70)
    print("\nFeatures:")
    print("  OK Dynamic retrieval with relevance grading")
    print("  OK Automatic query rewriting if docs irrelevant")
    print("  OK LangGraph StateGraph for orchestration")
    print("  OK MemorySaver checkpointing for persistence")
    print("  OK Conditional routing based on relevance")
    print("  OK LangSmith tracing for full observability")
    print("\n" + "=" * 70 + "\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: ERROR: OPENAI_API_KEY not found!")
        exit(1)

    try:
        test_rag_agent()

        print("\n" + "=" * 70)
        print("OK ALL RAG TESTS COMPLETED")
        print("=" * 70)
        print("\nTIP: Agentic RAG Pattern Benefits:")
        print("  • Dynamic retrieval strategy")
        print("  • Self-correction through query rewriting")
        print("  • Relevance grading prevents hallucination")
        print("  • Checkpointing enables pause/resume")
        print("\nTIP: Check LangSmith for full execution traces!\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted")
    except Exception as e:
        print(f"\nERROR: Error: {e}")
        import traceback
        traceback.print_exc()
