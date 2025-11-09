"""
МУЛЬТИАГЕНТНА СИСТЕМА - LangGraph 1.0 Supervisor Pattern
Розширення 03_rag_agent_langgraph.py з координацією кількох спеціалізованих агентів

Architecture:
- SupervisorAgent: Координує команду агентів
- ResearcherAgent: RAG-пошук в knowledge base
- AnalyzerAgent: Аналіз знайденої інформації
- SynthesizerAgent: Синтез фінальної відповіді

LangSmith Integration: Трейсинг всіх агентів та їх взаємодії
"""

import os
from typing import TypedDict, Annotated, Literal, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import operator

load_dotenv()

# ============================================================================
# LANGSMITH VERIFICATION
# ============================================================================

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("✅ LangSmith трейсинг активний для мультиагентної системи")
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}\n")
else:
    print("⚠️  LangSmith не ввімкнений\n")


# ============================================================================
# STATE DEFINITION - Shared state для всіх агентів
# ============================================================================

class MultiAgentState(TypedDict):
    """
    Спільний state для всієї мультиагентної системи

    Включає:
    - messages: історія комунікації між агентами
    - question: початкове питання користувача
    - current_agent: який агент зараз активний
    - retrieved_docs: документи з RAG (ResearcherAgent)
    - analysis: результат аналізу (AnalyzerAgent)
    - final_answer: фінальна відповідь (SynthesizerAgent)
    - supervisor_decision: рішення supervisor про наступний крок
    - iteration_count: лічильник ітерацій
    """
    messages: Annotated[List, operator.add]  # Додавання повідомлень
    question: str
    current_agent: str
    retrieved_docs: List[Document]
    analysis: str
    final_answer: str
    supervisor_decision: str
    iteration_count: int


# ============================================================================
# KNOWLEDGE BASE - LangGraph 1.0 Documentation
# ============================================================================

# Створюємо knowledge base з інформацією про LangGraph 1.0
documents = [
    Document(
        page_content="""LangGraph 1.0 Supervisor Pattern:
        Hierarchical multi-agent architecture where a central supervisor agent coordinates multiple specialized agents.
        The supervisor receives user input, delegates work to sub-agents based on their capabilities,
        and when sub-agents respond, control returns to the supervisor. Each agent maintains its own scratchpad
        while the supervisor orchestrates communication and task delegation. This pattern is ideal for complex
        workflows requiring specialized expertise.""",
        metadata={"source": "supervisor_pattern", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph StateGraph API:
        StateGraph is the core abstraction for building multi-agent systems in LangGraph 1.0.
        It maintains centralized state storing intermediate results and metadata. Agents are represented as nodes,
        connections as edges. Control flow is managed by edges with conditional routing.
        StateGraph enables parallel execution, conditional branching, and state persistence through checkpointing.
        Key methods: add_node(), add_edge(), add_conditional_edges(), compile().""",
        metadata={"source": "stategraph_api", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Checkpointing Mechanisms:
        LangGraph 1.0 provides persistent state storage through checkpointing. MemorySaver for development,
        PostgresSaver/SqliteSaver for production. Checkpoints enable time-travel through execution states,
        rollback to prior points, and replay workflows with adjusted parameters. Each checkpoint is identified
        by thread_id allowing separate conversation sessions. Prevents state corruption and ensures data integrity.
        Checkpoint memory is managed using threads for isolation.""",
        metadata={"source": "checkpointing", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Multi-Agent Coordination:
        Agent coordination patterns in LangGraph 1.0 include: 1) Supervisor Pattern - central coordinator,
        2) Hierarchical Teams - nested supervision layers, 3) Network Pattern - peer-to-peer communication.
        State management handles agent communication through shared StateGraph. Each agent reads/writes to state.
        Communication via messages in state. Output consolidation through final synthesis node.
        Guardrails via conditional routing and validation nodes.""",
        metadata={"source": "coordination", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Swarm (2025 Release):
        New lightweight library for swarm-style multi-agent systems. Maintains shared state with conversation history
        and active_agent marker. Uses checkpointer (in-memory or database) to persist state across turns.
        Aims to make multi-agent coordination easier and more reliable. Provides abstractions to link individual
        LLM agents into one integrated application. Emphasizes state management and checkpointing for reliability.
        Supports parallel agent execution with conflict resolution.""",
        metadata={"source": "swarm", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Server & Persistence:
        LangGraph 1.0 includes LangGraph Server for production deployments. Provides comprehensive persistence:
        stores checkpoints, memories, thread metadata, and assistant configurations. Enables distributed multi-agent
        systems with API endpoints. Supports horizontal scaling. Built-in monitoring and observability.
        Integration with LangSmith for tracing all agents. REST API for agent invocation and state inspection.
        Webhook support for async workflows.""",
        metadata={"source": "server", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Error Handling & Recovery:
        Multi-agent systems in LangGraph 1.0 include robust error handling. Each node can handle exceptions gracefully.
        Conditional edges for error routing. Retry mechanisms with exponential backoff. Circuit breakers to prevent
        cascade failures. State rollback on errors using checkpoints. Validation nodes before critical operations.
        Supervisor can reassign tasks if agent fails. Timeout handling at node level.
        Error messages propagated through state.""",
        metadata={"source": "error_handling", "version": "1.0"}
    ),
    Document(
        page_content="""LangGraph Best Practices for Multi-Agent Systems:
        1) Define clear agent responsibilities and capabilities. 2) Use supervisor for complex coordination.
        3) Implement checkpointing for long-running workflows. 4) Add validation nodes between critical steps.
        5) Use conditional edges for dynamic routing. 6) Keep state schema simple and typed.
        7) Implement timeouts for all agent operations. 8) Use LangSmith for observability.
        9) Test each agent independently before integration. 10) Design for agent failure and recovery.""",
        metadata={"source": "best_practices", "version": "1.0"}
    ),
]

print(f"📚 Knowledge Base: {len(documents)} документів про LangGraph 1.0\n")

# ============================================================================
# VECTOR STORE SETUP
# ============================================================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Топ-3 документи
)

print("✅ Vector store готовий (FAISS)\n")

# ============================================================================
# LLM SETUP
# ============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================================
# PYDANTIC MODELS для structured output
# ============================================================================

class SupervisorDecision(BaseModel):
    """Рішення supervisor агента про наступний крок"""
    next_agent: Literal["researcher", "analyzer", "synthesizer", "FINISH"] = Field(
        description="Який агент має виконувати наступний крок або FINISH"
    )
    reasoning: str = Field(description="Пояснення чому обрано цього агента")


class ResearchQuality(BaseModel):
    """Оцінка якості знайдених документів"""
    is_sufficient: bool = Field(description="Чи достатньо інформації знайдено")
    confidence: float = Field(description="Впевненість в якості (0.0-1.0)")
    reasoning: str = Field(description="Обґрунтування оцінки")


# ============================================================================
# AGENT NODES - Спеціалізовані агенти
# ============================================================================

def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """
    SupervisorAgent: Координує команду агентів

    Обов'язки:
    - Оцінює поточний прогрес
    - Вирішує який агент має виконувати наступний крок
    - Визначає коли завершити роботу
    """
    print("\n" + "="*70)
    print("🎯 SUPERVISOR AGENT: Приймає рішення про делегування")
    print("="*70)

    iteration = state.get("iteration_count", 0) + 1
    question = state["question"]

    # Аналізуємо поточний стан
    has_docs = bool(state.get("retrieved_docs"))
    has_analysis = bool(state.get("analysis"))
    has_answer = bool(state.get("final_answer"))

    print(f"📊 Iteration: {iteration}")
    print(f"📝 Question: {question}")
    print(f"📚 Docs retrieved: {has_docs}")
    print(f"🔍 Analysis done: {has_analysis}")
    print(f"✅ Answer ready: {has_answer}")

    # Створюємо промпт для supervisor
    supervisor_prompt = f"""You are a Supervisor Agent coordinating a multi-agent research team.

Current state:
- Question: {question}
- Documents retrieved: {has_docs}
- Analysis completed: {has_analysis}
- Answer ready: {has_answer}
- Iteration: {iteration}

Team members:
- researcher: Searches knowledge base for relevant information
- analyzer: Analyzes retrieved information and extracts insights
- synthesizer: Creates final comprehensive answer

Decide which agent should act next or if we should FINISH.

Rules:
1. Start with researcher if no docs retrieved
2. After researcher, use analyzer to process docs
3. After analyzer, use synthesizer for final answer
4. FINISH when synthesizer provides answer
5. Maximum 5 iterations

What's the next step?"""

    messages = [
        SystemMessage(content="You are an expert supervisor agent."),
        HumanMessage(content=supervisor_prompt)
    ]

    # Отримуємо structured decision
    structured_llm = llm.with_structured_output(SupervisorDecision)
    decision = structured_llm.invoke(messages)

    print(f"\n🎯 Decision: {decision.next_agent}")
    print(f"💭 Reasoning: {decision.reasoning}\n")

    return {
        **state,
        "current_agent": decision.next_agent,
        "supervisor_decision": decision.reasoning,
        "iteration_count": iteration,
        "messages": [AIMessage(content=f"Supervisor → {decision.next_agent}: {decision.reasoning}")]
    }


def researcher_node(state: MultiAgentState) -> MultiAgentState:
    """
    ResearcherAgent: RAG-пошук в knowledge base

    Обов'язки:
    - Шукає релевантні документи
    - Оцінює якість знайденої інформації
    """
    print("\n" + "="*70)
    print("🔍 RESEARCHER AGENT: Шукає інформацію")
    print("="*70)

    question = state["question"]

    # Виконуємо RAG retrieval
    retrieved_docs = retriever.invoke(question)

    print(f"📚 Знайдено {len(retrieved_docs)} документів")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. {doc.metadata.get('source', 'unknown')}: {doc.page_content[:100]}...")

    # Оцінюємо якість
    quality_prompt = f"""Evaluate if these documents are sufficient to answer the question.

Question: {question}

Documents:
{chr(10).join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)])}

Are these documents sufficient?"""

    structured_llm = llm.with_structured_output(ResearchQuality)
    quality = structured_llm.invoke([HumanMessage(content=quality_prompt)])

    print(f"\n✅ Quality: {'Sufficient' if quality.is_sufficient else 'Insufficient'}")
    print(f"🎯 Confidence: {quality.confidence:.2f}")
    print(f"💭 Reasoning: {quality.reasoning}\n")

    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "messages": [AIMessage(content=f"Researcher: Found {len(retrieved_docs)} docs (confidence: {quality.confidence:.2f})")]
    }


def analyzer_node(state: MultiAgentState) -> MultiAgentState:
    """
    AnalyzerAgent: Аналізує знайдену інформацію

    Обов'язки:
    - Аналізує документи
    - Витягує ключові insights
    - Структурує інформацію
    """
    print("\n" + "="*70)
    print("🔬 ANALYZER AGENT: Аналізує інформацію")
    print("="*70)

    question = state["question"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        print("⚠️  No documents to analyze")
        return {
            **state,
            "analysis": "No documents found for analysis",
            "messages": [AIMessage(content="Analyzer: No documents to analyze")]
        }

    # Аналізуємо документи
    analysis_prompt = f"""Analyze the following documents and extract key insights to answer the question.

Question: {question}

Documents:
{chr(10).join([f"{i+1}. Source: {doc.metadata.get('source', 'unknown')}\\n{doc.page_content}\\n" for i, doc in enumerate(docs)])}

Provide a structured analysis with:
1. Key concepts found
2. Relevant patterns/architectures
3. Best practices mentioned
4. Specific technical details"""

    messages = [
        SystemMessage(content="You are an expert technical analyst specializing in LangGraph and multi-agent systems."),
        HumanMessage(content=analysis_prompt)
    ]

    response = llm.invoke(messages)
    analysis = response.content

    print(f"📊 Analysis:\n{analysis[:300]}...\n")

    return {
        **state,
        "analysis": analysis,
        "messages": [AIMessage(content=f"Analyzer: Completed analysis ({len(analysis)} chars)")]
    }


def synthesizer_node(state: MultiAgentState) -> MultiAgentState:
    """
    SynthesizerAgent: Синтезує фінальну відповідь

    Обов'язки:
    - Комбінує analysis з retrieved docs
    - Створює вичерпну відповідь
    - Структурує результат для користувача
    """
    print("\n" + "="*70)
    print("🎨 SYNTHESIZER AGENT: Створює фінальну відповідь")
    print("="*70)

    question = state["question"]
    analysis = state.get("analysis", "")
    docs = state.get("retrieved_docs", [])

    # Синтезуємо фінальну відповідь
    synthesis_prompt = f"""Create a comprehensive, well-structured answer to the question based on the analysis.

Question: {question}

Analysis:
{analysis}

Source Documents:
{chr(10).join([f"- {doc.metadata.get('source', 'unknown')}" for doc in docs])}

Create a clear, informative answer that:
1. Directly addresses the question
2. Incorporates key insights from analysis
3. Provides specific technical details
4. Includes examples where relevant
5. Cites sources when appropriate"""

    messages = [
        SystemMessage(content="You are an expert technical writer creating clear, comprehensive answers."),
        HumanMessage(content=synthesis_prompt)
    ]

    response = llm.invoke(messages)
    final_answer = response.content

    print(f"✅ Final Answer:\n{final_answer[:300]}...\n")

    return {
        **state,
        "final_answer": final_answer,
        "messages": [AIMessage(content=f"Synthesizer: Created final answer ({len(final_answer)} chars)")]
    }


# ============================================================================
# ROUTING LOGIC - Умовна маршрутизація
# ============================================================================

def route_after_supervisor(state: MultiAgentState) -> str:
    """Визначає наступний крок після supervisor"""
    decision = state.get("current_agent", "researcher")

    if decision == "FINISH":
        return "end"
    elif decision == "researcher":
        return "researcher"
    elif decision == "analyzer":
        return "analyzer"
    elif decision == "synthesizer":
        return "synthesizer"
    else:
        return "end"


# ============================================================================
# GRAPH CONSTRUCTION - Мультиагентний workflow
# ============================================================================

def create_multiagent_system():
    """
    Створює мультиагентну систему з Supervisor Pattern

    Architecture:
    START → Supervisor → [Researcher → Analyzer → Synthesizer] → END
            ↑______________|
            (повертається до supervisor після кожного агента)
    """
    print("=" * 70)
    print("🏗️  СТВОРЕННЯ МУЛЬТИАГЕНТНОЇ СИСТЕМИ")
    print("=" * 70 + "\n")

    # Створюємо StateGraph
    workflow = StateGraph(MultiAgentState)

    # Додаємо агентів як nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Встановлюємо entry point
    workflow.set_entry_point("supervisor")

    # Conditional edges від supervisor до спеціалізованих агентів
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "researcher": "researcher",
            "analyzer": "analyzer",
            "synthesizer": "synthesizer",
            "end": END
        }
    )

    # Після кожного агента повертаємось до supervisor
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyzer", "supervisor")
    workflow.add_edge("synthesizer", "supervisor")

    # Компілюємо з checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    print("✅ Мультиагентна система створена\n")
    print("Agents:")
    print("  🎯 Supervisor - координує команду")
    print("  🔍 Researcher - RAG-пошук")
    print("  🔬 Analyzer - аналіз інформації")
    print("  🎨 Synthesizer - синтез відповіді\n")

    return app


# ============================================================================
# TESTING - Тестування мультиагентної системи
# ============================================================================

def test_multiagent_system():
    """Тестує мультиагентну систему з різними запитами"""

    app = create_multiagent_system()

    test_queries = [
        "What is the Supervisor Pattern in LangGraph 1.0 and how does it work?",
        "Explain LangGraph StateGraph API and checkpointing mechanisms",
        "How do multi-agent coordination patterns work in LangGraph 1.0?",
    ]

    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {query}")
        print("=" * 70)

        # Ініціалізуємо state
        initial_state = {
            "messages": [],
            "question": query,
            "current_agent": "supervisor",
            "retrieved_docs": [],
            "analysis": "",
            "final_answer": "",
            "supervisor_decision": "",
            "iteration_count": 0
        }

        # Виконуємо з checkpointing
        config = {"configurable": {"thread_id": f"test_{i}"}}

        try:
            # Stream результатів
            for event in app.stream(initial_state, config):
                agent_name = list(event.keys())[0]
                print(f"\n📍 Event from: {agent_name}")

            # Отримуємо фінальний state
            final_state = app.get_state(config)

            print("\n" + "-" * 70)
            print("📊 FINAL RESULT")
            print("-" * 70)
            print(f"\n🎯 Question: {query}\n")
            print(f"✅ Answer:\n{final_state.values.get('final_answer', 'No answer')}\n")
            print(f"📈 Stats:")
            print(f"  - Iterations: {final_state.values.get('iteration_count', 0)}")
            print(f"  - Documents used: {len(final_state.values.get('retrieved_docs', []))}")
            print(f"  - Messages exchanged: {len(final_state.values.get('messages', []))}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        if i < len(test_queries):
            input("\n⏸️  Press Enter to continue to next test...\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎯 LangGraph 1.0 - Multi-Agent System (Supervisor Pattern)")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✅ Supervisor Pattern - hierarchical coordination")
    print("  ✅ 4 Specialized Agents (Supervisor, Researcher, Analyzer, Synthesizer)")
    print("  ✅ RAG Integration - knowledge base search")
    print("  ✅ StateGraph - centralized state management")
    print("  ✅ Checkpointing - persistent state with MemorySaver")
    print("  ✅ Conditional Routing - dynamic agent selection")
    print("  ✅ LangSmith Tracing - full observability")
    print("  ✅ Knowledge Base - LangGraph 1.0 documentation")
    print()
    print("=" * 70 + "\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found!")
        exit(1)

    try:
        test_multiagent_system()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 70)
        print("\n💡 Check LangSmith dashboard for full trace!")
        print("   https://smith.langchain.com/\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
