# LangChain 1.0 & LangGraph 1.0 Agents

Production-ready agent implementations showcasing key innovations in LangChain 1.0 and LangGraph 1.0 (2025 release).

## Overview

This repository contains four comprehensive agent examples demonstrating the latest LangChain/LangGraph patterns:

1. **Basic Agent** - Agent using `create_agent` API (LangChain 1.0)
2. **Middleware Agent** - Agent with before/after/modify hooks
3. **RAG Agent** - Agentic RAG with LangGraph StateGraph and checkpointing
4. **Multi-Agent System** - Supervisor Pattern with coordinated specialized agents

All agents include **LangSmith tracing** for full observability.

## Features

### LangChain 1.0
-  `create_agent()` - New unified agent creation API
-  Callbacks architecture with BaseCallbackHandler
-  Stable APIs (no breaking changes until 2.0)
-  Production-ready patterns

### LangGraph 1.0
-  StateGraph for complex orchestration
-  Checkpointing with MemorySaver
-  Conditional routing and loops
-  Thread-based conversation persistence
-  Supervisor Pattern for multi-agent coordination
-  Hierarchical agent teams
-  Shared state management across agents

### LangSmith Integration
-  Automatic tracing for all agents
-  Cost tracking and latency analysis
-  Trace visualization for debugging

## Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

**ВАЖЛИВО:** Використовуйте Python 3.11 або новіший (не 3.14, оскільки є проблеми сумісності з Pydantic).

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required variables:
```bash
# OpenAI (Required)
OPENAI_API_KEY=sk-your-key          # Get from https://platform.openai.com/api-keys

# LangSmith (Optional - for tracing)
LANGCHAIN_TRACING_V2=true            # Enable LangSmith tracing
LANGCHAIN_API_KEY=ls__your-key       # Get from https://smith.langchain.com
LANGCHAIN_PROJECT=langchain-v1-agents

# Real API Integrations (Required for agent tools)
OPENWEATHERMAP_API_KEY=your-key     # Get from https://openweathermap.org/api
TAVILY_API_KEY=tvly-your-key        # Get from https://tavily.com
```

## Agents

### 1. Basic Agent (01_basic_agent.py)

**Purpose:** Demonstrates the LangChain 1.0 `create_agent` API with multiple tools.

**Pattern:** Automatic Agent Orchestration
```
User Query  Agent  Tool Selection  Tool Execution  Response
```

**Tools:**
- `get_weather` - Real weather data via OpenWeatherMap API
- `web_search` - Real web search via Tavily Search API
- `calculate` - Safe calculations using numexpr
- `get_stock_price` - Real stock prices via yfinance API

**Run:**
```bash
python 01_basic_agent.py
```

**Key Features:**
- Uses `create_agent` (LangChain 1.0 API - October 2025)
- Automatic prompt optimization
- Error handling with `handle_parsing_errors=True`
- Max iterations limit for safety
- Returns intermediate steps for debugging
- Full LangSmith trace integration

**Example Output:**
```
Query: "What's the weather in Kyiv and calculate 15% tip on 250 UAH?"

Thought: I need to get weather first, then calculate the tip
Action: get_weather
Action Input: Kyiv
Observation: Partly cloudy, 18°C

Thought: Now I'll calculate the tip
Action: calculate
Action Input: 250 * 0.15
Observation: 37.5

Final Answer: Weather in Kyiv is partly cloudy at 18°C.
A 15% tip on 250 UAH is 37.50 UAH.
```

---

### 2. Middleware Agent (02_agent_with_middleware.py)

**Purpose:** Demonstrates the new middleware architecture introduced in LangChain 1.0 (2025).

**Pattern:** Agent with Middleware Hooks
```
Request  before_model  [modify_model_request]  LLM Call  after_model  Response
```

**Middleware Implementations:**

1. **LoggingMiddleware** - Observability
   - `before_model`: Log input state, token count
   - `after_model`: Log output, execution time

2. **SecurityMiddleware** - Safety controls
   - `modify_model_request`: Block high-risk tools (execute_trade, send_notification)
   - Prevent unauthorized actions

3. **TokenLimitMiddleware** - Cost control
   - `before_model`: Check token usage
   - Throttle requests if limit exceeded

**Run:**
```bash
python 02_agent_with_middleware.py
```

**Key Features:**
- Custom `MiddlewareAgentExecutor` class
- Three middleware hooks: before_model, after_model, modify_model_request
- High-risk tool blocking for security
- Token usage tracking for cost control
- Full execution logging

**Example Output:**
```
Query: "Execute a trade for 1000 BTC"

[LoggingMiddleware] Before model call...
[SecurityMiddleware] Checking tools for security risks...
[SecurityMiddleware] ⚠️  Blocking high-risk tool: execute_trade
[TokenLimitMiddleware] Token usage: 150/1000

Agent Response: "I cannot execute trades as the execute_trade tool
has been disabled for security reasons."

[LoggingMiddleware] After model call (1.2s, 45 tokens)
```

---

### 3. RAG Agent (03_rag_agent_langgraph.py)

**Purpose:** Implements Agentic RAG pattern using LangGraph with dynamic retrieval strategies.

**Pattern:** Agentic RAG with Query Rewriting
```
User Query  Retrieve Docs  Grade Relevance
     (if relevant)
Generate Answer  END
     (if irrelevant)
Rewrite Query  Retrieve Again  Grade  ...
```

**Graph Structure:**
```
START  retrieve  grade  [decide]
                    
           if relevant: generate  END
           if irrelevant: rewrite  retrieve (loop)
```

**Run:**
```bash
python 03_rag_agent_langgraph.py
```

**Key Features:**
- LangGraph StateGraph for orchestration
- FAISS vector store with 6 documents about LangChain/LangGraph
- Dynamic relevance grading using LLM
- Automatic query rewriting if docs irrelevant
- MemorySaver checkpointing (thread_id for separate sessions)
- Conditional routing based on relevance
- Max rewrite limit to prevent infinite loops

**Knowledge Base Topics:**
- LangChain 1.0 Release features
- LangGraph checkpointing (MemorySaver, PostgresSaver)
- Agent middleware architecture
- Agentic RAG patterns
- LangSmith integration
- Production best practices

**Example Flow:**

```
Test 1: "What are the new middleware hooks in LangChain 1.0?"

NODE: Retrieve Documents
  Retrieved 2 documents: middleware, langchain

NODE: Grade Document Relevance
  Grade: RELEVANT
  Reasoning: Documents contain info about middleware hooks

ROUTING: Going to generate_answer

NODE: Generate Answer
  Answer: LangChain 1.0 introduces three middleware hooks:
  - before_model: Pre-processing before LLM calls
  - after_model: Post-processing after responses
  - modify_model_request: Modify tools, prompts, messages
  Built-in middlewares include Human-in-the-loop,
  Summarization, and PII redaction.

Stats:
  - Query rewrites: 0
  - Documents used: 2
  - Final grade: relevant
```

**Checkpointing Example:**
```python
# Each query uses unique thread_id
config = {"configurable": {"thread_id": "conversation_1"}}

# Resume from checkpoint
result = agent.invoke(state, config)

# Different conversation
config2 = {"configurable": {"thread_id": "conversation_2"}}
```

---

### 4. Multi-Agent System (04_multiagent_langgraph.py)

**Purpose:** Implements Supervisor Pattern with hierarchical multi-agent coordination using LangGraph 1.0.

**Pattern:** Supervisor coordinating specialized agents
```
User Query  Supervisor  [Researcher  Supervisor  Analyzer  Supervisor  Synthesizer]  END
```

**Architecture:**
```
START  Supervisor
          (delegates to specialist)
    [Researcher | Analyzer | Synthesizer]
          (returns control)
      Supervisor (decides next step)
         
      END (when complete)
```

**Run:**
```bash
python 04_multiagent_langgraph.py
```

**Specialized Agents:**

1. **SupervisorAgent** 
   - Coordinates team of specialists
   - Makes delegation decisions
   - Evaluates progress and determines completion
   - Uses structured output for routing logic

2. **ResearcherAgent** 
   - RAG-based knowledge retrieval
   - Searches LangGraph 1.0 documentation
   - Evaluates quality of retrieved documents
   - Returns top-k relevant sources

3. **AnalyzerAgent** 
   - Analyzes retrieved information
   - Extracts key insights and patterns
   - Structures technical details
   - Identifies best practices

4. **SynthesizerAgent** 
   - Creates comprehensive final answer
   - Combines analysis with source docs
   - Formats response for readability
   - Cites sources appropriately

**Key Features:**
- Supervisor Pattern from LangGraph 1.0 (2025)
- Hierarchical multi-agent coordination
- Shared StateGraph across all agents
- Conditional routing based on supervisor decisions
- MemorySaver checkpointing for workflow persistence
- RAG integration with 8-document knowledge base
- LangSmith tracing for all agent interactions
- Structured decision-making with Pydantic models

**Knowledge Base (LangGraph 1.0 Topics):**
- Supervisor Pattern architecture
- StateGraph API and features
- Checkpointing mechanisms (MemorySaver, PostgresSaver)
- Multi-agent coordination patterns
- LangGraph Swarm (2025 release)
- LangGraph Server & persistence
- Error handling & recovery strategies
- Best practices for multi-agent systems

**Example Flow:**

```
Query: "What is the Supervisor Pattern in LangGraph 1.0?"

 SUPERVISOR: Decides to delegate to researcher
   Decision: researcher
   Reasoning: Need to find information first

 RESEARCHER: Searches knowledge base
   Found 3 documents about Supervisor Pattern
   Quality: Sufficient (confidence: 0.95)

 SUPERVISOR: Delegates to analyzer
   Decision: analyzer
   Reasoning: Documents found, need analysis

 ANALYZER: Analyzes documents
   Extracted: Pattern definition, architecture, use cases
   Key concepts: hierarchical coordination, specialized agents

 SUPERVISOR: Delegates to synthesizer
   Decision: synthesizer
   Reasoning: Analysis ready, create final answer

 SYNTHESIZER: Creates comprehensive answer
   Combined analysis with source citations
   Formatted with examples and technical details
   Answer: [Detailed explanation of Supervisor Pattern]

 SUPERVISOR: Work complete
   Decision: FINISH

Stats:
  - Iterations: 4
  - Agents invoked: 3 specialists
  - Documents retrieved: 3
  - Messages exchanged: 7
```

**State Schema:**
```python
class MultiAgentState(TypedDict):
    messages: Annotated[List, operator.add]  # Agent communication
    question: str  # User query
    current_agent: str  # Active agent
    retrieved_docs: List[Document]  # From researcher
    analysis: str  # From analyzer
    final_answer: str  # From synthesizer
    supervisor_decision: str  # Reasoning
    iteration_count: int  # Progress tracking
```

**Workflow Diagram:**
```
                Supervisor
                    |
        +-----------+-----------+
        |           |           |
   Researcher   Analyzer   Synthesizer
        |           |           |
        +-----------+-----------+
                    |
                    v
            (back to Supervisor)
```

**Checkpointing:**
```python
# Each workflow run uses unique thread
config = {"configurable": {"thread_id": "workflow_1"}}

# Execute multi-agent workflow
for event in app.stream(initial_state, config):
    print(f"Agent: {list(event.keys())[0]}")

# Retrieve final state
final_state = app.get_state(config)
answer = final_state.values['final_answer']
```

---

## LangSmith Tracing

All agents automatically send traces to LangSmith when `LANGCHAIN_TRACING_V2=true`.

**View traces at:** https://smith.langchain.com

**What's traced:**
- Agent reasoning steps (Thought  Action  Observation)
- Tool calls and results
- LLM prompts and completions
- Token usage and costs
- Latency for each operation
- RAG retrieval and grading steps

**Example trace hierarchy:**
```
AgentExecutor
├- LLM Call (Reasoning)
├- Tool: get_weather
|  +- Result: "Partly cloudy, 18°C"
├- LLM Call (Next action)
├- Tool: calculate
|  +- Result: "37.5"
+- LLM Call (Final answer)
```

## Architecture Patterns

### LangChain 1.0 Pattern (Agent 1)
```python
agent = create_agent(
    model="gpt-4o-mini",  # model as string, not ChatOpenAI object
    tools=tools,
    system_prompt="You are a helpful assistant"
)
# Direct invocation - no AgentExecutor needed
result = agent.invoke({
    "messages": [{"role": "user", "content": "Your question here"}]
})
```

### Middleware Pattern (Agent 2)
```python
class LoggingMiddleware:
    def before_model(self, state): ...
    def after_model(self, state, result): ...

# Wrapper around create_agent with middleware
agent = MiddlewareAgent(
    model="gpt-4o-mini",
    tools=tools,
    system_prompt="You are a helpful assistant",
    middlewares=[LoggingMiddleware(), SecurityMiddleware()]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "question"}]
})
```

### Agentic RAG Pattern (Agent 3)
```python
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_conditional_edges("grade", decide_next_step, {...})

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```

### Multi-Agent Supervisor Pattern (Agent 4)
```python
# Define shared state for all agents
class MultiAgentState(TypedDict):
    messages: Annotated[List, operator.add]
    question: str
    current_agent: str
    retrieved_docs: List[Document]
    analysis: str
    final_answer: str
    supervisor_decision: str
    iteration_count: int

# Create StateGraph with all agents
workflow = StateGraph(MultiAgentState)

# Add specialized agent nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)  # RAG search
workflow.add_node("analyzer", analyzer_node)      # Analysis
workflow.add_node("synthesizer", synthesizer_node) # Final answer

# Entry point: supervisor
workflow.set_entry_point("supervisor")

# Conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_after_supervisor,  # Decision function
    {
        "researcher": "researcher",
        "analyzer": "analyzer",
        "synthesizer": "synthesizer",
        "end": END
    }
)

# Return control to supervisor after each agent
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyzer", "supervisor")
workflow.add_edge("synthesizer", "supervisor")

# Compile with checkpointer
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Execute with thread-based isolation
config = {"configurable": {"thread_id": "workflow_1"}}
for event in app.stream(initial_state, config):
    agent_name = list(event.keys())[0]
    print(f"Current agent: {agent_name}")
```

## Production Considerations

### Error Handling
- All agents use `handle_parsing_errors=True`
- Max iterations limit prevents infinite loops
- Try-except blocks for graceful degradation

### Security
- Middleware can block dangerous tools
- Input validation via Pydantic models
- No execution of arbitrary code

### Cost Control
- Token limit middleware for budget management
- Efficient retrieval with `k=2` top documents
- Model selection: `gpt-4o-mini` for cost efficiency

### Observability
- LangSmith tracing for all operations
- Logging middleware for debugging
- Structured outputs for reliability

### Scalability
- Stateless agent design (except RAG checkpointing)
- Async support available (use `ainvoke`)
- Thread-based isolation for concurrent sessions

## Troubleshooting

**Issue:** `ImportError: cannot import name 'create_agent'`
- **Fix:** Ensure `langchain>=1.0.0` is installed (October 2025 release)

**Issue:** LangSmith traces not appearing
- **Fix:** Check `.env` has `LANGCHAIN_TRACING_V2=true` and valid `LANGCHAIN_API_KEY`

**Issue:** RAG agent not finding relevant documents
- **Fix:** Check FAISS vector store creation, try different queries, inspect retrieved docs

**Issue:** Middleware not being called
- **Fix:** Ensure using `MiddlewareAgentExecutor`, not standard `AgentExecutor`

## Migration from v0.x

If migrating from LangChain 0.x:

**Old (Deprecated):**
```python
from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

**New (v1.0):**
```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",  # model name as string
    tools=tools,
    system_prompt="You are a helpful assistant"
)

# Direct invocation - no AgentExecutor needed
result = agent.invoke({
    "messages": [{"role": "user", "content": "question"}]
})
```

## Resources

- **LangChain Docs:** https://docs.langchain.com/oss/python/langchain/agents
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangSmith:** https://smith.langchain.com
- **LangChain 1.0 Release Notes:** https://blog.langchain.dev/langchain-v1-0/

## Changelog

### 2025-11-13
- Замінено всі моки на реальні API інтеграції
  - OpenWeatherMap API для погоди
  - Tavily Search API для веб-пошуку
  - yfinance для біржових даних
- Виправлено middleware на офіційний LangChain 1.0 API
- Додано безпечні обчислення через numexpr (замість eval)
- Оновлено requirements.txt та .env.example

### 2025-11-09
- Створено 4 production-ready агенти на LangChain 1.0 / LangGraph 1.0
- Впроваджено офіційний `create_agent` API
- Додано middleware архітектуру
- Реалізовано Agentic RAG з StateGraph
- Додано Supervisor Pattern для multi-agent системи

## License

MIT

## Contributing

This is a demonstration repository for educational purposes. All agents use production-ready patterns from LangChain 1.0 and LangGraph 1.0 (2025 release).
