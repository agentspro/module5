# 🚀 LangChain/LangGraph Workshop - Cheat Sheet

## 📦 Quick Setup

```bash
# Встановлення
pip install langchain langchain-openai langgraph langchain-community faiss-cpu

# .env файл
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
```

---

## 🔗 LCEL - Основний синтаксис

### Базовий ланцюг
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = prompt | model | StrOutputParser()
result = chain.invoke({"variable": "value"})
```

### Паралельне виконання
```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    task1=chain1,
    task2=chain2
)
results = parallel.invoke(input)  # {"task1": ..., "task2": ...}
```

### Streaming
```python
# Sync
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)

# Async
async for chunk in chain.astream(input):
    print(chunk, end="", flush=True)
```

### Batch
```python
inputs = [{"x": 1}, {"x": 2}, {"x": 3}]
results = chain.batch(inputs)
```

---

## 🤖 Агенти

### Створення агента (v1.0)
```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """Tool description for LLM"""
    return "result"

agent = create_react_agent(model, [my_tool])
result = agent.invoke({"messages": [("user", "query")]})
```

### Custom System Message
```python
agent = create_react_agent(
    model,
    tools,
    state_modifier="Your custom system prompt here"
)
```

---

## 🔄 LangGraph

### Базовий StateGraph
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    field: str

def node_func(state: State) -> State:
    return {"field": "new_value"}

workflow = StateGraph(State)
workflow.add_node("node_name", node_func)
workflow.set_entry_point("node_name")
workflow.add_edge("node_name", END)

app = workflow.compile()
result = app.invoke({"field": "initial"})
```

### Умовні переходи
```python
def router(state: State) -> str:
    if state["field"] == "value":
        return "path_a"
    return "path_b"

workflow.add_conditional_edges(
    "source_node",
    router,
    {
        "path_a": "node_a",
        "path_b": "node_b"
    }
)
```

### Checkpointing (Persistence)
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(input, config)
```

### Цикли
```python
def should_continue(state: State) -> str:
    if state["counter"] < 5:
        return "continue"
    return "end"

workflow.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # Повернення до себе
        "end": END
    }
)
```

---

## 🎭 Multi-Agent Patterns

### Supervisor Pattern
```python
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: list
    next_agent: str

def supervisor(state):
    # Логіка вибору агента
    return {"next_agent": "researcher"}

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_agent"],
    {
        "researcher": "researcher",
        "writer": "writer"
    }
)
```

---

## 🛠️ Tools Best Practices

### Хороший tool
```python
@tool
def good_tool(param: str) -> str:
    """
    Чітке описання що робить tool.

    Args:
        param: Що означає цей параметр

    Returns:
        Що повертає

    Use when: Коли використовувати цей tool
    """
    try:
        result = do_something(param)
        return f"Success: {result}"
    except Exception as e:
        return f"Error: {e}"
```

### Tool з Pydantic
```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=5, description="Max results")

@tool(args_schema=SearchInput)
def search(query: str, limit: int = 5) -> str:
    """Search the web"""
    pass
```

---

## 📊 Моніторинг

### LangSmith
```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=project_name
```

### Custom Callbacks
```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished")

model = ChatOpenAI(callbacks=[MyCallback()])
```

---

## 💾 RAG Pattern

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Створення vector store
docs = [Document(page_content="text", metadata={"source": "doc1"})]
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG ланцюг
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What is X?")
```

---

## 🐛 Debugging Tips

### Інспекція messages
```python
result = agent.invoke(input)
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content[:100]}")
```

### Langraph visualize
```python
from langraph.graph import StateGraph
from IPython.display import Image, display

app = workflow.compile()
display(Image(app.get_graph().draw_mermaid_png()))
```

### Exception handling
```python
@tool
def safe_tool(input: str) -> str:
    """Safe tool with error handling"""
    try:
        return risky_operation(input)
    except ValueError as e:
        return f"Invalid input: {e}"
    except Exception as e:
        return f"Tool error: {e}"
```

---

## ⚡ Performance Tips

1. **Використовуйте cache**
```python
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()
```

2. **Batch де можливо**
```python
results = chain.batch(many_inputs)  # Краще ніж loop
```

3. **Правильні моделі**
- GPT-3.5: швидко + дешево
- GPT-4: точно + дорого
- Reasoning = GPT-4, Execution = GPT-3.5

4. **Token limits**
```python
model = ChatOpenAI(max_tokens=500)  # Контролюйте витрати!
```

---

## 🔗 Корисні посилання

- **Docs:** https://python.langchain.com/
- **LangGraph:** https://langchain-ai.github.io/langgraph/
- **Templates:** https://python.langchain.com/docs/templates
- **Discord:** https://discord.gg/langchain
- **GitHub:** https://github.com/langchain-ai/langchain

---

## 🚨 Typical Errors & Fixes

### "No API key found"
```python
# Перевірте .env файл або:
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### "Module not found"
```bash
pip install langchain-openai  # Не просто langchain!
```

### "Rate limit exceeded"
```python
# Додайте retry
from langchain.llms import OpenAI
llm = OpenAI(max_retries=3, request_timeout=60)
```

### "Memory too large"
```python
# Обрізайте історію
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=5)  # Останні 5 messages
```

---

## 📝 Quick Commands Reference

```python
# Invoke (one input)
result = chain.invoke(input)

# Stream (see tokens as they arrive)
for chunk in chain.stream(input):
    pass

# Batch (multiple inputs)
results = chain.batch([input1, input2, input3])

# Async versions
await chain.ainvoke(input)
async for chunk in chain.astream(input):
    pass
await chain.abatch(inputs)
```

---

**💡 Remember:**
- Pin versions in requirements.txt!
- Always handle errors in tools
- Use type hints everywhere
- Test with small models first
- Monitor costs in production

**🎯 Practice flow:**
Simple chain → RAG → Agent with tools → LangGraph → Multi-agent

---

_Workshop by [Your Name] | 2024_
