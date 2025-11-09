# LangChain v1.0 та LangGraph 1.0 - Демонстраційні Скрипти

Цей репозиторій містить комплексні приклади використання ключових нововведень **LangChain v1.0** та **LangGraph 1.0**.

## 📋 Зміст

- [Встановлення](#встановлення)
- [Налаштування](#налаштування)
- [LangChain v1.0 Скрипти](#langchain-v10-скрипти)
- [LangGraph 1.0 Скрипти](#langgraph-10-скрипти)
- [Ключові Нововведення](#ключові-нововведення)
- [Використання](#використання)

## 🚀 Встановлення

```bash
# Клонування репозиторію
git clone <repository-url>
cd module5

# Створення віртуального середовища
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Встановлення залежностей
pip install -r requirements.txt
```

## ⚙️ Налаштування

1. Скопіюйте файл `.env.example` в `.env`:
```bash
cp .env.example .env
```

2. Додайте ваші API ключі в `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langchain-langgraph-v1-demo
```

## 📚 LangChain v1.0 Скрипти

### 1. `langchain_v1_lcel.py` - LCEL (LangChain Expression Language)

**Ключові можливості:**
- ✅ Композиція ланцюгів з оператором `|`
- ✅ Паралельне виконання ланцюгів
- ✅ RunnablePassthrough для передачі контексту
- ✅ Стрімінг відповідей
- ✅ Пакетна обробка

**Приклад використання:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Композиція з оператором |
prompt = ChatPromptTemplate.from_template("Розкажи про {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt | model | output_parser
result = chain.invoke({"topic": "LangChain"})
```

**Запуск:**
```bash
python langchain_v1_lcel.py
```

---

### 2. `langchain_v1_structured_output.py` - Structured Output

**Ключові можливості:**
- ✅ Pydantic моделі для структурованих даних
- ✅ PydanticOutputParser
- ✅ with_structured_output() метод (function calling)
- ✅ Валідація та типізація
- ✅ Пакетна обробка структурованих даних

**Приклад використання:**
```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Person(BaseModel):
    name: str = Field(description="Ім'я персони")
    age: int = Field(description="Вік персони")
    occupation: str = Field(description="Професія")

model = ChatOpenAI(model="gpt-3.5-turbo")
structured_llm = model.with_structured_output(Person)

result = structured_llm.invoke("Марія, 28 років, UX дизайнер")
print(f"{result.name}, {result.age} років, {result.occupation}")
```

**Запуск:**
```bash
python langchain_v1_structured_output.py
```

---

### 3. `langchain_v1_rag.py` - RAG (Retrieval-Augmented Generation)

**Ключові можливості:**
- ✅ LCEL для RAG pipeline
- ✅ Векторні сховища (FAISS)
- ✅ RAG з джерелами
- ✅ Фільтрація по метаданим
- ✅ Multi-Query RAG
- ✅ Стрімінг RAG відповідей

**Приклад використання:**
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Створення RAG ланцюга
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("Що таке LCEL?")
```

**Запуск:**
```bash
python langchain_v1_rag.py
```

---

## 🔄 LangGraph 1.0 Скрипти

### 4. `langgraph_v1_basics.py` - Основи LangGraph

**Ключові можливості:**
- ✅ Створення State Graphs
- ✅ Умовні переходи (conditional edges)
- ✅ Цикли в графах
- ✅ Message Graphs для чат-ботів
- ✅ Багатокрокове міркування

**Приклад використання:**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    counter: int

def increment(state: State) -> State:
    return {"counter": state["counter"] + 1}

workflow = StateGraph(State)
workflow.add_node("increment", increment)
workflow.set_entry_point("increment")
workflow.add_edge("increment", END)

app = workflow.compile()
result = app.invoke({"counter": 0})
```

**Запуск:**
```bash
python langgraph_v1_basics.py
```

---

### 5. `langgraph_v1_agents.py` - Агенти з інструментами

**Ключові можливості:**
- ✅ Агенти з інструментами (tools)
- ✅ ToolNode для виконання інструментів
- ✅ Агенти з пам'яттю
- ✅ ReAct паттерн (міркування та дії)
- ✅ Supervisor агенти

**Приклад використання:**
```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def calculator(expression: str) -> str:
    """Виконує математичні обчислення."""
    return str(eval(expression))

tools = [calculator]
model = ChatOpenAI(model="gpt-3.5-turbo").bind_tools(tools)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
# ... додавання ребер
```

**Запуск:**
```bash
python langgraph_v1_agents.py
```

---

### 6. `langgraph_v1_persistence.py` - Persistence та Checkpointing

**Ключові можливості:**
- ✅ MemorySaver для збереження стану
- ✅ Checkpointing між викликами
- ✅ Thread-based conversations
- ✅ Відновлення стану
- ✅ Історія checkpoints

**Приклад використання:**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Конфігурація з thread_id
config = {"configurable": {"thread_id": "conversation_1"}}

# Перша взаємодія
result1 = app.invoke(input1, config)

# Друга взаємодія - зберігається контекст
result2 = app.invoke(input2, config)
```

**Запуск:**
```bash
python langgraph_v1_persistence.py
```

---

### 7. `langgraph_v1_human_in_loop.py` - Human-in-the-Loop

**Ключові можливості:**
- ✅ Workflow з підтвердженням людини
- ✅ Переривання для вводу
- ✅ Умовне схвалення
- ✅ Цикли зворотного зв'язку
- ✅ Інтерактивні агенти

**Приклад використання:**
```python
def should_get_approval(state: State) -> str:
    if state["requires_approval"]:
        return "approval"
    return "execute"

workflow.add_conditional_edges(
    "agent",
    should_get_approval,
    {
        "approval": "human_approval_node",
        "execute": "execute_node"
    }
)
```

**Запуск:**
```bash
python langgraph_v1_human_in_loop.py
```

---

## 🎯 Ключові Нововведення

### LangChain v1.0

| Нововведення | Опис | Скрипт |
|--------------|------|--------|
| **LCEL** | Новий спосіб композиції з оператором `\|` | `langchain_v1_lcel.py` |
| **Runnable Interface** | Уніфікований інтерфейс (invoke, stream, batch) | Всі скрипти |
| **Structured Output** | Pydantic моделі для типізованих даних | `langchain_v1_structured_output.py` |
| **Parallel Chains** | RunnableParallel для паралельного виконання | `langchain_v1_lcel.py` |
| **Streaming** | Покращена підтримка стрімінгу | `langchain_v1_rag.py` |
| **Function Calling** | with_structured_output() | `langchain_v1_structured_output.py` |

### LangGraph 1.0

| Нововведення | Опис | Скрипт |
|--------------|------|--------|
| **State Graphs** | Побудова графів станів для агентів | `langgraph_v1_basics.py` |
| **Conditional Edges** | Умовні переходи між вузлами | `langgraph_v1_basics.py` |
| **Cycles** | Підтримка циклів в графах | `langgraph_v1_basics.py` |
| **Checkpointing** | Збереження та відновлення стану | `langgraph_v1_persistence.py` |
| **Human-in-Loop** | Інтеграція людини в процес | `langgraph_v1_human_in_loop.py` |
| **ToolNode** | Спрощене виконання інструментів | `langgraph_v1_agents.py` |
| **Memory Saver** | Збереження історії розмов | `langgraph_v1_persistence.py` |

## 💡 Використання

### Запуск окремого скрипта:
```bash
python langchain_v1_lcel.py
```

### Запуск всіх LangChain демо:
```bash
python langchain_v1_lcel.py
python langchain_v1_structured_output.py
python langchain_v1_rag.py
```

### Запуск всіх LangGraph демо:
```bash
python langgraph_v1_basics.py
python langgraph_v1_agents.py
python langgraph_v1_persistence.py
python langgraph_v1_human_in_loop.py
```

## 📖 Додаткові Ресурси

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/)

## 🤝 Внесок

Ці скрипти створені для навчальних цілей та демонстрації можливостей LangChain v1.0 та LangGraph 1.0.

## 📝 Ліцензія

MIT License

---

**Автор:** Claude Agent
**Дата:** 2024
**Версія:** 1.0
