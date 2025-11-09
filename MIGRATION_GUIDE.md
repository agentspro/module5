# 🚀 LangChain v0.x → v1.0 та LangGraph: Гід з Міграції

## 📖 Зміст

- [Чому v1.0?](#чому-v10)
- [Ключові зміни](#ключові-зміни)
- [Міграція ланцюгів](#міграція-ланцюгів)
- [Міграція агентів](#міграція-агентів)
- [Реальний приклад](#реальний-приклад)
- [Чек-лист міграції](#чек-лист-міграції)

---

## 🎯 Чому v1.0?

### Проблеми v0.x

1. **Складна композиція**
   ```python
   # v0.x - багато boilerplate
   chain1 = LLMChain(llm=llm, prompt=prompt1)
   chain2 = LLMChain(llm=llm, prompt=prompt2)
   overall_chain = SimpleSequentialChain(chains=[chain1, chain2])
   ```

2. **Різні API для різних операцій**
   - `.run()` vs `.predict()` vs `__call__()`
   - Не всі компоненти підтримували streaming
   - Batch обробка була незручною

3. **Агенти як чорна скринька**
   - `AgentExecutor` - складно кастомізувати
   - Важко дебажити
   - Обмежений контроль над логікою

### Рішення v1.0

1. **LCEL - інтуїтивна композиція**
   ```python
   # v1.0 - просто і зрозуміло
   chain = prompt | model | output_parser
   ```

2. **Єдиний Runnable інтерфейс**
   - `.invoke()` - для одного входу
   - `.stream()` - для streaming
   - `.batch()` - для багатьох входів
   - Async варіанти: `.ainvoke()`, `.astream()`, `.abatch()`

3. **LangGraph - явні state machines**
   - Повний контроль над логікою
   - Візуалізація потоку виконання
   - Checkpointing з коробки

---

## 🔄 Ключові зміни

### 1. Композиція ланцюгів

| Аспект | v0.x | v1.0 (LCEL) |
|--------|------|-------------|
| **Синтаксис** | `LLMChain(llm, prompt)` | `prompt \| model \| parser` |
| **Читабельність** | ⭐⭐ Verbose | ⭐⭐⭐⭐⭐ Чисто |
| **Послідовність** | `SequentialChain` | Просто додати `\|` |
| **Паралельність** | `asyncio.gather()` | `RunnableParallel()` |

### 2. Виклик методів

```python
# v0.x - різні методи
result = chain.run(input)
result = chain.predict(input)
result = chain(input)

# v1.0 - єдиний інтерфейс
result = chain.invoke(input)      # sync
result = await chain.ainvoke(input)  # async
for chunk in chain.stream(input):    # streaming
results = chain.batch(inputs)        # batch
```

### 3. Побудова агентів

| Аспект | Без LangGraph | З LangGraph |
|--------|---------------|-------------|
| **Підхід** | Імперативний | Декларативний |
| **Візуалізація** | ❌ Немає | ✅ Граф |
| **Контроль** | ⭐⭐ Обмежений | ⭐⭐⭐⭐⭐ Повний |
| **Дебаг** | ❌ Складно | ✅ Легко |
| **Пам'ять** | Manual Memory | Checkpointing |
| **Цикли** | ❌ Обмежені | ✅ Повна підтримка |

---

## 📦 Міграція ланцюгів

### Простий ланцюг

**До (v0.x):**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Розкажи про {topic}"
)
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="Python")
```

**Після (v1.0):**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Розкажи про {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | model | StrOutputParser()
result = chain.invoke({"topic": "Python"})
```

**Переваги:**
- ✅ Менше коду
- ✅ Інтуїтивний pipe оператор
- ✅ Єдиний `.invoke()`

### RAG ланцюг

**До (v0.x):**
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
result = qa_chain({"query": "What is LCEL?"})
```

**Після (v1.0):**
```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
result = rag_chain.invoke("What is LCEL?")
```

**Переваги:**
- ✅ Візуально видно потік даних
- ✅ Легко кастомізувати
- ✅ Підтримка streaming

### Паралельні ланцюги

**До (v0.x):**
```python
import asyncio

async def run_parallel():
    results = await asyncio.gather(
        chain1.arun(input1),
        chain2.arun(input2),
        chain3.arun(input3)
    )
    return results
```

**Після (v1.0):**
```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    result1=chain1,
    result2=chain2,
    result3=chain3
)
results = parallel.invoke(input)
# {"result1": ..., "result2": ..., "result3": ...}
```

**Переваги:**
- ✅ Автоматична паралелізація
- ✅ Працює в sync та async
- ✅ Результат в dict

---

## 🤖 Міграція агентів

### Простий агент з інструментами

**До (без LangGraph):**
```python
from langchain.agents import AgentExecutor, create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "What's the weather?"})

# Проблеми:
# - Чорна скринька
# - Важко кастомізувати
# - Складно дебажити
```

**Після (з LangGraph):**
```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def should_continue(state):
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"

def call_model(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})
workflow.add_edge("tools", "agent")

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage("What's the weather?")]})

# Переваги:
# ✅ Повна прозорість
# ✅ Легко кастомізувати
# ✅ Можна візуалізувати граф
```

### Агент з пам'яттю

**До (без LangGraph):**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# Проблеми:
# - Потрібно вручну під'єднувати memory
# - Важко зберігати стан між сесіями
# - Немає checkpointing
```

**Після (з LangGraph):**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user_123"}}

# Автоматичне збереження стану!
result1 = app.invoke({"messages": [HumanMessage("Мене звати Іван")]}, config)
result2 = app.invoke({"messages": [HumanMessage("Як мене звуть?")]}, config)

# Переваги:
# ✅ Автоматичний checkpointing
# ✅ Thread-based conversations
# ✅ Можна використовувати різні бекенди (Memory, SQLite, PostgreSQL)
```

---

## 🎯 Реальний приклад

### Customer Support Bot - Еволюція

#### v1.0: Простий чат
```python
chain = prompt | model | StrOutputParser()
```
❌ Проблеми: Немає пам'яті, немає знань

#### v2.0: + RAG
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)
```
✅ Має знання з документації
❌ Проблеми: Все ще немає пам'яті, не може діяти

#### v3.0: + LangGraph + Tools
```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([search_docs, reset_password, create_ticket]))

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```
✅ Пам'ятає контекст
✅ Може виконувати дії
✅ Готово до production!

**Дивіться:** `03_real_world_example.py`

---

## ✅ Чек-лист міграції

### Крок 1: Оновіть залежності
```bash
pip install langchain>=0.1.0 langchain-core>=0.1.0 langgraph>=0.0.20
```

### Крок 2: Замініть ланцюги
- [ ] `LLMChain` → LCEL з оператором `|`
- [ ] `SimpleSequentialChain` → Послідовні `|`
- [ ] `RetrievalQA` → RAG через LCEL
- [ ] `.run()`, `.predict()` → `.invoke()`

### Крок 3: Мігруйте агентів
- [ ] `AgentExecutor` → LangGraph `StateGraph`
- [ ] `ConversationBufferMemory` → Checkpointing
- [ ] Додайте явні conditional edges

### Крок 4: Structured Output
- [ ] Використовуйте Pydantic моделі
- [ ] `.with_structured_output()` для function calling
- [ ] `PydanticOutputParser` де потрібно

### Крок 5: Тестування
- [ ] Перевірте `.invoke()`, `.stream()`, `.batch()`
- [ ] Протестуйте checkpointing
- [ ] Перевірте async варіанти

---

## 📊 Таблиця міграції API

| v0.x | v1.0 | Примітка |
|------|------|----------|
| `LLMChain` | `prompt \| model \| parser` | Використовуйте LCEL |
| `.run()` | `.invoke()` | Єдиний метод |
| `.predict()` | `.invoke()` | Єдиний метод |
| `SimpleSequentialChain` | `chain1 \| chain2` | Просто pipe |
| `RetrievalQA` | Custom RAG chain | Більше контролю |
| `AgentExecutor` | `StateGraph` | LangGraph |
| `Memory` | `Checkpointer` | Автоматично |
| `.apply()` | `.batch()` | Оптимізовано |
| Різні streaming API | `.stream()` | Єдиний для всіх |

---

## 🎓 Додаткові ресурси

### Скрипти в цьому репозиторії

1. **`01_migration_chains_comparison.py`**
   - Детальне порівняння побудови ланцюгів
   - v0.x vs v1.0 side-by-side
   - Запуск: `python 01_migration_chains_comparison.py`

2. **`02_migration_agents_comparison.py`**
   - Порівняння підходів до агентів
   - Без LangGraph vs з LangGraph
   - Запуск: `python 02_migration_agents_comparison.py`

3. **`03_real_world_example.py`**
   - Customer Support Bot від простого до складного
   - Показує еволюцію з v1.0 можливостями
   - Запуск: `python 03_real_world_example.py`

### Документація

- [LangChain v1.0 Docs](https://python.langchain.com/)
- [LCEL Guide](https://python.langchain.com/docs/expression_language/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Migration Guide](https://python.langchain.com/docs/migration/)

---

## 💡 Головне

### Чому варто мігрувати?

1. **Простіше** - LCEL інтуїтивніший за class-based підхід
2. **Потужніше** - LangGraph дає повний контроль
3. **Надійніше** - Checkpointing та state management
4. **Швидше** - Оптимізована batch та streaming обробка
5. **Зручніше** - Єдиний Runnable інтерфейс

### З чого почати?

1. Прочитайте `MIGRATION_GUIDE.md` (цей документ)
2. Запустіть `01_migration_chains_comparison.py`
3. Запустіть `02_migration_agents_comparison.py`
4. Вивчіть `03_real_world_example.py`
5. Почніть міграцію свого коду!

---

**Версія:** 1.0
**Дата:** 2024
**Автор:** Claude Agent
