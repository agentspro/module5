"""
ПОРІВНЯННЯ: Побудова Агентів - До LangGraph vs З LangGraph

ПРОБЛЕМА: Створення stateful агентів було складним, багато ручного коду
РІШЕННЯ: LangGraph - декларативна побудова агентів через графи
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()


print("=" * 80)
print("МІГРАЦІЯ: Побудова Агентів - Без LangGraph → З LangGraph")
print("=" * 80 + "\n")


# ============================================================================
# ПРИКЛАД 1: Простий агент з інструментами
# ============================================================================

print("\n" + "=" * 80)
print("1. АГЕНТ З ІНСТРУМЕНТАМИ")
print("=" * 80 + "\n")

print("❌ БЕЗ LANGGRAPH - Ручне управління циклом виконання")
print("-" * 80)
print("""
from langchain.agents import AgentExecutor, create_openai_functions_agent

# Створюємо агента
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# ПРОБЛЕМИ:
# 1. AgentExecutor - чорна скринька, складно кастомізувати
# 2. Важко контролювати логіку виконання
# 3. Обмежені можливості для додавання кроків
# 4. Складно дебажити що відбувається всередині
# 5. Немає візуалізації потоку виконання

result = agent_executor.invoke({"input": "Яка погода?"})
""")

print("\n✅ З LANGGRAPH - Явний граф виконання")
print("-" * 80)
print("""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

def call_model(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Будуємо граф - ЯВНО видно що відбувається
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
result = app.invoke({"messages": [HumanMessage(content="Яка погода?")]})

# ПЕРЕВАГИ:
# 1. Візуально видно граф виконання
# 2. Повний контроль над кожним кроком
# 3. Легко додавати власні ноди
# 4. Просто дебажити - бачимо весь потік
# 5. Можна експортувати граф для візуалізації
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Декларативний підхід замість імперативного")
print("  • Повна прозорість логіки виконання")
print("  • Легко кастомізувати будь-який крок")
print("  • Можливість візуалізації графа")
print("  • Простіше тестувати окремі компоненти\n")


# Демонстрація LangGraph підходу
@tool
def get_current_weather(location: str) -> str:
    """Отримує поточну погоду для вказаного міста"""
    return f"Погода в {location}: Сонячно, +22°C"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [get_current_weather]
model_with_tools = model.bind_tools(tools)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

def call_model(state: AgentState) -> AgentState:
    response = model_with_tools.invoke(state["messages"])
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

print("📝 Демонстрація LangGraph агента:")
result = app.invoke({"messages": [HumanMessage(content="Яка погода в Києві?")]})
for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"   Агент: {msg.content}")
print()


# ============================================================================
# ПРИКЛАД 2: Збереження стану між викликами
# ============================================================================

print("\n" + "=" * 80)
print("2. ЗБЕРЕЖЕННЯ СТАНУ (MEMORY)")
print("=" * 80 + "\n")

print("❌ БЕЗ LANGGRAPH - Ручне управління пам'яттю")
print("-" * 80)
print("""
from langchain.memory import ConversationBufferMemory

# Потрібно вручну створювати та під'єднувати memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory  # Під'єднуємо вручну
)

# ПРОБЛЕМИ:
# 1. Різні типи Memory для різних задач
# 2. Важко керувати довгостроковою пам'яттю
# 3. Складно зберігати стан між сесіями
# 4. Немає вбудованого checkpointing
# 5. Важко відкотитися до попереднього стану

result1 = agent_executor.invoke({"input": "Мене звати Іван"})
result2 = agent_executor.invoke({"input": "Як мене звуть?"})
""")

print("\n✅ З LANGGRAPH - Вбудований Checkpointing")
print("-" * 80)
print("""
from langgraph.checkpoint.memory import MemorySaver

# Просто додаємо checkpointer при компіляції
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Використовуємо thread_id для різних розмов
config = {"configurable": {"thread_id": "user_123"}}

# Стан автоматично зберігається!
result1 = app.invoke(
    {"messages": [HumanMessage("Мене звати Іван")]},
    config
)

result2 = app.invoke(
    {"messages": [HumanMessage("Як мене звуть?")]},
    config  # Той самий thread_id - пам'ятає контекст!
)

# ПЕРЕВАГИ:
# 1. Автоматичне збереження стану після кожного кроку
# 2. Можливість відновлення з будь-якого checkpoint
# 3. Різні thread_id для різних користувачів
# 4. Можна використовувати різні бекенди (Memory, SQLite, PostgreSQL)
# 5. Вбудована підтримка time travel - повернутись до попереднього стану
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Автоматичне управління станом")
print("  • Checkpointing з коробки")
print("  • Легке масштабування (різні бекенди)")
print("  • Можливість відновлення після збою")
print("  • Time travel для дебагу\n")


# Демонстрація
workflow_with_memory = StateGraph(AgentState)
workflow_with_memory.add_node("agent", call_model)
workflow_with_memory.add_node("tools", ToolNode(tools))
workflow_with_memory.set_entry_point("agent")
workflow_with_memory.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})
workflow_with_memory.add_edge("tools", "agent")

checkpointer = MemorySaver()
app_with_memory = workflow_with_memory.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "demo_conversation"}}

print("📝 Демонстрація пам'яті:")
result1 = app_with_memory.invoke(
    {"messages": [HumanMessage(content="Мене звати Олексій")]},
    config
)
print(f"   Користувач: Мене звати Олексій")
print(f"   Агент: {result1['messages'][-1].content}\n")

result2 = app_with_memory.invoke(
    {"messages": [HumanMessage(content="Як мене звуть?")]},
    config
)
print(f"   Користувач: Як мене звуть?")
print(f"   Агент: {result2['messages'][-1].content}\n")


# ============================================================================
# ПРИКЛАД 3: Циклічна логіка
# ============================================================================

print("\n" + "=" * 80)
print("3. ЦИКЛІЧНА ЛОГІКА (LOOPS)")
print("=" * 80 + "\n")

print("❌ БЕЗ LANGGRAPH - Обмежені можливості")
print("-" * 80)
print("""
# AgentExecutor має вбудований цикл, але:

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # Тільки базовий лімітер
    max_execution_time=60
)

# ПРОБЛЕМИ:
# 1. Не можна створювати власні цикли
# 2. Циклічна логіка захована всередині AgentExecutor
# 3. Важко додати умовні переходи
# 4. Немає можливості повернутись до попереднього кроку
# 5. Складно реалізувати retry логіку

result = agent_executor.invoke({"input": "складне завдання"})
""")

print("\n✅ З LANGGRAPH - Повний контроль над циклами")
print("-" * 80)
print("""
class LoopState(TypedDict):
    count: int
    max_iterations: int

def increment(state: LoopState):
    return {"count": state["count"] + 1, "max_iterations": state["max_iterations"]}

def should_continue(state: LoopState) -> str:
    if state["count"] < state["max_iterations"]:
        return "continue"  # Повернутись до того ж вузла!
    return "end"

workflow = StateGraph(LoopState)
workflow.add_node("increment", increment)
workflow.set_entry_point("increment")

# Створюємо цикл!
workflow.add_conditional_edges("increment", should_continue, {
    "continue": "increment",  # Цикл: повертаємось до себе
    "end": END
})

# ПЕРЕВАГИ:
# 1. Можна створювати будь-які цикли
# 2. Повний контроль над умовами виходу
# 3. Можливість retry логіки
# 4. Легко візуалізувати циклічний потік
# 5. Можна комбінувати різні типи циклів
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Явні цикли замість прихованих")
print("  • Умовні переходи між вузлами")
print("  • Можливість повертатись до попередніх кроків")
print("  • Retry та error handling логіка")
print("  • Складні циклічні патерни\n")


# ============================================================================
# ПРИКЛАД 4: Human-in-the-Loop
# ============================================================================

print("\n" + "=" * 80)
print("4. HUMAN-IN-THE-LOOP")
print("=" * 80 + "\n")

print("❌ БЕЗ LANGGRAPH - Важко реалізувати")
print("-" * 80)
print("""
# Потрібно вручну переривати та відновлювати виконання

agent_executor = AgentExecutor(...)

# Немає вбудованого способу зупинити виконання
# Потрібно писати власну логіку:
# 1. Зупинити агента
# 2. Отримати ввід від людини
# 3. Якось зберегти стан
# 4. Відновити виконання зі збереженого стану

# Це все потрібно реалізовувати вручну!
""")

print("\n✅ З LANGGRAPH - Вбудована підтримка")
print("-" * 80)
print("""
class ApprovalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    requires_approval: bool

def should_ask_human(state: ApprovalState) -> str:
    if state["requires_approval"]:
        return "approval"  # Перейти до вузла схвалення
    return "execute"

workflow = StateGraph(ApprovalState)
workflow.add_node("agent", agent_node)
workflow.add_node("approval", human_approval_node)
workflow.add_node("execute", execute_node)

workflow.add_conditional_edges("agent", should_ask_human, {
    "approval": "approval",  # Запит схвалення
    "execute": "execute"
})

# З checkpointing можна:
# 1. Зупинити виконання в вузлі "approval"
# 2. Чекати на ввід людини
# 3. Продовжити з того ж місця з thread_id

# ПЕРЕВАГИ:
# 1. Явні точки зупинки
# 2. Збереження стану автоматичне
# 3. Легко відновити виконання
# 4. Можна мати кілька точок схвалення
""")

print("\n🎯 ЩО ПОКРАЩИЛОСЬ:")
print("  • Явні точки для людського вводу")
print("  • Автоматичне збереження стану")
print("  • Легке відновлення після паузи")
print("  • Підтримка approval workflows\n")


# ============================================================================
# ПІДСУМОК
# ============================================================================

print("\n" + "=" * 80)
print("📊 ПІДСУМОК: Чому LangGraph важливий")
print("=" * 80 + "\n")

print("┌──────────────────────┬────────────────────────┬──────────────────────────┐")
print("│ Можливість           │ Без LangGraph          │ З LangGraph              │")
print("├──────────────────────┼────────────────────────┼──────────────────────────┤")
print("│ Створення агента     │ AgentExecutor (чорна   │ Явний граф - повний      │")
print("│                      │ скринька)              │ контроль                 │")
print("├──────────────────────┼────────────────────────┼──────────────────────────┤")
print("│ Візуалізація логіки  │ ❌ Важко побачити       │ ✅ Граф можна експортувати│")
print("├──────────────────────┼────────────────────────┼──────────────────────────┤")
print("│ Збереження стану     │ Manual Memory управління│ Автоматичний checkpointing│")
print("├──────────────────────┼────────────────────────┼──────────────────────────┤")
print("│ Циклічна логіка      │ ❌ Обмежена             │ ✅ Повний контроль        │")
print("├──────────────────────┼────────────────────────┼──────────────────────────┤")
print("│ Human-in-the-loop    │ ❌ Потрібно писати вручну│ ✅ Вбудована підтримка   │")
print("├──────────────────────┼────────────────────────┼──────────────────────────┤")
print("│ Дебаг                │ ⭐⭐ Складно             │ ⭐⭐⭐⭐⭐ Легко           │")
print("├──────────────────────┼────────────────────────┼──────────────────────────┤")
print("│ Кастомізація         │ ⭐⭐ Обмежена           │ ⭐⭐⭐⭐⭐ Необмежена      │")
print("├──────────────────────┼────────────────────────┼──────────────────────────┤")
print("│ Складні workflow     │ ❌ Важко реалізувати    │ ✅ Природньо             │")
print("└──────────────────────┴────────────────────────┴──────────────────────────┘")

print("\n💡 КЛЮЧОВІ ПЕРЕВАГИ LANGGRAPH:")
print("  1. 🎯 Декларативний підхід - граф описує логіку")
print("  2. 👁️  Прозорість - видно весь потік виконання")
print("  3. 💾 Checkpointing - автоматичне збереження стану")
print("  4. 🔄 Цикли - підтримка складної циклічної логіки")
print("  5. 👤 Human-in-the-loop - легка інтеграція людини")
print("  6. 🔧 Кастомізація - повний контроль над кожним кроком")
print("  7. 🐛 Дебаг - простіше знайти та виправити проблеми")
print("  8. 📊 Візуалізація - граф можна експортувати та переглянути")

print("\n🎓 КОЛИ ВИКОРИСТОВУВАТИ LANGGRAPH:")
print("  ✅ Складні multi-step агенти")
print("  ✅ Потрібна циклічна логіка")
print("  ✅ Збереження стану між викликами")
print("  ✅ Human approval workflows")
print("  ✅ Складні умовні переходи")
print("  ✅ Потрібен повний контроль над виконанням")

print("\n" + "=" * 80)
