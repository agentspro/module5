"""
РЕАЛЬНИЙ ПРИКЛАД: Customer Support Bot
Показує еволюцію від простого до складного з використанням v1.0 можливостей

СЦЕНАРІЙ: Бот технічної підтримки з:
- Пошуком в базі знань (RAG)
- Використанням інструментів
- Збереженням історії
- Ескалацією до людини
"""

from typing import TypedDict, Annotated, Sequence, Literal
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


print("=" * 80)
print("РЕАЛЬНИЙ ПРИКЛАД: Еволюція Customer Support Bot")
print("=" * 80 + "\n")


# ============================================================================
# КРОК 1: Простий чат-бот (Базовий рівень)
# ============================================================================

print("\n" + "=" * 80)
print("ВЕРСІЯ 1.0: Простий чат-бот без контексту")
print("=" * 80 + "\n")

print("⚠️  ПРОБЛЕМА: Бот не пам'ятає контекст, немає доступу до бази знань")
print("-" * 80 + "\n")


def simple_chatbot_v1():
    """Найпростіший чат-бот - просто відповідає"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ти помічник технічної підтримки. Будь ввічливим та корисним."),
        ("user", "{input}")
    ])

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    chain = prompt | model | StrOutputParser()

    return chain


print("Код:")
print("""
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ти помічник технічної підтримки..."),
    ("user", "{input}")
])
chain = prompt | model | StrOutputParser()
""")

chatbot_v1 = simple_chatbot_v1()

print("\n📝 Тест:")
response = chatbot_v1.invoke({"input": "Як скинути пароль?"})
print(f"Користувач: Як скинути пароль?")
print(f"Бот: {response}")

response2 = chatbot_v1.invoke({"input": "А де кнопка налаштувань?"})
print(f"\nКористувач: А де кнопка налаштувань?")
print(f"Бот: {response2}")

print("\n❌ ПРОБЛЕМИ:")
print("  • Бот не пам'ятає попередні повідомлення")
print("  • Немає доступу до документації")
print("  • Не може виконувати дії (наприклад, скинути пароль)\n")


# ============================================================================
# КРОК 2: Додаємо RAG (База знань)
# ============================================================================

print("\n" + "=" * 80)
print("ВЕРСІЯ 2.0: + RAG для пошуку в базі знань")
print("=" * 80 + "\n")

print("✅ ПОКРАЩЕННЯ: Використовуємо LCEL для RAG pipeline")
print("-" * 80 + "\n")


def create_knowledge_base():
    """Створюємо базу знань для підтримки"""
    docs = [
        Document(
            page_content="Щоб скинути пароль: 1) Натисніть 'Забули пароль' на сторінці входу 2) Введіть email 3) Перейдіть за посиланням з листа",
            metadata={"category": "authentication", "topic": "password_reset"}
        ),
        Document(
            page_content="Кнопка налаштувань знаходиться в правому верхньому куті. Клікніть на іконку профілю → Налаштування",
            metadata={"category": "navigation", "topic": "settings"}
        ),
        Document(
            page_content="Щоб завантажити файл: клікніть на кнопку Upload → Оберіть файл → Підтвердіть завантаження. Максимальний розмір: 10MB",
            metadata={"category": "features", "topic": "file_upload"}
        ),
        Document(
            page_content="Якщо виникає помилка 'Connection timeout': 1) Перевірте інтернет з'єднання 2) Спробуйте оновити сторінку 3) Очистіть кеш браузера",
            metadata={"category": "troubleshooting", "topic": "connection_errors"}
        ),
        Document(
            page_content="Щоб змінити мову інтерфейсу: Налаштування → Мова → Оберіть потрібну мову зі списку → Зберегти",
            metadata={"category": "settings", "topic": "language"}
        ),
    ]
    return docs


def chatbot_with_rag_v2():
    """Чат-бот з RAG - шукає відповіді в базі знань"""
    # Створюємо векторну базу
    docs = create_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # RAG промпт
    template = """Ти помічник технічної підтримки. Використовуй наступну інформацію для відповіді.

База знань:
{context}

Питання користувача: {question}

Дай детальну та корисну відповідь на основі бази знань. Якщо інформації немає - так і скажи."""

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL RAG ланцюг
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


print("Код:")
print("""
# Створюємо векторну базу
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# LCEL RAG ланцюг
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)
""")

chatbot_v2 = chatbot_with_rag_v2()

print("\n📝 Тест:")
response = chatbot_v2.invoke("Як скинути пароль?")
print(f"Користувач: Як скинути пароль?")
print(f"Бот: {response}\n")

print("✅ ПОКРАЩЕННЯ:")
print("  • Бот знаходить інформацію в базі знань")
print("  • Відповіді більш точні та детальні")
print("  • Використовує LCEL для елегантного RAG pipeline\n")

print("❌ ЩЕ ЗАЛИШИЛОСЬ:")
print("  • Все ще немає пам'яті розмови")
print("  • Не може виконувати дії (інструменти)\n")


# ============================================================================
# КРОК 3: Додаємо інструменти (Tools) та стан
# ============================================================================

print("\n" + "=" * 80)
print("ВЕРСІЯ 3.0: + Інструменти + Стан (LangGraph)")
print("=" * 80 + "\n")

print("✅ ПОКРАЩЕННЯ: LangGraph для stateful агента з інструментами")
print("-" * 80 + "\n")


# Визначаємо інструменти
@tool
def reset_password(email: str) -> str:
    """Скидає пароль користувача та відправляє лист"""
    return f"✅ Лист для скидання пароля відправлено на {email}"


@tool
def check_account_status(user_id: str) -> str:
    """Перевіряє статус облікового запису"""
    statuses = {
        "user123": "Активний, Premium підписка до 2024-12-31",
        "user456": "Активний, Free план",
    }
    return statuses.get(user_id, "Користувача не знайдено")


@tool
def create_ticket(issue_description: str, priority: str = "normal") -> str:
    """Створює тікет підтримки для складних проблем"""
    ticket_id = f"TICKET-{hash(issue_description) % 10000}"
    return f"✅ Створено тікет {ticket_id} з пріоритетом {priority}. Наша команда розгляне його протягом 24 годин."


@tool
def search_documentation(query: str) -> str:
    """Шукає в документації"""
    # Використовуємо RAG з попереднього кроку
    docs = create_knowledge_base()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    results = retriever.invoke(query)
    if results:
        return "\n".join([doc.page_content for doc in results])
    return "Інформації не знайдено"


def chatbot_with_tools_v3():
    """Stateful агент з інструментами та пам'яттю"""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_id: str

    tools = [reset_password, check_account_status, create_ticket, search_documentation]
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    model_with_tools = model.bind_tools(tools)

    def should_continue(state: AgentState) -> Literal["continue", "end"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    def call_model(state: AgentState) -> AgentState:
        # Додаємо системний промпт
        system_msg = SystemMessage(content="""Ти професійний помічник технічної підтримки.

У тебе є доступ до інструментів:
- search_documentation: шукай в базі знань
- reset_password: скидай пароль
- check_account_status: перевіряй статус акаунта
- create_ticket: створюй тікети для складних питань

Спочатку спробуй знайти відповідь в документації. Якщо потрібно - використовуй інші інструменти.
Будь ввічливим та корисним.""")

        messages = [system_msg] + list(state["messages"])
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # Будуємо граф
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")

    # Додаємо checkpointer для збереження історії
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    return app


print("Код:")
print("""
# Визначаємо інструменти
@tool
def reset_password(email: str) -> str:
    '''Скидає пароль користувача'''
    ...

# Будуємо LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})

# Додаємо checkpointing
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
""")

chatbot_v3 = chatbot_with_tools_v3()

print("\n📝 Тест multi-turn розмови:")

config = {"configurable": {"thread_id": "customer_1"}}

# Запит 1
print("Користувач: Не можу увійти в акаунт")
result = chatbot_v3.invoke({
    "messages": [HumanMessage(content="Не можу увійти в акаунт")],
    "user_id": "user123"
}, config)
for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"Бот: {msg.content}")
        break

# Запит 2 - в тому ж треді
print("\nКористувач: Скинь мені пароль на test@example.com")
result = chatbot_v3.invoke({
    "messages": [HumanMessage(content="Скинь мені пароль на test@example.com")],
    "user_id": "user123"
}, config)
for msg in reversed(result["messages"]):
    if isinstance(msg, AIMessage) and msg.content and "відправлено" in msg.content.lower():
        print(f"Бот: {msg.content}")
        break

print("\n✅ ПОКРАЩЕННЯ:")
print("  • Агент має доступ до інструментів")
print("  • Пам'ятає контекст розмови (checkpointing)")
print("  • Може виконувати реальні дії")
print("  • Автоматично вибирає потрібний інструмент")
print("  • LangGraph дає повний контроль над логікою\n")


# ============================================================================
# ПІДСУМОК ЕВОЛЮЦІЇ
# ============================================================================

print("\n" + "=" * 80)
print("📊 ЕВОЛЮЦІЯ CUSTOMER SUPPORT BOT")
print("=" * 80 + "\n")

print("┌────────┬──────────────────┬──────────────────┬──────────────────────────┐")
print("│ Версія │ Технології      │ Можливості       │ Обмеження                │")
print("├────────┼──────────────────┼──────────────────┼──────────────────────────┤")
print("│ v1.0   │ Prompt + Model   │ Базові відповіді │ Немає пам'яті,           │")
print("│        │                  │                  │ немає знань              │")
print("├────────┼──────────────────┼──────────────────┼──────────────────────────┤")
print("│ v2.0   │ + LCEL RAG       │ Пошук в базі     │ Немає пам'яті,           │")
print("│        │ + FAISS          │ знань            │ не може діяти            │")
print("├────────┼──────────────────┼──────────────────┼──────────────────────────┤")
print("│ v3.0   │ + LangGraph      │ Інструменти,     │ Готово до                │")
print("│        │ + Tools          │ пам'ять,         │ production! 🎉           │")
print("│        │ + Checkpointing  │ дії              │                          │")
print("└────────┴──────────────────┴──────────────────┴──────────────────────────┘")

print("\n💡 КЛЮЧОВІ УРОКИ:")
print("  1. LCEL (|) робить композицію простою та зрозумілою")
print("  2. RAG додає знання з документації")
print("  3. LangGraph дає stateful логіку з пам'яттю")
print("  4. Tools дозволяють агенту діяти, а не лише говорити")
print("  5. Checkpointing зберігає контекст між викликами")

print("\n🎯 ЩО РОБИТЬ v1.0 ОСОБЛИВИМ:")
print("  ✅ LCEL - інтуїтивна композиція через |")
print("  ✅ Єдиний Runnable інтерфейс - invoke/stream/batch для всього")
print("  ✅ LangGraph - декларативні state machines")
print("  ✅ Checkpointing - автоматичне збереження стану")
print("  ✅ Structured Output - типізовані дані з Pydantic")
print("  ✅ Tools інтеграція - природна для LLM")

print("\n🚀 ГОТОВО ДО PRODUCTION:")
print("  • Додайте персистентний checkpointer (PostgreSQL/Redis)")
print("  • Підключіть реальні API замість mock функцій")
print("  • Додайте error handling та retry логіку")
print("  • Налаштуйте LangSmith для моніторингу")
print("  • Масштабуйте з thread_id для різних користувачів")

print("\n" + "=" * 80)
