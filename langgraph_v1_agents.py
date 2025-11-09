"""
LangGraph 1.0 - Агенти
Демонстрація побудови агентів з інструментами та циклами міркування
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import json

load_dotenv()


# Визначення інструментів
@tool
def calculator(expression: str) -> str:
    """Виконує математичні обчислення. Приймає вираз як рядок."""
    try:
        result = eval(expression)
        return f"Результат: {result}"
    except Exception as e:
        return f"Помилка обчислення: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Отримує погоду для вказаного міста."""
    # Симуляція API погоди
    weather_data = {
        "Київ": "Сонячно, +20°C",
        "Львів": "Хмарно, +18°C",
        "Одеса": "Дощ, +22°C",
    }
    return weather_data.get(city, f"Погода для {city} недоступна")


@tool
def search_info(query: str) -> str:
    """Шукає інформацію на запит."""
    # Симуляція пошуку
    info_base = {
        "LangGraph": "LangGraph - бібліотека для побудови stateful агентів з циклами",
        "Python": "Python - високорівнева мова програмування",
        "AI": "Штучний інтелект - галузь комп'ютерних наук",
    }
    for key, value in info_base.items():
        if key.lower() in query.lower():
            return value
    return f"Інформацію про '{query}' не знайдено"


def demo_simple_agent():
    """
    Простий агент з одним інструментом
    """
    print("=== Простий агент з інструментом ===\n")

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    # Ініціалізація моделі з інструментами
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [calculator]
    model_with_tools = model.bind_tools(tools)

    def should_continue(state: AgentState) -> str:
        """Визначає чи потрібно викликати інструменти"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    def call_model(state: AgentState) -> AgentState:
        """Викликає модель"""
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Створення графа
    workflow = StateGraph(AgentState)

    # Додавання вузлів
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")

    # Умовні переходи
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # Тестування
    inputs = {"messages": [HumanMessage(content="Скільки буде 123 * 456?")]}
    result = app.invoke(inputs)

    print("Діалог:")
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"  Користувач: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                print(f"  Агент: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"  Агент викликає: {msg.tool_calls[0]['name']}")
        elif isinstance(msg, ToolMessage):
            print(f"  Інструмент: {msg.content}")
    print()

    return app


def demo_multi_tool_agent():
    """
    Агент з кількома інструментами
    """
    print("=== Агент з кількома інструментами ===\n")

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [calculator, get_weather, search_info]
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
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # Тестування різних запитів
    queries = [
        "Яка погода в Києві?",
        "Розкажи про LangGraph",
        "Порахуй 250 + 750",
    ]

    for query in queries:
        print(f"Запит: {query}")
        result = app.invoke({"messages": [HumanMessage(content=query)]})
        # Виводимо тільки фінальну відповідь
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, "tool_calls"):
                print(f"Відповідь: {msg.content}")
        print()

    return app


def demo_agent_with_memory():
    """
    Агент з пам'яттю попередніх взаємодій
    """
    print("=== Агент з пам'яттю ===\n")

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        conversation_history: list[dict]

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    tools = [calculator]
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
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # Багаторазова взаємодія з пам'яттю
    state = {"messages": [], "conversation_history": []}

    interactions = [
        "Привіт! Мене звати Іван.",
        "Як мене звуть?",
        "Скільки буде 100 + 50?",
        "А помножити цей результат на 2?",
    ]

    for user_input in interactions:
        print(f"Користувач: {user_input}")
        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        state["messages"] = result["messages"]

        # Виводимо останню відповідь агента
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"Агент: {msg.content}\n")
                break

    return app


def demo_reasoning_agent():
    """
    Агент з поетапним міркуванням (ReAct паттерн)
    """
    print("=== Агент з поетапним міркуванням (ReAct) ===\n")

    class ReActState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        iterations: int

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [calculator, search_info]
    model_with_tools = model.bind_tools(tools)

    MAX_ITERATIONS = 5

    def should_continue(state: ReActState) -> str:
        if state["iterations"] >= MAX_ITERATIONS:
            return "end"

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"

    def call_model(state: ReActState) -> ReActState:
        print(f"  Ітерація {state['iterations'] + 1}")
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response], "iterations": state["iterations"] + 1}

    workflow = StateGraph(ReActState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # Складне завдання
    query = "Знайди інформацію про Python і порахуй, скільки буде 2^8"
    print(f"Завдання: {query}\n")

    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "iterations": 0
    })

    print(f"\nВсього ітерацій: {result['iterations']}")
    print("Фінальна відповідь:")
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"  {msg.content}")
            break
    print()

    return app


def demo_supervisor_agent():
    """
    Агент-супервізор, який координує роботу інших агентів
    """
    print("=== Агент-супервізор ===\n")

    class SupervisorState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        next_agent: str

    # Спеціалізовані агенти
    math_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    math_tools = [calculator]
    math_agent = math_model.bind_tools(math_tools)

    info_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    info_tools = [search_info]
    info_agent = info_model.bind_tools(info_tools)

    # Супервізор
    supervisor_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def supervisor(state: SupervisorState) -> SupervisorState:
        """Визначає який агент потрібен"""
        last_message = state["messages"][-1]

        # Проста логіка маршрутизації
        if any(word in last_message.content.lower() for word in ["порахуй", "скільки", "математика"]):
            next_agent = "math"
        elif any(word in last_message.content.lower() for word in ["інформація", "розкажи", "що таке"]):
            next_agent = "info"
        else:
            next_agent = "end"

        print(f"  Супервізор направляє до: {next_agent}")
        return {"messages": [], "next_agent": next_agent}

    def math_node(state: SupervisorState) -> SupervisorState:
        """Математичний агент"""
        print("  Працює математичний агент")
        response = math_agent.invoke(state["messages"])
        return {"messages": [response], "next_agent": ""}

    def info_node(state: SupervisorState) -> SupervisorState:
        """Інформаційний агент"""
        print("  Працює інформаційний агент")
        response = info_agent.invoke(state["messages"])
        return {"messages": [response], "next_agent": ""}

    def route_agent(state: SupervisorState) -> str:
        return state["next_agent"]

    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("math", math_node)
    workflow.add_node("info", info_node)
    workflow.add_node("math_tools", ToolNode(math_tools))
    workflow.add_node("info_tools", ToolNode(info_tools))

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {"math": "math", "info": "info", "end": END}
    )

    # Додавання обробки інструментів
    workflow.add_edge("math", "math_tools")
    workflow.add_edge("info", "info_tools")
    workflow.add_edge("math_tools", END)
    workflow.add_edge("info_tools", END)

    app = workflow.compile()

    # Тестування
    queries = [
        "Розкажи про Python",
        "Скільки буде 15 * 20?",
    ]

    for query in queries:
        print(f"Запит: {query}")
        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "next_agent": ""
        })
        print()

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 1.0 - Agents Demonstration")
    print("=" * 60 + "\n")

    try:
        demo_simple_agent()
        demo_multi_tool_agent()
        demo_agent_with_memory()
        demo_reasoning_agent()
        demo_supervisor_agent()

        print("\n" + "=" * 60)
        print("Всі демонстрації агентів завершені!")
        print("=" * 60)

    except Exception as e:
        print(f"Помилка: {e}")
        print("Переконайтесь, що ви налаштували .env файл")
