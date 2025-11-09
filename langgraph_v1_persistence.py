"""
LangGraph 1.0 - Persistence та Checkpointing
Демонстрація збереження стану та відновлення роботи агентів
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import time

load_dotenv()


def demo_basic_checkpointing():
    """
    Базова демонстрація checkpointing - збереження стану між викликами
    """
    print("=== Базовий Checkpointing ===\n")

    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        step_count: int

    def chatbot(state: State) -> State:
        """Простий чат-бот"""
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        response = model.invoke(state["messages"])
        return {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1
        }

    # Створення графа з checkpointer
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)

    # Додавання memory checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    # Конфігурація з thread_id для збереження історії
    config = {"configurable": {"thread_id": "conversation_1"}}

    # Перша взаємодія
    print("Сесія 1:")
    inputs = {"messages": [HumanMessage(content="Привіт! Мене звати Олексій.")]}
    result = app.invoke(inputs, config)
    print(f"Користувач: {inputs['messages'][0].content}")
    print(f"Бот: {result['messages'][-1].content}")
    print(f"Кроків виконано: {result['step_count']}\n")

    # Друга взаємодія - бот пам'ятає контекст
    print("Сесія 2 (з тим самим thread_id):")
    inputs2 = {"messages": [HumanMessage(content="Як мене звуть?")]}
    result2 = app.invoke(inputs2, config)
    print(f"Користувач: {inputs2['messages'][0].content}")
    print(f"Бот: {result2['messages'][-1].content}")
    print(f"Кроків виконано: {result2['step_count']}\n")

    # Нова розмова з іншим thread_id
    print("Сесія 3 (новий thread_id):")
    config_new = {"configurable": {"thread_id": "conversation_2"}}
    inputs3 = {"messages": [HumanMessage(content="Як мене звуть?")]}
    result3 = app.invoke(inputs3, config_new)
    print(f"Користувач: {inputs3['messages'][0].content}")
    print(f"Бот: {result3['messages'][-1].content}")
    print(f"Кроків виконано: {result3['step_count']}\n")

    return app


def demo_stateful_counter():
    """
    Демонстрація збереження стану лічильника між сесіями
    """
    print("=== Stateful Counter з Checkpointing ===\n")

    class CounterState(TypedDict):
        count: int
        history: list[str]

    def increment(state: CounterState) -> CounterState:
        """Збільшує лічильник"""
        new_count = state["count"] + 1
        timestamp = time.strftime("%H:%M:%S")
        history_entry = f"[{timestamp}] Лічильник: {state['count']} -> {new_count}"
        return {
            "count": new_count,
            "history": state["history"] + [history_entry]
        }

    workflow = StateGraph(CounterState)
    workflow.add_node("increment", increment)
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "counter_1"}}

    # Кілька інкрементів
    print("Інкременти:")
    state = {"count": 0, "history": []}

    for i in range(3):
        result = app.invoke(state, config)
        print(f"  Виклик {i+1}: count = {result['count']}")
        state = result
        time.sleep(0.1)

    print("\nІсторія змін:")
    for entry in result["history"]:
        print(f"  {entry}")

    print("\nПродовження після паузи (використовується збережений стан):")
    # Не передаємо state - граф відновить його з checkpoint
    result = app.invoke({"count": 0, "history": []}, config)
    print(f"  Поточний count: {result['count']}")
    print()

    return app


def demo_multi_step_with_resume():
    """
    Демонстрація багатокрокового процесу з можливістю відновлення
    """
    print("=== Багатокроковий процес з Resume ===\n")

    class WorkflowState(TypedDict):
        task: str
        completed_steps: list[str]
        current_step: int
        total_steps: int

    def step_1(state: WorkflowState) -> WorkflowState:
        """Крок 1: Аналіз"""
        print("  Виконується Крок 1: Аналіз задачі")
        return {
            "task": state["task"],
            "completed_steps": state["completed_steps"] + ["Аналіз"],
            "current_step": 1,
            "total_steps": state["total_steps"]
        }

    def step_2(state: WorkflowState) -> WorkflowState:
        """Крок 2: Планування"""
        print("  Виконується Крок 2: Планування рішення")
        return {
            "task": state["task"],
            "completed_steps": state["completed_steps"] + ["Планування"],
            "current_step": 2,
            "total_steps": state["total_steps"]
        }

    def step_3(state: WorkflowState) -> WorkflowState:
        """Крок 3: Виконання"""
        print("  Виконується Крок 3: Виконання плану")
        return {
            "task": state["task"],
            "completed_steps": state["completed_steps"] + ["Виконання"],
            "current_step": 3,
            "total_steps": state["total_steps"]
        }

    workflow = StateGraph(WorkflowState)
    workflow.add_node("step_1", step_1)
    workflow.add_node("step_2", step_2)
    workflow.add_node("step_3", step_3)

    workflow.set_entry_point("step_1")
    workflow.add_edge("step_1", "step_2")
    workflow.add_edge("step_2", "step_3")
    workflow.add_edge("step_3", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "workflow_1"}}

    # Початковий стан
    initial_state = {
        "task": "Розробити чат-бота",
        "completed_steps": [],
        "current_step": 0,
        "total_steps": 3
    }

    print("Запуск багатокрокового процесу:")
    result = app.invoke(initial_state, config)

    print(f"\nЗавершені кроки: {', '.join(result['completed_steps'])}")
    print(f"Прогрес: {result['current_step']}/{result['total_steps']}\n")

    return app


def demo_conversation_branches():
    """
    Демонстрація роботи з різними гілками розмови
    """
    print("=== Гілки розмови з Checkpointing ===\n")

    class ConversationState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        branch_name: str

    def chatbot(state: ConversationState) -> ConversationState:
        """Чат-бот з врахуванням гілки"""
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        # Додаємо контекст гілки до системного повідомлення
        system_msg = f"Ти асистент в гілці розмови: {state.get('branch_name', 'основна')}"
        messages = [HumanMessage(content=system_msg)] + list(state["messages"])

        response = model.invoke(messages)
        return {"messages": [response], "branch_name": state.get("branch_name", "")}

    workflow = StateGraph(ConversationState)
    workflow.add_node("chatbot", chatbot)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    # Гілка 1: Технічна підтримка
    print("Гілка 1: Технічна підтримка")
    config_tech = {"configurable": {"thread_id": "tech_support"}}
    result = app.invoke({
        "messages": [HumanMessage(content="У мене проблема з комп'ютером")],
        "branch_name": "технічна підтримка"
    }, config_tech)
    print(f"Користувач: У мене проблема з комп'ютером")
    print(f"Бот: {result['messages'][-1].content}\n")

    # Гілка 2: Загальні питання
    print("Гілка 2: Загальні питання")
    config_general = {"configurable": {"thread_id": "general_questions"}}
    result = app.invoke({
        "messages": [HumanMessage(content="Розкажи цікавий факт")],
        "branch_name": "загальні питання"
    }, config_general)
    print(f"Користувач: Розкажи цікавий факт")
    print(f"Бот: {result['messages'][-1].content}\n")

    # Продовження гілки 1
    print("Повернення до Гілки 1:")
    result = app.invoke({
        "messages": [HumanMessage(content="Що робити далі?")],
        "branch_name": "технічна підтримка"
    }, config_tech)
    print(f"Користувач: Що робити далі?")
    print(f"Бот: {result['messages'][-1].content}\n")

    return app


def demo_checkpoint_history():
    """
    Демонстрація доступу до історії checkpoints
    """
    print("=== Історія Checkpoints ===\n")

    class State(TypedDict):
        value: int
        operation: str

    def multiply_by_2(state: State) -> State:
        return {"value": state["value"] * 2, "operation": "multiply_by_2"}

    def add_10(state: State) -> State:
        return {"value": state["value"] + 10, "operation": "add_10"}

    workflow = StateGraph(State)
    workflow.add_node("multiply", multiply_by_2)
    workflow.add_node("add", add_10)

    workflow.set_entry_point("multiply")
    workflow.add_edge("multiply", "add")
    workflow.add_edge("add", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "math_ops"}}

    print("Виконання операцій:")
    initial_state = {"value": 5, "operation": "start"}
    print(f"Початкове значення: {initial_state['value']}")

    result = app.invoke(initial_state, config)
    print(f"Після multiply_by_2: {result['value']}")
    print(f"Остання операція: {result['operation']}\n")

    # Можна отримати історію станів (якщо checkpointer це підтримує)
    print(f"Фінальний стан: value={result['value']}, operation={result['operation']}")
    print()

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 1.0 - Persistence & Checkpointing Demonstration")
    print("=" * 60 + "\n")

    try:
        demo_basic_checkpointing()
        demo_stateful_counter()
        demo_multi_step_with_resume()
        demo_conversation_branches()
        demo_checkpoint_history()

        print("\n" + "=" * 60)
        print("Всі демонстрації persistence завершені!")
        print("=" * 60)

    except Exception as e:
        print(f"Помилка: {e}")
        print("Переконайтесь, що ви налаштували .env файл")
