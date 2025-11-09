"""
LangGraph 1.0 - Основи
Демонстрація побудови state machines та графів для агентів
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


# Визначення стану для простого графа
class SimpleState(TypedDict):
    """Простий стан з лічильником"""
    counter: int
    messages: list[str]


def demo_basic_graph():
    """
    Базова демонстрація створення графа з LangGraph
    """
    print("=== Базовий граф LangGraph ===\n")

    # Визначення вузлів (nodes)
    def increment_counter(state: SimpleState) -> SimpleState:
        """Збільшує лічильник"""
        print(f"  Поточний лічильник: {state['counter']}")
        return {
            "counter": state["counter"] + 1,
            "messages": state["messages"] + [f"Лічильник збільшено до {state['counter'] + 1}"]
        }

    def double_counter(state: SimpleState) -> SimpleState:
        """Подвоює лічильник"""
        new_value = state["counter"] * 2
        print(f"  Подвоюємо лічильник: {state['counter']} -> {new_value}")
        return {
            "counter": new_value,
            "messages": state["messages"] + [f"Лічильник подвоєно до {new_value}"]
        }

    # Створення графа
    workflow = StateGraph(SimpleState)

    # Додавання вузлів
    workflow.add_node("increment", increment_counter)
    workflow.add_node("double", double_counter)

    # Додавання ребер
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", "double")
    workflow.add_edge("double", END)

    # Компіляція графа
    app = workflow.compile()

    # Виконання
    initial_state = {"counter": 1, "messages": []}
    result = app.invoke(initial_state)

    print(f"\nПочатковий стан: counter={initial_state['counter']}")
    print(f"Фінальний стан: counter={result['counter']}")
    print(f"Повідомлення: {result['messages']}\n")

    return app


def demo_conditional_graph():
    """
    Граф з умовними переходами
    """
    print("=== Граф з умовними переходами ===\n")

    class ConditionalState(TypedDict):
        number: int
        result: str

    def check_number(state: ConditionalState) -> ConditionalState:
        """Перевіряє число"""
        return state

    def process_even(state: ConditionalState) -> ConditionalState:
        """Обробка парного числа"""
        print(f"  {state['number']} - парне число")
        return {"number": state["number"], "result": "парне"}

    def process_odd(state: ConditionalState) -> ConditionalState:
        """Обробка непарного числа"""
        print(f"  {state['number']} - непарне число")
        return {"number": state["number"], "result": "непарне"}

    # Умовна функція для маршрутизації
    def route_number(state: ConditionalState) -> str:
        """Визначає наступний вузол в залежності від парності"""
        if state["number"] % 2 == 0:
            return "even"
        else:
            return "odd"

    # Створення графа
    workflow = StateGraph(ConditionalState)

    workflow.add_node("check", check_number)
    workflow.add_node("even", process_even)
    workflow.add_node("odd", process_odd)

    workflow.set_entry_point("check")

    # Умовне ребро
    workflow.add_conditional_edges(
        "check",
        route_number,
        {
            "even": "even",
            "odd": "odd"
        }
    )

    workflow.add_edge("even", END)
    workflow.add_edge("odd", END)

    app = workflow.compile()

    # Тестування
    for number in [4, 7, 10]:
        print(f"Тестуємо число: {number}")
        result = app.invoke({"number": number, "result": ""})
        print(f"Результат: {result['result']}\n")

    return app


def demo_loop_graph():
    """
    Граф з циклом - ключова можливість LangGraph
    """
    print("=== Граф з циклом ===\n")

    class LoopState(TypedDict):
        count: int
        max_iterations: int

    def increment(state: LoopState) -> LoopState:
        """Інкремент лічильника"""
        new_count = state["count"] + 1
        print(f"  Ітерація {new_count}")
        return {"count": new_count, "max_iterations": state["max_iterations"]}

    def should_continue(state: LoopState) -> str:
        """Перевіряє чи продовжувати цикл"""
        if state["count"] < state["max_iterations"]:
            return "continue"
        else:
            return "end"

    workflow = StateGraph(LoopState)

    workflow.add_node("increment", increment)

    workflow.set_entry_point("increment")

    # Умовний цикл
    workflow.add_conditional_edges(
        "increment",
        should_continue,
        {
            "continue": "increment",  # Повернення до того ж вузла
            "end": END
        }
    )

    app = workflow.compile()

    # Виконання циклу
    print("Виконуємо 5 ітерацій:")
    result = app.invoke({"count": 0, "max_iterations": 5})
    print(f"Фінальний лічильник: {result['count']}\n")

    return app


def demo_message_graph():
    """
    Граф для роботи з повідомленнями - паттерн для чат-ботів
    """
    print("=== Граф для чат-бота ===\n")

    # Спеціальний стан для повідомлень
    class MessagesState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    def chatbot_node(state: MessagesState) -> MessagesState:
        """Вузол чат-бота"""
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)

    app = workflow.compile()

    # Тестування
    messages = [HumanMessage(content="Привіт! Розкажи короткий жарт про програмістів.")]
    result = app.invoke({"messages": messages})

    print(f"Користувач: {messages[0].content}")
    print(f"Бот: {result['messages'][-1].content}\n")

    return app


def demo_multi_step_reasoning():
    """
    Граф для багатокрокового міркування
    """
    print("=== Багатокроковий граф міркування ===\n")

    class ReasoningState(TypedDict):
        question: str
        steps: list[str]
        answer: str

    def analyze_question(state: ReasoningState) -> ReasoningState:
        """Аналіз питання"""
        step = f"Проаналізовано питання: {state['question']}"
        print(f"  Крок 1: {step}")
        return {
            "question": state["question"],
            "steps": state["steps"] + [step],
            "answer": state["answer"]
        }

    def break_down(state: ReasoningState) -> ReasoningState:
        """Розбиття на підзадачі"""
        step = "Розбито на підзадачі"
        print(f"  Крок 2: {step}")
        return {
            "question": state["question"],
            "steps": state["steps"] + [step],
            "answer": state["answer"]
        }

    def solve(state: ReasoningState) -> ReasoningState:
        """Вирішення"""
        step = "Знайдено рішення"
        answer = f"Відповідь на питання: {state['question']}"
        print(f"  Крок 3: {step}")
        print(f"  Фінальна відповідь: {answer}")
        return {
            "question": state["question"],
            "steps": state["steps"] + [step],
            "answer": answer
        }

    workflow = StateGraph(ReasoningState)

    workflow.add_node("analyze", analyze_question)
    workflow.add_node("break_down", break_down)
    workflow.add_node("solve", solve)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "break_down")
    workflow.add_edge("break_down", "solve")
    workflow.add_edge("solve", END)

    app = workflow.compile()

    # Виконання
    result = app.invoke({
        "question": "Як побудувати чат-бота з LangGraph?",
        "steps": [],
        "answer": ""
    })

    print(f"\nВсі кроки:")
    for i, step in enumerate(result["steps"], 1):
        print(f"  {i}. {step}")
    print()

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 1.0 - Basics Demonstration")
    print("=" * 60 + "\n")

    try:
        demo_basic_graph()
        demo_conditional_graph()
        demo_loop_graph()
        demo_message_graph()
        demo_multi_step_reasoning()

        print("\n" + "=" * 60)
        print("Всі базові демонстрації LangGraph завершені!")
        print("=" * 60)

    except Exception as e:
        print(f"Помилка: {e}")
        print("Переконайтесь, що ви налаштували .env файл")
