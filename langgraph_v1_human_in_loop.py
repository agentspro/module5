"""
LangGraph 1.0 - Human-in-the-Loop
Демонстрація інтеграції людини в процес роботи агента
"""

from typing import TypedDict, Annotated, Sequence, Literal
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


# Інструменти для демонстрації
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Відправляє email. ПОТРЕБУЄ ПІДТВЕРДЖЕННЯ ЛЮДИНИ."""
    return f"Email відправлено до {to} з темою '{subject}'"


@tool
def delete_file(filename: str) -> str:
    """Видаляє файл. ПОТРЕБУЄ ПІДТВЕРДЖЕННЯ ЛЮДИНИ."""
    return f"Файл {filename} видалено"


@tool
def get_info(query: str) -> str:
    """Отримує інформацію. Не потребує підтвердження."""
    return f"Інформація про {query}: це важлива тема"


def demo_approval_workflow():
    """
    Демонстрація workflow з обов'язковим схваленням людини
    """
    print("=== Workflow з підтвердженням ===\n")

    class ApprovalState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        approval_needed: bool
        approved: bool
        pending_action: str

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [send_email]
    model_with_tools = model.bind_tools(tools)

    def agent(state: ApprovalState) -> ApprovalState:
        """Агент що планує дії"""
        response = model_with_tools.invoke(state["messages"])

        # Перевірка чи потрібне схвалення
        approval_needed = False
        pending_action = ""

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]
            if tool_call["name"] in ["send_email", "delete_file"]:
                approval_needed = True
                pending_action = f"{tool_call['name']}: {tool_call['args']}"

        return {
            "messages": [response],
            "approval_needed": approval_needed,
            "approved": state.get("approved", False),
            "pending_action": pending_action
        }

    def human_approval(state: ApprovalState) -> ApprovalState:
        """Запит на підтвердження від людини"""
        if state["approval_needed"]:
            print(f"\n⚠️  ПОТРІБНЕ ПІДТВЕРДЖЕННЯ:")
            print(f"   Дія: {state['pending_action']}")
            print(f"   У реальному сценарії тут буде пауза для підтвердження людини\n")

            # Симуляція схвалення (в реальності тут буде пауза)
            approved = True  # В реальності: input("Підтвердити? (y/n): ") == "y"

            if approved:
                print("   ✅ Дію схвалено\n")
                return {
                    "messages": [],
                    "approval_needed": False,
                    "approved": True,
                    "pending_action": ""
                }
            else:
                print("   ❌ Дію відхилено\n")
                return {
                    "messages": [HumanMessage(content="Користувач відхилив цю дію")],
                    "approval_needed": False,
                    "approved": False,
                    "pending_action": ""
                }

        return {"messages": [], "approval_needed": False, "approved": True, "pending_action": ""}

    def should_continue(state: ApprovalState) -> str:
        """Визначає наступний крок"""
        if state["approval_needed"]:
            return "get_approval"

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if state.get("approved", True):
                return "execute_tools"
            else:
                return "end"

        return "end"

    # Побудова графа
    workflow = StateGraph(ApprovalState)

    workflow.add_node("agent", agent)
    workflow.add_node("approval", human_approval)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "get_approval": "approval",
            "execute_tools": "tools",
            "end": END
        }
    )

    workflow.add_edge("approval", "agent")
    workflow.add_edge("tools", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    # Тестування
    config = {"configurable": {"thread_id": "approval_demo"}}

    print("Запит: Відправ email на test@example.com з темою 'Привіт'")
    result = app.invoke({
        "messages": [HumanMessage(content="Відправ email на test@example.com з темою 'Привіт' та текстом 'Тестове повідомлення'")],
        "approval_needed": False,
        "approved": False,
        "pending_action": ""
    }, config)

    print("Фінальний результат:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(f"  {msg.content}")
    print()

    return app


def demo_interrupt_workflow():
    """
    Демонстрація переривання workflow для людського вводу
    """
    print("=== Workflow з перериванням для вводу ===\n")

    class InterruptState(TypedDict):
        step: int
        user_input: str
        result: str

    def step_1(state: InterruptState) -> InterruptState:
        """Перший автоматичний крок"""
        print("  Крок 1: Автоматична обробка")
        return {
            "step": 1,
            "user_input": state.get("user_input", ""),
            "result": "Крок 1 виконано"
        }

    def wait_for_input(state: InterruptState) -> InterruptState:
        """Очікування вводу користувача"""
        print("  Крок 2: Очікування вводу користувача")

        if not state.get("user_input"):
            # В реальному сценарії тут буде паузка
            print("  💬 Потрібен ввід користувача")
            user_input = "[Симуляція вводу: Так, продовжуй]"
            print(f"  Отримано: {user_input}\n")
        else:
            user_input = state["user_input"]

        return {
            "step": 2,
            "user_input": user_input,
            "result": state["result"] + " -> Отримано ввід"
        }

    def step_3(state: InterruptState) -> InterruptState:
        """Фінальний крок"""
        print("  Крок 3: Фінальна обробка")
        return {
            "step": 3,
            "user_input": state["user_input"],
            "result": state["result"] + " -> Крок 3 виконано"
        }

    workflow = StateGraph(InterruptState)
    workflow.add_node("step_1", step_1)
    workflow.add_node("wait_input", wait_for_input)
    workflow.add_node("step_3", step_3)

    workflow.set_entry_point("step_1")
    workflow.add_edge("step_1", "wait_input")
    workflow.add_edge("wait_input", "step_3")
    workflow.add_edge("step_3", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "interrupt_demo"}}

    print("Запуск workflow:")
    result = app.invoke({"step": 0, "user_input": "", "result": ""}, config)

    print(f"\nФінальний результат: {result['result']}\n")

    return app


def demo_conditional_approval():
    """
    Демонстрація умовного схвалення на основі критеріїв
    """
    print("=== Умовне схвалення ===\n")

    class ConditionalState(TypedDict):
        action: str
        amount: float
        requires_approval: bool
        approved: bool

    APPROVAL_THRESHOLD = 1000.0

    def evaluate_action(state: ConditionalState) -> ConditionalState:
        """Оцінює чи потрібне схвалення"""
        requires_approval = state["amount"] > APPROVAL_THRESHOLD

        if requires_approval:
            print(f"  ⚠️  Сума {state['amount']} перевищує ліміт {APPROVAL_THRESHOLD}")
            print(f"  Потрібне схвалення для: {state['action']}")
        else:
            print(f"  ✅ Сума {state['amount']} в межах ліміту")
            print(f"  Автоматичне виконання: {state['action']}")

        return {
            "action": state["action"],
            "amount": state["amount"],
            "requires_approval": requires_approval,
            "approved": not requires_approval  # Автоматично схвалюємо якщо не потрібне підтвердження
        }

    def request_approval(state: ConditionalState) -> ConditionalState:
        """Запит схвалення"""
        print(f"\n  💬 Запит схвалення від керівника...")
        # Симуляція схвалення
        approved = True
        print(f"  {'✅ Схвалено' if approved else '❌ Відхилено'}\n")

        return {
            "action": state["action"],
            "amount": state["amount"],
            "requires_approval": state["requires_approval"],
            "approved": approved
        }

    def execute_action(state: ConditionalState) -> ConditionalState:
        """Виконання дії"""
        print(f"  ⚡ Виконується: {state['action']}")
        print(f"  Сума: {state['amount']}\n")

        return state

    def route_decision(state: ConditionalState) -> str:
        """Маршрутизація в залежності від потреби схвалення"""
        if state["requires_approval"]:
            return "approval"
        else:
            return "execute"

    # Побудова графа
    workflow = StateGraph(ConditionalState)

    workflow.add_node("evaluate", evaluate_action)
    workflow.add_node("approval", request_approval)
    workflow.add_node("execute", execute_action)

    workflow.set_entry_point("evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        route_decision,
        {
            "approval": "approval",
            "execute": "execute"
        }
    )

    workflow.add_edge("approval", "execute")
    workflow.add_edge("execute", END)

    app = workflow.compile()

    # Тестування різних сценаріїв
    test_cases = [
        {"action": "Покупка обладнання", "amount": 500.0},
        {"action": "Оренда офісу", "amount": 2500.0},
        {"action": "Канцтовари", "amount": 150.0},
    ]

    for test in test_cases:
        print(f"Тест: {test['action']} - ${test['amount']}")
        result = app.invoke({
            "action": test["action"],
            "amount": test["amount"],
            "requires_approval": False,
            "approved": False
        })

    return app


def demo_feedback_loop():
    """
    Демонстрація циклу зворотного зв'язку з людиною
    """
    print("=== Цикл зворотного зв'язку ===\n")

    class FeedbackState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        iteration: int
        feedback_received: bool

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    def generate_content(state: FeedbackState) -> FeedbackState:
        """Генерація контенту"""
        iteration = state.get("iteration", 0)
        print(f"  Генерація контенту (ітерація {iteration + 1})")

        response = model.invoke(state["messages"])

        return {
            "messages": [response],
            "iteration": iteration + 1,
            "feedback_received": False
        }

    def request_feedback(state: FeedbackState) -> FeedbackState:
        """Запит фідбеку"""
        print(f"\n  💬 Контент згенеровано:")
        print(f"     {state['messages'][-1].content[:100]}...")
        print(f"\n  Чи потрібні зміни? (симуляція фідбеку)")

        # Симуляція фідбеку (в реальності - ввід користувача)
        if state["iteration"] < 2:
            feedback = "Зроби більш стислим"
            needs_revision = True
        else:
            feedback = "Відмінно, схвалюю!"
            needs_revision = False

        print(f"  Фідбек: {feedback}\n")

        return {
            "messages": [HumanMessage(content=f"Фідбек: {feedback}")],
            "iteration": state["iteration"],
            "feedback_received": True
        }

    def should_continue(state: FeedbackState) -> str:
        """Визначає чи продовжувати"""
        if state["iteration"] >= 3:
            return "end"

        if state.get("feedback_received") and "схвалюю" not in state["messages"][-1].content.lower():
            return "revise"

        return "end"

    workflow = StateGraph(FeedbackState)

    workflow.add_node("generate", generate_content)
    workflow.add_node("feedback", request_feedback)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "feedback")

    workflow.add_conditional_edges(
        "feedback",
        should_continue,
        {
            "revise": "generate",
            "end": END
        }
    )

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "feedback_demo"}}

    print("Запуск циклу зворотного зв'язку:")
    result = app.invoke({
        "messages": [HumanMessage(content="Напиши короткий параграф про переваги LangGraph")],
        "iteration": 0,
        "feedback_received": False
    }, config)

    print(f"Всього ітерацій: {result['iteration']}")
    print(f"Фінальний контент:\n{result['messages'][-2].content}\n")

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 1.0 - Human-in-the-Loop Demonstration")
    print("=" * 60 + "\n")

    try:
        demo_approval_workflow()
        demo_interrupt_workflow()
        demo_conditional_approval()
        demo_feedback_loop()

        print("\n" + "=" * 60)
        print("Всі Human-in-the-Loop демонстрації завершені!")
        print("Примітка: В реальних сценаріях workflow буде призупинятись")
        print("для очікування вводу/схвалення від людини.")
        print("=" * 60)

    except Exception as e:
        print(f"Помилка: {e}")
        print("Переконайтесь, що ви налаштували .env файл")
