"""
MODULE 2: Агенти та інструменти
СЛАЙД 9-10: Basic Agent з Tools

Цей приклад демонструє:
- Створення агента з v1.0 API
- Визначення custom tools
- Tool calling process
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# ВИЗНАЧЕННЯ TOOLS
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """
    Виконує математичні обчислення.

    Args:
        expression: Математичний вираз як string (наприклад, "2 + 2" або "15 * 7")

    Returns:
        Результат обчислення як string

    Examples:
        calculator("2 + 2") -> "4"
        calculator("100 / 5") -> "20.0"
    """
    try:
        result = eval(expression)
        return f"Результат: {result}"
    except Exception as e:
        return f"Помилка обчислення: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    Отримує поточну погоду для вказаного міста.

    Args:
        city: Назва міста українською або англійською

    Returns:
        Інформація про погоду

    Note:
        Це mock функція. В production тут був би API виклик.
    """
    # Mock data - в production це був би реальний weather API
    weather_data = {
        "київ": "☀️ Сонячно, +22°C, вологість 65%",
        "львів": "⛅ Хмарно, +18°C, вологість 70%",
        "одеса": "🌧️ Дощ, +20°C, вологість 85%",
        "харків": "☀️ Сонячно, +24°C, вологість 60%",
    }

    city_lower = city.lower()
    return weather_data.get(city_lower, f"Погода для міста '{city}' недоступна")


@tool
def search_python_docs(query: str) -> str:
    """
    Шукає інформацію в документації Python.

    Args:
        query: Що шукати (функція, модуль, концепція)

    Returns:
        Знайдена інформація або посилання

    Use this when user asks about Python language features, built-in functions, or standard library.
    """
    # Mock documentation - в production це був би реальний search
    docs = {
        "декоратори": "Декоратори - це функції які модифікують поведінку інших функцій. Синтаксис: @decorator",
        "list comprehension": "List comprehension: [expression for item in iterable if condition]",
        "lambda": "Lambda - анонімна функція: lambda arguments: expression",
        "generators": "Generators використовують yield для ледачого обчислення: def gen(): yield value",
    }

    query_lower = query.lower()
    for key, value in docs.items():
        if key in query_lower:
            return f"📚 {value}\nДетальніше: https://docs.python.org"

    return f"Інформацію про '{query}' не знайдено. Спробуйте docs.python.org"


# ============================================================================
# DEMOS
# ============================================================================

def demo_single_tool_agent():
    """Агент з одним інструментом"""
    print("=" * 60)
    print("🤖 АГЕНТ З ОДНИМ ІНСТРУМЕНТОМ")
    print("=" * 60 + "\n")

    # Створюємо агента
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [calculator]

    agent = create_react_agent(model, tools)

    print("Доступні tools: calculator")
    print("\n📝 Питання: Скільки буде 123 * 456?\n")

    # Викликаємо агента
    result = agent.invoke({
        "messages": [("user", "Скільки буде 123 * 456?")]
    })

    # Виводимо результат
    print("🤖 Відповідь агента:")
    print(result["messages"][-1].content)
    print()


def demo_multi_tool_agent():
    """Агент з кількома інструментами"""
    print("=" * 60)
    print("🎭 АГЕНТ З КІЛЬКОМА ІНСТРУМЕНТАМИ")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [calculator, get_weather, search_python_docs]

    agent = create_react_agent(model, tools)

    print("Доступні tools:")
    for tool_obj in tools:
        print(f"  • {tool_obj.name}: {tool_obj.description[:50]}...")
    print()

    # Різні запити які потребують різних tools
    queries = [
        "Яка погода в Києві?",
        "Порахуй 2500 / 50",
        "Поясни що таке декоратори в Python"
    ]

    for i, query in enumerate(queries, 1):
        print(f"📌 Запит {i}: {query}")

        result = agent.invoke({
            "messages": [("user", query)]
        })

        print(f"🤖 Відповідь: {result['messages'][-1].content}\n")


def demo_agent_reasoning_process():
    """Показуємо процес міркування агента"""
    print("=" * 60)
    print("🧠 ПРОЦЕС МІРКУВАННЯ АГЕНТА")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [calculator, get_weather]

    agent = create_react_agent(model, tools)

    print("📝 Складне питання: Яка погода в Львові і скільки буде 15 + 27?\n")

    result = agent.invoke({
        "messages": [("user", "Яка погода в Львові і скільки буде 15 + 27?")]
    })

    print("🔍 Процес виконання:")
    print("-" * 60)

    for i, msg in enumerate(result["messages"], 1):
        msg_type = type(msg).__name__

        if msg_type == "HumanMessage":
            print(f"{i}. 👤 Користувач: {msg.content[:100]}")

        elif msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"{i}. 🤖 Агент викликає tool: {tool_call['name']}")
                    print(f"     Args: {tool_call['args']}")
            else:
                print(f"{i}. 🤖 Агент відповідає: {msg.content[:100]}")

        elif msg_type == "ToolMessage":
            print(f"{i}. 🔧 Tool результат: {msg.content[:100]}")

    print("-" * 60)
    print(f"\n✅ Фінальна відповідь:\n{result['messages'][-1].content}\n")


def demo_agent_with_system_prompt():
    """Агент з кастомним system message"""
    print("=" * 60)
    print("📋 АГЕНТ З КАСТОМНИМ SYSTEM MESSAGE")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    tools = [calculator, get_weather, search_python_docs]

    # Кастомний system message змінює поведінку
    system_message = """Ти дружній AI асистент-україномовний помічник.

ТВОЯ РОЛЬ:
- Завжди відповідай українською мовою
- Будь ввічливим та допомагай
- Використовуй емодзі для кращої комунікації
- Якщо не знаєш відповіді - так і скажи

ТВОЇ ІНСТРУМЕНТИ:
- calculator: для математичних обчислень
- get_weather: для інформації про погоду
- search_python_docs: для питань про Python

Завжди пояснюй чому використовуєш конкретний tool."""

    agent = create_react_agent(
        model,
        tools,
        state_modifier=system_message
    )

    print("🎭 System message налаштовано\n")

    queries = [
        "Привіт! Хто ти?",
        "Скільки буде 100 * 5?",
        "Яка погода в Одесі?"
    ]

    for query in queries:
        print(f"👤 {query}")

        result = agent.invoke({
            "messages": [("user", query)]
        })

        print(f"🤖 {result['messages'][-1].content}\n")


# ============================================================================
# WORKSHOP EXERCISE
# ============================================================================

def workshop_exercise():
    """
    ВПРАВА: Створіть code review agent
    """
    print("=" * 60)
    print("🎯 ВПРАВА: Code Review Agent")
    print("=" * 60 + "\n")

    print("Завдання:")
    print("Створіть агента з наступними tools:")
    print("  1. check_syntax: перевіряє синтаксис коду")
    print("  2. find_bugs: шукає потенційні баги")
    print("  3. suggest_improvements: пропонує покращення")
    print()
    print("Агент має приймати код і автоматично викликати потрібні tools.")
    print()

    code_sample = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
"""

    print("Тестовий код:")
    print(code_sample)

    input("\n⏸️  Натисніть Enter щоб побачити рішення...")
    show_solution()


def show_solution():
    """Рішення вправи"""
    print("\n" + "=" * 60)
    print("✅ РІШЕННЯ")
    print("=" * 60 + "\n")

    @tool
    def check_syntax(code: str) -> str:
        """Перевіряє синтаксис Python коду."""
        try:
            compile(code, "<string>", "exec")
            return "✅ Синтаксис коректний"
        except SyntaxError as e:
            return f"❌ Синтаксична помилка: {e}"

    @tool
    def find_bugs(code: str) -> str:
        """Шукає потенційні баги в коді."""
        bugs = []
        if "len(" in code and "/" in code:
            bugs.append("⚠️ Можливе ділення на нуль якщо список порожній")
        if "numbers[" in code and "len(" not in code:
            bugs.append("⚠️ Можливий IndexError")
        return "\n".join(bugs) if bugs else "✅ Очевидних багів не знайдено"

    @tool
    def suggest_improvements(code: str) -> str:
        """Пропонує покращення коду."""
        suggestions = []
        if "for" in code and "range(len(" not in code:
            suggestions.append("💡 Використовуйте enumerate() для ітерації")
        if "total = 0" in code:
            suggestions.append("💡 Розгляньте sum() функцію")
        return "\n".join(suggestions) if suggestions else "✅ Код виглядає добре"

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [check_syntax, find_bugs, suggest_improvements]

    agent = create_react_agent(
        model,
        tools,
        state_modifier="Ти code reviewer. Аналізуй код систематично використовуючи всі доступні tools."
    )

    code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
"""

    print("🔍 Аналізуємо код...\n")

    result = agent.invoke({
        "messages": [("user", f"Проаналізуй цей код:\n{code}")]
    })

    print("📊 Code Review Result:")
    print(result["messages"][-1].content)
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎓 MODULE 2: Агенти та Інструменти")
    print("=" * 60 + "\n")

    try:
        # Demo 1: Single tool
        demo_single_tool_agent()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 2: Multiple tools
        demo_multi_tool_agent()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 3: Reasoning process
        demo_agent_reasoning_process()
        input("⏸️  Натисніть Enter для продовження...")

        # Demo 4: System prompt
        demo_agent_with_system_prompt()
        input("⏸️  Натисніть Enter для вправи...")

        # Workshop exercise
        workshop_exercise()

        print("\n" + "=" * 60)
        print("✅ AGENTS MODULE ЗАВЕРШЕНО!")
        print("=" * 60)
        print("\n📝 Key Takeaways:")
        print("  1. create_react_agent() - новий v1.0 API")
        print("  2. @tool decorator для створення tools")
        print("  3. Агент сам вирішує які tools викликати")
        print("  4. System message контролює поведінку")

    except Exception as e:
        print(f"\n❌ Помилка: {e}")
