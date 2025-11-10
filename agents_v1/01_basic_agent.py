"""
БАЗОВИЙ АГЕНТ - LangChain 1.0 API
На базі офіційної документації: create_agent з LangChain 1.0

LangSmith Integration: Автоматично ввімкнений через environment variables
"""

import os
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# LANGSMITH SETUP - Автоматичний трейсинг
# ============================================================================

# LangSmith автоматично підключається якщо є env variables:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_key
# LANGCHAIN_PROJECT=langchain-agents-v1

if not os.getenv("LANGCHAIN_TRACING_V2"):
    print("WARNING  LangSmith трейсинг не ввімкнено. Додайте в .env:")
    print("LANGCHAIN_TRACING_V2=true")
    print("LANGCHAIN_API_KEY=your_key")
    print("LANGCHAIN_PROJECT=langchain-agents-v1\n")
else:
    print("OK LangSmith трейсинг активний")
    print(f"Stats: Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}\n")


# ============================================================================
# ВИЗНАЧЕННЯ TOOLS
# ============================================================================

@tool
def get_weather(location: str) -> str:
    """
    Get current weather for a specific location.

    Args:
        location: City name (e.g., 'London', 'Kyiv', 'New York')

    Returns:
        Weather information as string
    """
    # Mock implementation - в production це був би API виклик
    weather_db = {
        "london": "Rainy Rainy, 12°C",
        "kyiv": "Sunny Sunny, 18°C",
        "new york": "Partly cloudy Partly cloudy, 15°C",
        "tokyo": "Clear Clear, 22°C",
    }

    location_lower = location.lower()
    for city, weather in weather_db.items():
        if city in location_lower:
            return f"Weather in {location}: {weather}"

    return f"Weather data for {location} not available"


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.

    Args:
        expression: Mathematical expression as string (e.g., '2 + 2', '10 * 5')

    Returns:
        Result of calculation

    Use this when user asks for math operations, calculations, or numeric operations.
    """
    try:
        # Note: eval() not safe for production - use safe_eval or ast.literal_eval
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def search_docs(query: str) -> str:
    """
    Search technical documentation and knowledge base.

    Args:
        query: Search query

    Returns:
        Relevant documentation snippets

    Use when user asks about technical concepts, how-to questions, or needs documentation.
    """
    # Mock documentation database
    docs = {
        "langchain": "LangChain is a framework for developing applications powered by language models. Version 1.0 introduces stable APIs with create_agent.",
        "python": "Python is a high-level programming language known for readability and versatility.",
        "agents": "Agents in LangChain use LLMs to determine which tools to use and in what sequence to achieve a goal.",
        "tools": "Tools are functions that agents can call. Define tools using @tool decorator or StructuredTool class.",
    }

    query_lower = query.lower()
    results = []
    for topic, info in docs.items():
        if topic in query_lower or query_lower in info.lower():
            results.append(f"KB: {topic.title()}: {info}")

    return "\n\n".join(results) if results else f"No documentation found for '{query}'"


# ============================================================================
# СТВОРЕННЯ БАЗОВОГО АГЕНТА - LangChain 1.0 API
# ============================================================================

def create_basic_agent():
    """
    Створює базового агента з LangChain 1.0 API

    В LangChain 1.0:
    - create_agent приймає model (string), tools (list), system_prompt (string)
    - Автоматично створює оптимальний промпт
    - Повертає agent який можна викликати через .invoke()
    - Не потрібен AgentExecutor
    """
    print("=" * 70)
    print("AGENT БАЗОВИЙ АГЕНТ - LangChain 1.0")
    print("=" * 70 + "\n")

    # 1. Список tools
    tools = [get_weather, calculate, search_docs]

    print("Available tools:")
    for tool_item in tools:
        print(f"  • {tool_item.name}: {tool_item.description[:60]}...")
    print()

    # 2. Створення агента (LangChain 1.0 API)
    # Передаємо model як string, а не ChatOpenAI об'єкт!
    agent = create_agent(
        model="gpt-4o-mini",  # або "gpt-3.5-turbo"
        tools=tools,
        system_prompt="""You are a helpful AI assistant with access to tools.

Use the available tools to answer user questions accurately.
When you need information, use the appropriate tool.
Always provide clear, helpful responses."""
    )

    return agent


# ============================================================================
# ТЕСТУВАННЯ АГЕНТА
# ============================================================================

def test_basic_agent():
    """Тестує базового агента з різними запитами"""

    agent = create_basic_agent()

    # Тестові запити що вимагають різних tools
    test_queries = [
        {
            "query": "What's the weather in Kyiv?",
            "expected_tool": "get_weather"
        },
        {
            "query": "Calculate 123 * 456",
            "expected_tool": "calculate"
        },
        {
            "query": "What is LangChain and how do I create agents?",
            "expected_tool": "search_docs"
        },
        {
            "query": "What's the weather in Tokyo and what's 50 + 50?",
            "expected_tool": "multiple"
        }
    ]

    for i, query_data in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {query_data['query']}")
        print(f"Expected tool(s): {query_data['expected_tool']}")
        print("=" * 70 + "\n")

        try:
            # LangChain 1.0 API: invoke приймає messages
            result = agent.invoke({
                "messages": [{"role": "user", "content": query_data["query"]}]
            })

            print("\n" + "-" * 70)
            print("RESULT:")
            print("-" * 70)

            # Результат може бути в різних форматах залежно від версії
            if isinstance(result, dict):
                if "messages" in result:
                    # Витягуємо останнє повідомлення
                    last_message = result["messages"][-1]
                    if hasattr(last_message, "content"):
                        print(f"Output: {last_message.content}\n")
                    else:
                        print(f"Output: {last_message}\n")
                elif "output" in result:
                    print(f"Output: {result['output']}\n")
                else:
                    print(f"Output: {result}\n")
            else:
                print(f"Output: {result}\n")

        except Exception as e:
            print(f"\nERROR: Error: {e}\n")
            import traceback
            traceback.print_exc()

        input("\nPAUSE  Press Enter to continue to next test...\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("TARGET LangChain 1.0 - Basic Agent with LangSmith Tracing")
    print("=" * 70)
    print()
    print("Features:")
    print("  OK create_agent - LangChain 1.0 API (October 2025)")
    print("  OK Model as string parameter (not ChatOpenAI object)")
    print("  OK Multiple tools (weather, calculator, docs)")
    print("  OK Automatic optimal prompting")
    print("  OK LangSmith automatic tracing")
    print("  OK Direct agent invocation (no AgentExecutor)")
    print()
    print("=" * 70 + "\n")

    # Перевірка API ключів
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: ERROR: OPENAI_API_KEY not found in environment!")
        print("Please set it in .env file")
        exit(1)

    try:
        test_basic_agent()

        print("\n" + "=" * 70)
        print("OK ALL TESTS COMPLETED")
        print("=" * 70)
        print("\nTIP: Check LangSmith dashboard to see traces:")
        print("   https://smith.langchain.com/\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: Error: {e}")
        import traceback
        traceback.print_exc()
