"""
БАЗОВИЙ АГЕНТ - LangChain 1.0 API
На базі офіційної документації: create_agent з LangChain 1.0

LangSmith Integration: Автоматично ввімкнений через environment variables
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent, AgentExecutor
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
    print("⚠️  LangSmith трейсинг не ввімкнено. Додайте в .env:")
    print("LANGCHAIN_TRACING_V2=true")
    print("LANGCHAIN_API_KEY=your_key")
    print("LANGCHAIN_PROJECT=langchain-agents-v1\n")
else:
    print("✅ LangSmith трейсинг активний")
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}\n")


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
        "london": "🌧️ Rainy, 12°C",
        "kyiv": "☀️ Sunny, 18°C",
        "new york": "⛅ Partly cloudy, 15°C",
        "tokyo": "🌸 Clear, 22°C",
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
            results.append(f"📚 {topic.title()}: {info}")

    return "\n\n".join(results) if results else f"No documentation found for '{query}'"


# ============================================================================
# СТВОРЕННЯ БАЗОВОГО АГЕНТА - LangChain 1.0 API
# ============================================================================

def create_basic_agent():
    """
    Створює базового агента з LangChain 1.0 API

    Uses:
    - create_agent: Створює агента з LangChain 1.0 (October 2025 API)
    - AgentExecutor: Виконує агента з tools
    """
    print("=" * 70)
    print("🤖 БАЗОВИЙ АГЕНТ - LangChain 1.0")
    print("=" * 70 + "\n")

    # 1. Ініціалізація LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # or gpt-3.5-turbo for cheaper option
        temperature=0,
        model_kwargs={"response_format": {"type": "text"}}
    )

    # 2. Список tools
    tools = [get_weather, calculate, search_docs]

    print("Available tools:")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description[:60]}...")
    print()

    # 3. Створення агента (LangChain 1.0 API)
    # create_agent автоматично створює оптимальний промпт
    agent = create_agent(
        llm=llm,
        tools=tools
    )

    # 4. Створення Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Show reasoning process
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True
    )

    return agent_executor


# ============================================================================
# ТЕСТУВАННЯ АГЕНТА
# ============================================================================

def test_basic_agent():
    """Тестує базового агента з різними запитами"""

    agent = create_basic_agent()

    # Тестові запити що вимагають різних tools
    test_queries = [
        {
            "input": "What's the weather in Kyiv?",
            "expected_tool": "get_weather"
        },
        {
            "input": "Calculate 123 * 456",
            "expected_tool": "calculate"
        },
        {
            "input": "What is LangChain and how do I create agents?",
            "expected_tool": "search_docs"
        },
        {
            "input": "What's the weather in Tokyo and what's 50 + 50?",
            "expected_tool": "multiple"
        }
    ]

    for i, query_data in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {query_data['input']}")
        print(f"Expected tool(s): {query_data['expected_tool']}")
        print("=" * 70 + "\n")

        try:
            result = agent.invoke({"input": query_data["input"]})

            print("\n" + "-" * 70)
            print("RESULT:")
            print("-" * 70)
            print(f"Output: {result['output']}\n")

            # Show which tools were used
            if result.get('intermediate_steps'):
                print("Tools used:")
                for step in result['intermediate_steps']:
                    action, observation = step
                    print(f"  → {action.tool}: {action.tool_input}")

        except Exception as e:
            print(f"\n❌ Error: {e}\n")

        input("\n⏸️  Press Enter to continue to next test...\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎯 LangChain 1.0 - Basic Agent with LangSmith Tracing")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✅ create_agent - LangChain 1.0 API (October 2025)")
    print("  ✅ Multiple tools (weather, calculator, docs)")
    print("  ✅ Automatic optimal prompting")
    print("  ✅ LangSmith automatic tracing")
    print("  ✅ Error handling and max iterations")
    print()
    print("=" * 70 + "\n")

    # Перевірка API ключів
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in environment!")
        print("Please set it in .env file")
        exit(1)

    try:
        test_basic_agent()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 70)
        print("\n💡 Check LangSmith dashboard to see traces:")
        print("   https://smith.langchain.com/\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
