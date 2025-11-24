"""
БАЗОВИЙ АГЕНТ - LangChain 1.0 з реальними інструментами
На базі офіційної документації: create_react_agent

ВАЖЛИВО: Потребує langchain>=1.0.0 (див. requirements.txt)

LangSmith Integration: Автоматично ввімкнений через environment variables

РЕАЛЬНІ ІНСТРУМЕНТИ:
- Weather: OpenWeatherMap API
- Search: Tavily Search API
- Calculator: Безпечний numexpr
"""

import os
import requests
from langchain.tools import tool  # Офіційний імпорт згідно LangChain docs
# Updated import: deprecated TavilySearchResults replaced by TavilySearch in langchain-tavily package
try:
    from langchain_tavily import TavilySearch
except ImportError:
    TavilySearch = None  # Graceful fallback if lib not installed
from dotenv import load_dotenv
import numexpr as ne

# LangChain 1.0 API
from langchain.agents import create_agent

load_dotenv()

# ============================================================================
# LANGSMITH SETUP - Автоматичний трейсинг
# ============================================================================

if not os.getenv("LANGCHAIN_TRACING_V2"):
    print("WARNING  LangSmith трейсинг не ввімкнено. Додайте в .env:")
    print("LANGCHAIN_TRACING_V2=true")
    print("LANGCHAIN_API_KEY=your_key")
    print("LANGCHAIN_PROJECT=langchain-agents-v1\n")
else:
    print("OK LangSmith трейсинг активний")
    print(f"Stats: Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}\n")


# ============================================================================
# РЕАЛЬНІ TOOLS - БЕЗ МОКІВ
# ============================================================================

@tool
def get_weather(location: str) -> str:
    """
    Get current weather for a specific location using OpenWeatherMap API.

    Args:
        location: City name (e.g., 'London', 'Kyiv', 'New York')

    Returns:
        Real-time weather information as string
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")

    if not api_key:
        return (
            "ERROR: OPENWEATHERMAP_API_KEY not found. "
            "Get free API key at https://openweathermap.org/api"
        )

    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        return (
            f"Weather in {location}:\n"
            f"Conditions: {weather.capitalize()}\n"
            f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} m/s"
        )

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except KeyError as e:
        return f"Error parsing weather data. City might not be found: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Perform safe mathematical calculations using numexpr.

    Args:
        expression: Mathematical expression as string (e.g., '2 + 2', '10 * 5', 'sqrt(16)')

    Returns:
        Result of calculation

    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, exp, abs
    """
    try:
        # numexpr є безпечним - не виконує довільний Python код
        result = ne.evaluate(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information using Tavily Search API.

    Args:
        query: Search query

    Returns:
        Relevant search results from the web

    Use when user asks about current events, news, or information that needs web lookup.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return (
            "ERROR: TAVILY_API_KEY not found. "
            "Get free API key at https://tavily.com"
        )

    try:
        if TavilySearch is None:
            return (
                "ERROR: langchain-tavily not installed. Run: pip install -U langchain-tavily"
            )

        # Використовуємо оновлений TavilySearch tool (langchain-tavily)
        search_tool = TavilySearch(
            max_results=3,
            api_key=api_key
        )

        # New tool keeps same invoke contract with a dict
        results = search_tool.invoke({"query": query})

        if not results:
            return f"No results found for '{query}'"

        # Форматуємо результати
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            content = result.get('content', 'No description')

            formatted_results.append(
                f"{i}. {title}\n"
                f"   {content[:200]}...\n"
                f"   Source: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error searching web: {str(e)}"


@tool
def convert_currency(amount, from_currency, to_currency) -> str:
    """
    Convert a monetary `amount` from `from_currency` to `to_currency` using the
    ExchangeRate-API Pair Conversion endpoint.

    Args:
        amount: Numeric amount to convert (int/float or numeric string)
        from_currency: Source currency code (e.g., 'USD')
        to_currency: Target currency code (e.g., 'EUR')

    Returns:
        Formatted string with conversion result, rate and update time.

    Notes:
        - Uses ISO 4217 three-letter codes.
        - Endpoint docs attached in project.
        - Requires EXCHANGERATE_API_KEY set in environment (.env).
    """
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if not api_key:
        return (
            "ERROR: EXCHANGERATE_API_KEY not found. "
            "Get free API key at https://www.exchangerate-api.com"
        )

    # Normalize inputs
    try:
        amt = float(amount)
    except (ValueError, TypeError):
        return f"Error: 'amount' must be numeric (got: {amount})"

    base = str(from_currency).upper().strip()
    target = str(to_currency).upper().strip()

    if len(base) != 3 or len(target) != 3:
        return "Error: Currency codes must be 3-letter ISO 4217 codes"

    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base}/{target}/{amt}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("result") != "success":
            error_type = data.get("error-type", "unknown-error")
            return f"Error converting currency: {error_type}"

        rate = data.get("conversion_rate")
        converted = data.get("conversion_result")
        last_update = data.get("time_last_update_utc", "N/A")

        if rate is None or converted is None:
            return "Error: Incomplete response from ExchangeRate API"

        return (
            f"Currency Conversion:\n"
            f"  {amt:.2f} {base} -> {converted:.2f} {target}\n"
            f"  Rate: 1 {base} = {rate:.6f} {target}\n"
            f"  Last Update: {last_update}"
        )

    except requests.exceptions.RequestException as e:
        return f"Network/HTTP error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ============================================================================
# СТВОРЕННЯ БАЗОВОГО АГЕНТА - LangChain 1.0 API
# ============================================================================

def create_basic_agent():
    """
    Створює базового агента з LangChain 1.0 API та реальними інструментами

    Використовує create_agent (LangChain 1.0+)
    """
    print("=" * 70)
    print("🤖 БАЗОВИЙ АГЕНТ - LangChain 1.0 (РЕАЛЬНІ ІНСТРУМЕНТИ)")
    print("=" * 70 + "\n")

    # 1. Список реальних tools
    tools = [get_weather, calculate, web_search, convert_currency]

    print("Available tools (REAL APIs):")
    for tool_item in tools:
        print(f"  • {tool_item.name}: {tool_item.description[:53]}...")
    print()

    # 2. Створення агента з LangChain 1.0 API
    print("✅ Using LangChain 1.0+ create_agent API\n")

    agent = create_agent(
        model="gpt-4o",
        tools=tools,
        system_prompt="""You are a helpful AI assistant with access to REAL-TIME tools. Always choose a tool when factual / numeric data is requested.

TOOLS & USAGE RULES:
1. Weather (get_weather): Use ONLY for current weather in a city. Never guess weather.
2. Web Search (web_search): Use for current events, recent changes, news, or info not in your static memory. If the user asks for 'latest', 'current', 'recent', or cites a date in 2024/2025 -> use web_search.
3. Calculator (calculate): Use for pure mathematical expressions (includes sqrt, log, sin, ratios). If user asks multi-step math or arithmetic embedded in text, extract the expression and call calculate.
4. Currency Conversion (convert_currency): MANDATORY for ANY request about converting money OR asking an exchange rate between two currencies. NEVER hallucinate or approximate exchange rates.

CURRENCY CONVERSION INSTRUCTIONS:
- Trigger convert_currency when user says phrases like: "convert", "exchange", "how much is", "what is X USD in EUR", "rate USD to EUR", etc.
- If user asks "Convert 100 USD to EUR" -> amount=100, from_currency=USD, to_currency=EUR.
- If user asks only for the rate (e.g., "What's the USD to JPY rate?") call convert_currency with amount=1.
- Normalize currency codes to uppercase 3-letter ISO (usd -> USD). If non-ISO or slang (e.g., 'bucks') ASK the user to clarify rather than guessing.
- If API returns an error, surface the error concisely and invite the user to verify codes or retry.

DECISION LOGIC (STRICT):
- Any message containing pattern: /(convert|exchange|rate|USD|EUR|GBP|JPY|AUD|CAD|CHF|CNY|INR)/ AND two currency-like tokens -> use convert_currency.
- Do NOT use web_search for currency exchange unless specifically asked for sources or historical trends. For direct numeric conversion always use convert_currency.

MULTIPLE REQUESTS:
- If user combines weather + math + currency etc., you may call tools sequentially. Prioritize: weather first, then currency, then math, then web search if needed.

FORMAT OUTPUT:
- After a tool call, clearly label the section. Example:
  "[Tool: convert_currency]\n" then result.
- Do not add speculative commentary if tool provided authoritative data.

FAILURE HANDLING:
- If a tool fails, explain briefly and offer next steps (e.g., confirm city spelling, confirm currency code, retry later).

NEVER:
- Never fabricate exchange rates, weather details, or search results.
- Never bypass a tool for data it can provide.

GOAL: Precise, tool-backed answers. Always err toward invoking the correct tool rather than guessing."""
    )

    return agent


# ============================================================================
# ТЕСТУВАННЯ АГЕНТА
# ============================================================================

def test_basic_agent():
    """Тестує базового агента з реальними API викликами"""

    agent = create_basic_agent()

    # Тестові запити що використовують реальні API
    test_queries = [
        {
            "query": "Convert 100 USD to EUR",
            "expected_tool": "convert_currency"
        },
        {
            "query": "Convert 250 eur to gbp",
            "expected_tool": "convert_currency"
        },
        {
            "query": "What's the current weather in London?",
            "expected_tool": "get_weather"
        },
        {
            "query": "Calculate sqrt(144) + 25 * 2",
            "expected_tool": "calculate"
        },
        {
            "query": "Search for latest news about LangChain framework",
            "expected_tool": "web_search"
        },
        {
            "query": "What's the weather in Tokyo and calculate 100 / 4",
            "expected_tool": "multiple"
        }
    ]

    for i, query_data in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {query_data['query']}")
        print(f"Expected tool(s): {query_data['expected_tool']}")
        print("=" * 70 + "\n")

        try:
            # LangChain 1.0 create_agent invoke format
            result = agent.invoke({
                "messages": [{"role": "user", "content": query_data["query"]}]
            })

            # Extract output from messages
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                output = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                output = str(result)

            print("\n" + "-" * 70)
            print("✅ RESULT:")
            print("-" * 70)
            print(f"Output: {output}\n")

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
    print("TARGET LangChain 1.0 - Basic Agent with REAL Tools")
    print("=" * 70)
    print()
    print("Features:")
    print("  OK create_agent - LangChain 1.0 API (October 2025)")
    print("  OK Real Weather API - OpenWeatherMap")
    print("  OK Real Web Search - Tavily API")
    print("  OK Safe Calculator - numexpr")
    print("  OK LangSmith automatic tracing")
    print("  OK Direct agent invocation (no AgentExecutor)")
    print()
    print("=" * 70 + "\n")

    # Перевірка API ключів
    required_keys = {
        "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
        "OPENWEATHERMAP_API_KEY": "https://openweathermap.org/api",
        "TAVILY_API_KEY": "https://tavily.com"
    }

    missing_keys = []
    for key, url in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"  - {key}: Get at {url}")

    if missing_keys:
        print("ERROR: Missing required API keys:")
        print("\n".join(missing_keys))
        print("\nAdd them to your .env file")
        exit(1)

    try:
        test_basic_agent()

        print("\n" + "=" * 70)
        print("OK ALL TESTS COMPLETED")
        print("=" * 70)
        print("\nTIP: Check LangSmith dashboard to see traces:")
        print("   https://smith.langchain.com/\n")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: Error: {e}")
        import traceback
        traceback.print_exc()
