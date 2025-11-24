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
import re
import socket
from langchain.tools import tool  # Офіційний імпорт згідно LangChain docs
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import numexpr as ne
try:
    import dns.resolver  # dnspython for MX/A lookup
    _dns_available = True
except ImportError:
    _dns_available = False

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
        # Використовуємо Tavily для пошуку
        search_tool = TavilySearchResults(
            max_results=3,
            api_key=api_key
        )

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
def validate_email(email: str) -> str:
    """
    Validate an email address using regex and DNS lookup.

    Steps:
    1. Regex pattern validation (RFC-like simplified)
    2. Domain extraction
    3. DNS MX record lookup (fallback to A record if MX not found)

    Args:
        email: Email address string (e.g., 'user@example.com')

    Returns:
        Multi-line validation report indicating validity and discovered DNS info.

    Notes:
    - Regex is a pragmatic pattern (not full RFC 5322 coverage)
    - DNS lookup requires 'dnspython' package; if unavailable, falls back to socket resolution
    - Absence of MX and A records marks domain as invalid
    """
    email = email.strip()

    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    if not re.match(pattern, email):
        return (
            "Email Validation Report:\n"
            f"Input: {email}\n"
            "Status: INVALID\n"
            "Reason: Regex pattern mismatch (basic format invalid)."
        )

    local_part, domain = email.rsplit("@", 1)
    report_lines = [
        "Email Validation Report:",
        f"Input: {email}",
        "Step 1: Regex OK",\
        f"Local Part: {local_part}",
        f"Domain: {domain}",
    ]

    mx_records = []
    a_records = []
    dns_errors = []

    # DNS MX lookup
    if _dns_available:
        try:
            answers = dns.resolver.resolve(domain, 'MX')
            for rdata in answers:
                mx_records.append(str(rdata.exchange).rstrip('.'))
        except Exception as e:
            dns_errors.append(f"MX lookup failed: {e}")
        # Fallback A record
        if not mx_records:
            try:
                answers_a = dns.resolver.resolve(domain, 'A')
                for rdata in answers_a:
                    a_records.append(rdata.address)
            except Exception as e:
                dns_errors.append(f"A lookup failed: {e}")
    else:
        report_lines.append("DNS library (dnspython) not installed; using socket fallback.")
        try:
            ip = socket.gethostbyname(domain)
            a_records.append(ip)
        except Exception as e:
            dns_errors.append(f"Socket resolution failed: {e}")

    if mx_records:
        report_lines.append("Step 2: MX records found")
        for mx in mx_records:
            report_lines.append(f"  MX: {mx}")
    elif a_records:
        report_lines.append("Step 2: No MX - A record fallback successful")
        for ip in a_records:
            report_lines.append(f"  A: {ip}")
    else:
        report_lines.append("Step 2: Domain resolution FAILED (no MX/A)")

    if dns_errors:
        report_lines.append("DNS Errors:")
        for err in dns_errors:
            report_lines.append(f"  - {err}")

    is_valid = bool(mx_records or a_records)
    report_lines.append(f"Final Status: {'VALID' if is_valid else 'INVALID'}")
    if not is_valid:
        report_lines.append("Reason: Domain does not resolve (no usable DNS records).")

    return "\n".join(report_lines)


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
    tools = [get_weather, calculate, web_search, validate_email]

    print("Available tools (REAL APIs):")
    for tool_item in tools:
        print(f"  • {tool_item.name}: {tool_item.description[:60]}...")
    print()

    # 2. Створення агента з LangChain 1.0 API
    print("✅ Using LangChain 1.0+ create_agent API\n")

    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""You are a helpful AI assistant with access to real-time tools.

You have access to:
- Real-time weather data via OpenWeatherMap API
- Web search via Tavily API for current information
- Safe calculator for mathematical operations

Use the appropriate tool for each request and provide accurate, helpful responses.
When using tools, explain what you're doing and present results clearly."""
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
            "query": "Is test@example.com a valid email?",
            "expected_tool": "validate_email"
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
