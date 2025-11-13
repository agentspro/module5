"""
АГЕНТ З CALLBACKS - LangChain 1.0 ОФІЦІЙНИЙ API
Демонструє розширення агентів через callbacks (офіційний механізм LangChain 1.0)

ОФІЦІЙНИЙ LANGCHAIN 1.0 CALLBACKS API:
- BaseCallbackHandler для custom callbacks
- on_llm_start: Викликається перед LLM
- on_llm_end: Викликається після LLM
- on_tool_start: Викликається перед tool
- on_tool_end: Викликається після tool
- on_agent_action: Викликається при action агента

LangSmith Integration: Автоматично трейсить всі callback operations
"""

import os
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

# ============================================================================
# LANGSMITH VERIFICATION
# ============================================================================

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("✅ LangSmith трейсинг активний")
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print("🔍 Callback operations will be traced\n")
else:
    print("⚠️  LangSmith не ввімкнений\n")


# ============================================================================
# TOOLS
# ============================================================================

@tool
def get_stock_price(symbol: str) -> str:
    """Get real-time stock price using yfinance API."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")

        if data.empty:
            return f"No data found for symbol {symbol}"

        current_price = data['Close'].iloc[-1]
        return f"${current_price:.2f}"

    except Exception as e:
        return f"Error fetching price for {symbol}: {str(e)}"


@tool
def send_notification(message: str, recipient: str) -> str:
    """
    Send notification to user. This is a HIGH-RISK action.

    Args:
        message: Notification message
        recipient: Recipient email or ID
    """
    return f"✅ Notification sent to {recipient}: {message}"


@tool
def execute_trade(symbol: str, quantity: int, action: str) -> str:
    """
    Execute a trade. HIGH-RISK action.

    Args:
        symbol: Stock symbol
        quantity: Number of shares
        action: 'buy' or 'sell'
    """
    return f"⚠️  Would execute {action} {quantity} shares of {symbol}"


# ============================================================================
# CUSTOM CALLBACK HANDLERS - LangChain 1.0 ОФІЦІЙНИЙ API
# ============================================================================

class LoggingCallback(BaseCallbackHandler):
    """
    Офіційний LangChain 1.0 Callback Handler для детального логування
    """

    def __init__(self):
        super().__init__()
        self.llm_calls = 0
        self.tool_calls = 0
        self.logs = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Викликається ПЕРЕД кожним викликом LLM"""
        self.llm_calls += 1

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "llm_start",
            "call_number": self.llm_calls,
            "prompt_length": len(prompts[0]) if prompts else 0
        }
        self.logs.append(log_entry)

        print(f"\n{'='*60}")
        print(f"📝 LOGGING CALLBACK: LLM Call #{self.llm_calls} Started")
        print(f"⏰ Time: {log_entry['timestamp']}")
        print(f"📏 Prompt length: {log_entry['prompt_length']} chars")
        print(f"{'='*60}\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Викликається ПІСЛЯ кожного виклику LLM"""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "llm_end",
            "call_number": self.llm_calls,
            "generations": len(response.generations)
        }
        self.logs.append(log_entry)

        print(f"\n{'='*60}")
        print(f"✅ LOGGING CALLBACK: LLM Call #{self.llm_calls} Completed")
        print(f"⏰ Time: {log_entry['timestamp']}")
        print(f"📊 Generations: {log_entry['generations']}")
        print(f"{'='*60}\n")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """Викликається ПЕРЕД кожним викликом tool"""
        self.tool_calls += 1

        tool_name = serialized.get("name", "unknown")

        print(f"\n{'='*60}")
        print(f"🔧 TOOL CALL #{self.tool_calls}: {tool_name}")
        print(f"📥 Input: {input_str}")
        print(f"{'='*60}\n")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Викликається ПІСЛЯ кожного виклику tool"""

        print(f"\n{'='*60}")
        print(f"✅ TOOL COMPLETED")
        print(f"📤 Output: {output[:100]}...")
        print(f"{'='*60}\n")

    def get_stats(self):
        """Повертає статистику викликів"""
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "total_logs": len(self.logs)
        }


class SecurityCallback(BaseCallbackHandler):
    """
    Callback для перехоплення та блокування небезпечних дій
    """

    def __init__(self):
        super().__init__()
        self.high_risk_tools = ["execute_trade", "send_notification"]
        self.blocked_calls = 0

    def on_agent_action(
        self,
        action: AgentAction,
        **kwargs: Any
    ) -> None:
        """Перехоплює дії агента перед виконанням"""

        tool_name = action.tool

        if tool_name in self.high_risk_tools:
            self.blocked_calls += 1

            print(f"\n{'='*60}")
            print(f"🔒 SECURITY CALLBACK: HIGH-RISK ACTION DETECTED")
            print(f"⚠️  Tool: {tool_name}")
            print(f"📋 Input: {action.tool_input}")
            print(f"🚫 This would be blocked in production")
            print(f"   Total blocked: {self.blocked_calls}")
            print(f"{'='*60}\n")

    def get_stats(self):
        """Повертає статистику блокувань"""
        return {
            "blocked_calls": self.blocked_calls,
            "high_risk_tools": self.high_risk_tools
        }


class TokenCountCallback(BaseCallbackHandler):
    """
    Callback для підрахунку використаних токенів
    """

    def __init__(self, max_tokens: int = 10000):
        super().__init__()
        self.max_tokens = max_tokens
        self.total_tokens = 0
        self.calls_over_limit = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Оцінює кількість токенів перед викликом"""

        # Приблизна оцінка токенів (1 токен ≈ 4 символи для англійської мови)
        estimated_tokens = sum(len(p) // 4 for p in prompts)
        self.total_tokens += estimated_tokens

        print(f"\n{'='*60}")
        print(f"📊 TOKEN COUNTER CALLBACK:")
        print(f"   Estimated input tokens: ~{estimated_tokens}")
        print(f"   Total tokens used: {self.total_tokens}")
        print(f"   Max allowed: {self.max_tokens}")

        if self.total_tokens > self.max_tokens:
            self.calls_over_limit += 1
            print(f"   ⚠️  WARNING: Approaching token limit!")

        print(f"{'='*60}\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Враховує токени у відповіді"""

        # Якщо є token_usage в llm_output
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            if "total_tokens" in usage:
                actual_tokens = usage["total_tokens"]
                print(f"   📈 Actual tokens used: {actual_tokens}")

    def get_stats(self):
        """Повертає статистику використання"""
        return {
            "total_tokens": self.total_tokens,
            "calls_over_limit": self.calls_over_limit,
            "max_tokens": self.max_tokens
        }


# ============================================================================
# СТВОРЕННЯ АГЕНТА З CALLBACKS - ОФІЦІЙНИЙ API
# ============================================================================

def create_agent_with_callbacks():
    """
    Створює агента з callback handlers використовуючи ОФІЦІЙНИЙ LangChain 1.0 API

    Callbacks дозволяють:
    - Логувати всі операції
    - Моніторити використання токенів
    - Перехоплювати небезпечні дії
    - Додавати custom логіку без зміни агента
    """
    print("=" * 70)
    print("🤖 АГЕНТ З CALLBACKS - LangChain 1.0 (ОФІЦІЙНИЙ API)")
    print("=" * 70 + "\n")

    # Створюємо callback instances
    logging_cb = LoggingCallback()
    security_cb = SecurityCallback()
    token_cb = TokenCountCallback(max_tokens=10000)

    # Tools
    tools = [get_stock_price, send_notification, execute_trade]

    print("Available tools:")
    for tool_item in tools:
        risk = " (HIGH-RISK)" if tool_item.name in security_cb.high_risk_tools else ""
        print(f"  • {tool_item.name}{risk}")
    print()

    print("Callback handlers (ОФІЦІЙНИЙ LangChain 1.0 API):")
    print("  1. LoggingCallback (on_llm_start + on_llm_end + on_tool_*)")
    print("  2. SecurityCallback (on_agent_action)")
    print("  3. TokenCountCallback (on_llm_start + on_llm_end)")
    print()

    # Створюємо агента з LangChain 1.0 API
    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""You are a helpful financial assistant with access to tools.

IMPORTANT: When considering high-risk actions like execute_trade or send_notification, always explain why you would use them.

Think step-by-step and use tools when needed to answer questions accurately."""
    )

    return agent, logging_cb, security_cb, token_cb


# ============================================================================
# ТЕСТУВАННЯ АГЕНТА З CALLBACKS
# ============================================================================

def test_agent_with_callbacks():
    """Тестує агента з різними callback scenarios"""

    agent_executor, logging_cb, security_cb, token_cb = create_agent_with_callbacks()

    test_queries = [
        {
            "query": "What's the current price of AAPL stock?",
            "description": "Safe query - callbacks log everything",
            "expected": "get_stock_price tool call"
        },
        {
            "query": "Get TSLA price and send me notification about it",
            "description": "Contains HIGH-RISK tool - security callback detects it",
            "expected": "SecurityCallback logs HIGH-RISK action"
        },
        {
            "query": "Execute trade: buy 100 shares of GOOGL",
            "description": "HIGH-RISK action - security callback warns",
            "expected": "SecurityCallback detects execute_trade"
        }
    ]

    for i, test in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {test['description']}")
        print("=" * 70)
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print("-" * 70 + "\n")

        try:
            # LangChain 1.0 create_agent invoke з callbacks
            result = agent.invoke({
                "messages": [{"role": "user", "content": test["query"]}]
            }, config={"callbacks": [logging_cb, security_cb, token_cb]})

            # Extract output from messages
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                output = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                output = str(result)

            print("\n" + "-" * 70)
            print("📋 RESULT:")
            print("-" * 70)
            print(f"Output: {output}\n")

        except Exception as e:
            print(f"\n❌ ERROR: {e}\n")
            import traceback
            traceback.print_exc()

        input("\n⏸️  Press Enter to continue to next test...\n")

    # Виводимо статистику всіх callbacks
    print("\n" + "=" * 70)
    print("📊 CALLBACK STATISTICS")
    print("=" * 70 + "\n")

    print("Logging Callback:")
    logging_stats = logging_cb.get_stats()
    print(f"  LLM calls: {logging_stats['llm_calls']}")
    print(f"  Tool calls: {logging_stats['tool_calls']}")
    print(f"  Total logs: {logging_stats['total_logs']}")
    print()

    print("Security Callback:")
    security_stats = security_cb.get_stats()
    print(f"  Blocked calls: {security_stats['blocked_calls']}")
    print(f"  High-risk tools: {', '.join(security_stats['high_risk_tools'])}")
    print()

    print("Token Counter Callback:")
    token_stats = token_cb.get_stats()
    print(f"  Total tokens: {token_stats['total_tokens']}")
    print(f"  Calls over limit: {token_stats['calls_over_limit']}")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("🎯 LangChain 1.0 - Agent with Official Callback API")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✅ ОФІЦІЙНИЙ BaseCallbackHandler API")
    print("  ✅ on_llm_start + on_llm_end hooks")
    print("  ✅ on_tool_start + on_tool_end hooks")
    print("  ✅ on_agent_action для перехоплення дій")
    print("  ✅ Real financial data (yfinance)")
    print("  ✅ Security callback (detects risky actions)")
    print("  ✅ Token counting callback")
    print("  ✅ LangSmith automatic tracing")
    print()
    print("=" * 70 + "\n")

    # Перевірка API ключів
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in environment!")
        print("Please set it in .env file")
        exit(1)

    try:
        test_agent_with_callbacks()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 70)
        print("\n💡 Check LangSmith dashboard to see callback traces:")
        print("   https://smith.langchain.com/\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
