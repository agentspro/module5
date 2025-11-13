"""
АГЕНТ З MIDDLEWARE - LangChain 1.0 ОФІЦІЙНИЙ API
На базі офіційної документації: AgentMiddleware API (2025)

ОФІЦІЙНИЙ LANGCHAIN 1.0 MIDDLEWARE API:
- AgentMiddleware базовий клас
- before_model: Runs before model calls
- after_model: Runs after model calls
- wrap_model_call: Modify request/response

LangSmith Integration: Автоматично трейсить всі middleware operations
"""

import os
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelRequestHandler
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

# ============================================================================
# LANGSMITH VERIFICATION
# ============================================================================

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("OK LangSmith трейсинг активний")
    print(f"Stats: Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print("Middleware operations will be traced\n")
else:
    print("WARNING  LangSmith не ввімкнений\n")


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
    Send notification to user. REQUIRES APPROVAL in middleware.

    Args:
        message: Notification message
        recipient: Recipient email or ID
    """
    return f"OK Notification sent to {recipient}: {message}"


@tool
def execute_trade(symbol: str, quantity: int, action: str) -> str:
    """
    Execute a trade. HIGH-RISK action requiring approval.

    Args:
        symbol: Stock symbol
        quantity: Number of shares
        action: 'buy' or 'sell'
    """
    return f"WARNING Would execute {action} {quantity} shares of {symbol}"


# ============================================================================
# ОФІЦІЙНІ MIDDLEWARE КЛАСИ - LangChain 1.0 API
# ============================================================================

class LoggingMiddleware(AgentMiddleware):
    """
    Офіційний LangChain 1.0 Middleware для логування
    Використовує before_model та after_model hooks
    """

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.logs = []

    def before_model(self, state: AgentState, runtime: Any) -> Optional[Dict[str, Any]]:
        """
        Виконується ПЕРЕД кожним викликом моделі

        Args:
            state: Поточний стан агента
            runtime: Runtime об'єкт з контекстом

        Returns:
            Dict з оновленнями стану або None
        """
        self.call_count += 1

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_number": self.call_count,
            "event": "before_model",
            "message_count": len(state.get("messages", [])),
        }

        self.logs.append(log_entry)

        print(f"\n{'='*60}")
        print(f"LOG MIDDLEWARE: Before Model Call #{self.call_count}")
        print(f"Time: {log_entry['timestamp']}")
        print(f"Stats: Messages: {log_entry['message_count']}")
        print(f"{'='*60}\n")

        # Повертаємо None - не змінюємо state
        return None

    def after_model(self, state: AgentState, runtime: Any) -> Optional[Dict[str, Any]]:
        """
        Виконується ПІСЛЯ кожного виклику моделі

        Args:
            state: Оновлений стан після виклику моделі
            runtime: Runtime об'єкт з контекстом

        Returns:
            Dict з оновленнями стану або None
        """
        last_message = state.get("messages", [])[-1] if state.get("messages") else None

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_number": self.call_count,
            "event": "after_model",
            "response_type": type(last_message).__name__ if last_message else "None",
        }

        self.logs.append(log_entry)

        print(f"\n{'='*60}")
        print(f"OK MIDDLEWARE: After Model Call #{self.call_count}")
        print(f"Time: {log_entry['timestamp']}")
        print(f"Output: {log_entry['response_type']}")
        print(f"{'='*60}\n")

        return None

    def get_stats(self):
        """Повертає статистику викликів"""
        return {
            "total_calls": self.call_count,
            "logs": self.logs
        }


class SecurityMiddleware(AgentMiddleware):
    """
    Офіційний LangChain 1.0 Middleware для безпеки
    Використовує wrap_model_call для модифікації запитів
    """

    def __init__(self):
        super().__init__()
        self.high_risk_tools = ["execute_trade", "send_notification"]
        self.calls_blocked = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: ModelRequestHandler
    ) -> AIMessage:
        """
        Обгортає виклик моделі для перевірки безпеки

        Args:
            request: ModelRequest з tools, messages, model
            handler: Функція для виконання запиту

        Returns:
            AIMessage з результатом
        """
        # Перевіряємо чи є high-risk tools у запиті
        messages_text = " ".join([str(m) for m in request.state.get("messages", [])])

        has_risky_request = any(
            tool_name.lower() in messages_text.lower()
            for tool_name in self.high_risk_tools
        )

        if has_risky_request:
            print(f"\nSECURITY SECURITY MIDDLEWARE:")
            print(f"   Detected HIGH-RISK tool request")
            print(f"   Tools: {', '.join(self.high_risk_tools)}")
            print(f"   ACTION: Filtering out risky tools\n")

            self.calls_blocked += 1

            # Фільтруємо risky tools
            safe_tools = [
                tool for tool in request.tools
                if tool.name not in self.high_risk_tools
            ]

            # Створюємо новий request без risky tools
            modified_request = request.replace(tools=safe_tools)

            return handler(modified_request)

        # Якщо нема ризикових інструментів - пропускаємо як є
        return handler(request)

    def get_stats(self):
        """Повертає статистику блокувань"""
        return {
            "calls_blocked": self.calls_blocked,
            "high_risk_tools": self.high_risk_tools
        }


class TokenLimitMiddleware(AgentMiddleware):
    """
    Офіційний LangChain 1.0 Middleware для контролю token usage
    Використовує before_model для перевірки лімітів
    """

    def __init__(self, max_tokens_per_call: int = 4000):
        super().__init__()
        self.max_tokens_per_call = max_tokens_per_call
        self.total_tokens_used = 0
        self.calls_throttled = 0

    def before_model(self, state: AgentState, runtime: Any) -> Optional[Dict[str, Any]]:
        """
        Перевіряє і обмежує token usage перед викликом

        Args:
            state: Поточний стан
            runtime: Runtime контекст

        Returns:
            Dict з оновленнями або None
        """
        # Оцінюємо кількість токенів (приблизно)
        messages = state.get("messages", [])
        total_text = " ".join([str(m) for m in messages])
        estimated_tokens = len(total_text.split()) * 1.3  # Rough estimate

        print(f"\nTOKEN TOKEN MIDDLEWARE:")
        print(f"   Estimated input tokens: ~{int(estimated_tokens)}")
        print(f"   Max allowed: {self.max_tokens_per_call}")
        print(f"   Total used so far: {self.total_tokens_used}")

        if estimated_tokens > self.max_tokens_per_call:
            print(f"   WARNING  WARNING: Input may exceed token limit!")
            self.calls_throttled += 1

            # В production тут можна truncate messages
            print(f"   Retry: Proceeding with warning\n")
        else:
            print()

        # Оновлюємо total usage
        self.total_tokens_used += int(estimated_tokens)

        return None

    def get_stats(self):
        """Повертає статистику використання"""
        return {
            "total_tokens_used": self.total_tokens_used,
            "calls_throttled": self.calls_throttled,
            "max_tokens_per_call": self.max_tokens_per_call
        }


# ============================================================================
# СТВОРЕННЯ АГЕНТА З MIDDLEWARE - ОФІЦІЙНИЙ API
# ============================================================================

def create_agent_with_middleware():
    """
    Створює агента з middleware hooks використовуючи ОФІЦІЙНИЙ LangChain 1.0 API

    ВАЖЛИВО: Використовується параметр middleware=[] в create_agent

    Middleware stack:
    1. LoggingMiddleware - logs all calls
    2. SecurityMiddleware - blocks risky operations
    3. TokenLimitMiddleware - controls costs
    """
    print("=" * 70)
    print("AGENT АГЕНТ З MIDDLEWARE - LangChain 1.0 (ОФІЦІЙНИЙ API)")
    print("=" * 70 + "\n")

    # Створюємо middleware instances
    logging_mw = LoggingMiddleware()
    security_mw = SecurityMiddleware()
    token_mw = TokenLimitMiddleware(max_tokens_per_call=4000)

    # Tools
    tools = [get_stock_price, send_notification, execute_trade]

    print("Available tools:")
    for tool in tools:
        risk = " (HIGH-RISK)" if tool.name in security_mw.high_risk_tools else ""
        print(f"  • {tool.name}{risk}")
    print()

    print("Middleware stack (ОФІЦІЙНИЙ LangChain 1.0 API):")
    print("  1. LoggingMiddleware (before_model + after_model)")
    print("  2. SecurityMiddleware (wrap_model_call)")
    print("  3. TokenLimitMiddleware (before_model)")
    print()

    # ОФІЦІЙНИЙ API: middleware передається в create_agent
    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""You are a financial assistant with access to stock data and notification tools.

You can:
- Get real-time stock prices
- Send notifications (requires approval)
- Execute trades (high-risk, requires approval)

Always explain your actions and ask for confirmation before risky operations.""",
        middleware=[logging_mw, security_mw, token_mw]  # ОФІЦІЙНИЙ API
    )

    return agent, logging_mw, security_mw, token_mw


# ============================================================================
# ТЕСТУВАННЯ АГЕНТА З MIDDLEWARE
# ============================================================================

def test_agent_with_middleware():
    """Тестує агента з різними middleware scenarios"""

    agent, logging_mw, security_mw, token_mw = create_agent_with_middleware()

    test_queries = [
        {
            "query": "What's the current price of AAPL stock?",
            "description": "Safe query - should pass all middleware",
            "expected": "get_stock_price tool call"
        },
        {
            "query": "Get TSLA price and send me notification about it",
            "description": "Contains HIGH-RISK tool - should be blocked by security",
            "expected": "SecurityMiddleware blocks send_notification"
        },
        {
            "query": "Execute trade: buy 100 shares of GOOGL",
            "description": "HIGH-RISK action - should be blocked",
            "expected": "SecurityMiddleware blocks execute_trade"
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
            # ОФІЦІЙНИЙ API: прямий invoke - middleware спрацюють автоматично
            result = agent.invoke({
                "messages": [{"role": "user", "content": test["query"]}]
            })

            print("\n" + "-" * 70)
            print("RESULT:")
            print("-" * 70)

            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    print(f"Output: {last_message.content}\n")
                else:
                    print(f"Output: {last_message}\n")
            else:
                print(f"Output: {result}\n")

        except Exception as e:
            print(f"\nERROR: {e}\n")
            import traceback
            traceback.print_exc()

        input("\nPAUSE  Press Enter to continue to next test...\n")

    # Виводимо статистику всіх middleware
    print("\n" + "=" * 70)
    print("MIDDLEWARE STATISTICS")
    print("=" * 70 + "\n")

    print("Logging Middleware:")
    logging_stats = logging_mw.get_stats()
    print(f"  Total calls: {logging_stats['total_calls']}")
    print()

    print("Security Middleware:")
    security_stats = security_mw.get_stats()
    print(f"  Calls blocked: {security_stats['calls_blocked']}")
    print(f"  High-risk tools: {', '.join(security_stats['high_risk_tools'])}")
    print()

    print("Token Limit Middleware:")
    token_stats = token_mw.get_stats()
    print(f"  Total tokens used: {token_stats['total_tokens_used']}")
    print(f"  Calls throttled: {token_stats['calls_throttled']}")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("TARGET LangChain 1.0 - Agent with OFFICIAL Middleware API")
    print("=" * 70)
    print()
    print("Features:")
    print("  OK ОФІЦІЙНИЙ AgentMiddleware API")
    print("  OK before_model + after_model hooks")
    print("  OK wrap_model_call для модифікації")
    print("  OK Real financial data (yfinance)")
    print("  OK Security middleware (blocks risky tools)")
    print("  OK Token limiting middleware")
    print("  OK LangSmith automatic tracing")
    print()
    print("=" * 70 + "\n")

    # Перевірка API ключів
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment!")
        print("Please set it in .env file")
        exit(1)

    try:
        test_agent_with_middleware()

        print("\n" + "=" * 70)
        print("OK ALL TESTS COMPLETED")
        print("=" * 70)
        print("\nTIP: Check LangSmith dashboard to see middleware traces:")
        print("   https://smith.langchain.com/\n")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
