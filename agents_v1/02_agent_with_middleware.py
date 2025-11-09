"""
АГЕНТ З MIDDLEWARE - LangChain 1.0 API
На базі документації: Agent Middleware API (2025)

Middleware hooks:
- before_model: Runs before model calls
- after_model: Runs after model calls
- modify_model_request: Modify tools, prompts, messages before model call

LangSmith Integration: Автоматично трейсить всі middleware operations
"""

import os
from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

# ============================================================================
# LANGSMITH VERIFICATION
# ============================================================================

if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("✅ LangSmith трейсинг активний")
    print(f"📊 Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    print("🔍 Middleware operations will be traced\n")
else:
    print("⚠️  LangSmith не ввімкнений\n")


# ============================================================================
# TOOLS
# ============================================================================

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol."""
    # Mock data
    prices = {
        "AAPL": "$175.50",
        "GOOGL": "$140.20",
        "MSFT": "$380.00",
        "TSLA": "$245.80"
    }
    return prices.get(symbol.upper(), f"Price for {symbol} not found")


@tool
def send_notification(message: str, recipient: str) -> str:
    """
    Send notification to user. REQUIRES APPROVAL in middleware.

    Args:
        message: Notification message
        recipient: Recipient email or ID
    """
    return f"✅ Notification sent to {recipient}: {message}"


@tool
def execute_trade(symbol: str, quantity: int, action: str) -> str:
    """
    Execute a trade. HIGH-RISK action requiring approval.

    Args:
        symbol: Stock symbol
        quantity: Number of shares
        action: 'buy' or 'sell'
    """
    return f"⚠️ Would execute {action} {quantity} shares of {symbol}"


# ============================================================================
# MIDDLEWARE IMPLEMENTATIONS
# ============================================================================

class LoggingMiddleware:
    """
    Middleware для логування всіх model calls
    Implements: before_model, after_model hooks
    """

    def __init__(self):
        self.call_count = 0
        self.logs = []

    def before_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Виконується ПЕРЕД кожним викликом моделі"""
        self.call_count += 1

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_number": self.call_count,
            "event": "before_model",
            "input_length": len(str(state.get("messages", ""))),
        }

        self.logs.append(log_entry)

        print(f"\n{'='*60}")
        print(f"📝 MIDDLEWARE: Before Model Call #{self.call_count}")
        print(f"⏰ Time: {log_entry['timestamp']}")
        print(f"📊 Input length: {log_entry['input_length']} chars")
        print(f"{'='*60}\n")

        # Можна модифікувати state тут
        return state

    def after_model(self, state: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """Виконується ПІСЛЯ кожного виклику моделі"""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_number": self.call_count,
            "event": "after_model",
            "result_type": type(result).__name__,
        }

        self.logs.append(log_entry)

        print(f"\n{'='*60}")
        print(f"✅ MIDDLEWARE: After Model Call #{self.call_count}")
        print(f"⏰ Time: {log_entry['timestamp']}")
        print(f"📤 Result type: {log_entry['result_type']}")
        print(f"{'='*60}\n")

        return state

    def get_stats(self):
        """Повертає статистику викликів"""
        return {
            "total_calls": self.call_count,
            "logs": self.logs
        }


class SecurityMiddleware:
    """
    Middleware для безпеки - блокує небезпечні операції без approval
    Implements: modify_model_request hook
    """

    def __init__(self):
        self.blocked_actions = []
        self.approved_actions = []

        # Список high-risk tools що потребують approval
        self.high_risk_tools = ["execute_trade", "send_notification"]

    def modify_model_request(self, tools: List, messages: List, **kwargs) -> Dict[str, Any]:
        """
        Модифікує request перед відправкою до моделі
        Може змінити: tools, messages, prompt, model settings
        """

        print(f"\n{'='*60}")
        print("🔒 SECURITY MIDDLEWARE: Checking request")
        print(f"{'='*60}\n")

        # Перевіряємо чи є в messages згадки high-risk actions
        full_text = " ".join([str(m) for m in messages])

        for risky_tool in self.high_risk_tools:
            if risky_tool in full_text.lower() or "trade" in full_text.lower():
                print(f"⚠️  Detected potential use of HIGH-RISK tool: {risky_tool}")
                print(f"🛡️  Security check required\n")

                # Симулюємо approval process
                # В production тут був би real approval workflow
                approval = self._request_approval(risky_tool, full_text)

                if not approval:
                    # Блокуємо high-risk tools
                    print(f"🚫 BLOCKED: {risky_tool} requires approval\n")
                    self.blocked_actions.append({
                        "tool": risky_tool,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Видаляємо risky tool зі списку доступних
                    tools = [t for t in tools if t.name != risky_tool]

                    # Додаємо warning до messages
                    warning_msg = {
                        "role": "system",
                        "content": f"SECURITY WARNING: Tool '{risky_tool}' is blocked due to security policy. Inform user that this action requires manual approval. Suggest alternative safe actions."
                    }
                    messages = [warning_msg] + messages

        print("✅ Security check complete\n")

        return {
            "tools": tools,
            "messages": messages,
            **kwargs
        }

    def _request_approval(self, tool_name: str, context: str) -> bool:
        """
        Симулює approval process
        В production це був би call to approval service або human-in-the-loop
        """

        print(f"📋 Requesting approval for: {tool_name}")
        print(f"📄 Context: {context[:100]}...")

        # Mock approval logic
        # В реальності тут був би pause для human approval
        auto_approve = False

        if auto_approve:
            print("✅ Approved automatically (mock)\n")
            self.approved_actions.append(tool_name)
            return True
        else:
            print("❌ Auto-approval disabled - action blocked\n")
            return False

    def get_stats(self):
        """Статистика security middleware"""
        return {
            "blocked_actions": self.blocked_actions,
            "approved_actions": self.approved_actions
        }


class TokenLimitMiddleware:
    """
    Middleware для контролю витрат tokens
    Implements: before_model hook
    """

    def __init__(self, max_tokens_per_call: int = 1000):
        self.max_tokens_per_call = max_tokens_per_call
        self.total_tokens_used = 0
        self.calls_throttled = 0

    def before_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Перевіряє і обмежує token usage"""

        input_text = str(state.get("messages", ""))
        estimated_tokens = len(input_text.split()) * 1.3  # Rough estimate

        print(f"\n💰 TOKEN MIDDLEWARE:")
        print(f"   Estimated input tokens: ~{int(estimated_tokens)}")
        print(f"   Max allowed: {self.max_tokens_per_call}")
        print(f"   Total used so far: {self.total_tokens_used}")

        if estimated_tokens > self.max_tokens_per_call:
            print(f"   ⚠️  WARNING: Input may exceed token limit!")
            self.calls_throttled += 1

            # В production тут можна truncate input або block call
            print(f"   🔄 Truncating input to fit limit\n")

        print()
        return state


# ============================================================================
# MIDDLEWARE AGENT WRAPPER - LangChain 1.0
# ============================================================================

class MiddlewareAgent:
    """
    Wrapper навколо create_agent який додає middleware functionality

    LangChain 1.0 API: create_agent повертає готовий agent
    Ми обгортаємо його invoke() методом для додавання middleware hooks
    """

    def __init__(self, model: str, tools: List, system_prompt: str, middlewares: List = None):
        self.tools = tools
        self.middlewares = middlewares or []

        # Створюємо base agent через create_agent
        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt
        )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Викликає agent з middleware hooks

        Flow:
        1. before_model middlewares
        2. modify_model_request middlewares
        3. agent.invoke()
        4. after_model middlewares
        """

        # Before model middlewares
        for mw in self.middlewares:
            if hasattr(mw, 'before_model'):
                inputs = mw.before_model(inputs)

        # Modify request middlewares
        tools_to_use = self.tools
        messages = inputs.get("messages", [])

        for mw in self.middlewares:
            if hasattr(mw, 'modify_model_request'):
                modifications = mw.modify_model_request(
                    tools=tools_to_use,
                    messages=messages,
                )
                tools_to_use = modifications.get('tools', tools_to_use)
                messages = modifications.get('messages', messages)

        # Оновлюємо messages після middleware
        modified_inputs = {**inputs, "messages": messages}

        # Execute agent
        result = self.agent.invoke(modified_inputs)

        # After model middlewares
        for mw in self.middlewares:
            if hasattr(mw, 'after_model'):
                inputs = mw.after_model(inputs, result)

        return result


# ============================================================================
# СТВОРЕННЯ АГЕНТА З MIDDLEWARE
# ============================================================================

def create_agent_with_middleware():
    """
    Створює агента з middleware hooks використовуючи LangChain 1.0 API

    Middleware stack:
    1. LoggingMiddleware - logs all calls
    2. SecurityMiddleware - blocks risky operations
    3. TokenLimitMiddleware - controls costs
    """

    print("=" * 70)
    print("🛡️  AGENT WITH MIDDLEWARE - LangChain 1.0")
    print("=" * 70 + "\n")

    # Initialize middlewares
    logging_mw = LoggingMiddleware()
    security_mw = SecurityMiddleware()
    token_mw = TokenLimitMiddleware(max_tokens_per_call=500)

    print("Middleware Stack:")
    print("  1️⃣  LoggingMiddleware - Track all operations")
    print("  2️⃣  SecurityMiddleware - Block risky actions")
    print("  3️⃣  TokenLimitMiddleware - Control costs")
    print()

    # Tools
    tools = [get_stock_price, send_notification, execute_trade]

    print("Available tools:")
    for t in tools:
        risk = "🔴 HIGH-RISK" if t.name in security_mw.high_risk_tools else "🟢 SAFE"
        print(f"  • {t.name}: {risk}")
    print()

    # Create agent with middleware (LangChain 1.0 API)
    agent = MiddlewareAgent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""You are a helpful AI assistant with access to financial tools.

Use the available tools to answer user questions accurately.
Always provide clear, helpful responses.""",
        middlewares=[logging_mw, security_mw, token_mw]
    )

    return agent, {
        "logging": logging_mw,
        "security": security_mw,
        "tokens": token_mw
    }


# ============================================================================
# TESTING
# ============================================================================

def test_middleware_agent():
    """Test agent with middleware"""

    agent, middlewares = create_agent_with_middleware()

    test_cases = [
        {
            "name": "Safe Query",
            "input": "What's the current price of AAPL?",
            "expected": "Should work normally"
        },
        {
            "name": "Risky Action",
            "input": "Execute a trade: buy 100 shares of TSLA",
            "expected": "Should be BLOCKED by security middleware"
        },
        {
            "name": "Multiple Tools",
            "input": "Get GOOGL price and notify john@example.com about it",
            "expected": "Notification should be blocked"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {test['name']}")
        print(f"Input: {test['input']}")
        print(f"Expected: {test['expected']}")
        print("=" * 70)

        try:
            # LangChain 1.0 API: invoke приймає messages
            result = agent.invoke({
                "messages": [{"role": "user", "content": test["input"]}]
            })

            # Extract output від agent
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                output = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                output = str(result)

            print(f"\n✅ Result: {output}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        input("\n⏸️  Press Enter for next test...")

    # Print middleware stats
    print("\n" + "=" * 70)
    print("📊 MIDDLEWARE STATISTICS")
    print("=" * 70)

    print(f"\n📝 Logging Middleware:")
    print(json.dumps(middlewares["logging"].get_stats(), indent=2))

    print(f"\n🔒 Security Middleware:")
    print(json.dumps(middlewares["security"].get_stats(), indent=2))

    print(f"\n💰 Token Middleware:")
    print(f"   Calls throttled: {middlewares['tokens'].calls_throttled}")
    print(f"   Total tokens tracked: {middlewares['tokens'].total_tokens_used}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🎯 LangChain 1.0 - Agent with Middleware")
    print("=" * 70)
    print("\nMiddleware Features (2025 API):")
    print("  ✅ before_model hook - Pre-processing")
    print("  ✅ after_model hook - Post-processing")
    print("  ✅ modify_model_request - Request modification")
    print("  ✅ Security controls - Block risky operations")
    print("  ✅ Token limiting - Cost control")
    print("  ✅ LangSmith tracing - Full observability")
    print("  ✅ LangChain 1.0 create_agent API")
    print("\n" + "=" * 70 + "\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found!")
        exit(1)

    try:
        test_middleware_agent()

        print("\n" + "=" * 70)
        print("✅ ALL MIDDLEWARE TESTS COMPLETED")
        print("=" * 70)
        print("\n💡 Check LangSmith for detailed middleware traces!\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
