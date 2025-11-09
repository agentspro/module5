# 🎯 Final Workshop Project

## Customer Support Agent System

**Time:** 30-40 minutes
**Difficulty:** Intermediate
**Modules used:** All (1-4)

---

## 📋 Project Description

Побудуйте повноцінну систему customer support агента з наступними можливостями:

### Must Have (Required):

1. **RAG для документації** (Module 1)
   - Векторна база знань з FAQs
   - Пошук релевантної інформації
   - Structured відповіді

2. **Tools для дій** (Module 2)
   - Create ticket
   - Check order status
   - Reset password
   - Search knowledge base

3. **Persistent memory** (Module 3)
   - Запам'ятовує історію розмови
   - Thread-based conversations
   - Checkpointing

4. **Escalation logic** (Module 3)
   - Human-in-the-loop для критичних дій
   - Умовні переходи

### Nice to Have (Bonus points):

5. **Multi-agent** (Module 4)
   - Technical support agent
   - Billing agent
   - Supervisor для routing

6. **Monitoring**
   - LangSmith integration
   - Cost tracking
   - Performance metrics

---

## 🏗️ Architecture

```
User Query
    ↓
Supervisor Agent
    ↓
   ├─→ Search Knowledge Base (RAG)
   ├─→ Technical Agent (if tech issue)
   ├─→ Billing Agent (if billing issue)
   └─→ Escalate to Human (if critical)
    ↓
Response + Actions Taken
```

---

## 📦 Starter Code

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# ============================================================================
# 1. DEFINE KNOWLEDGE BASE (RAG)
# ============================================================================

# TODO: Create vector store with FAQs
# Hint: Use FAISS + OpenAIEmbeddings

faqs = [
    "How to reset password? Click 'Forgot Password' on login page...",
    "Shipping takes 3-5 business days for standard delivery...",
    "Refund policy: 30 days money-back guarantee...",
]

# TODO: Implement search_knowledge_base tool


# ============================================================================
# 2. DEFINE TOOLS
# ============================================================================

@tool
def create_ticket(description: str, priority: str = "normal") -> str:
    """Creates a support ticket for complex issues."""
    ticket_id = f"TICKET-{hash(description) % 10000}"
    return f"✅ Created ticket {ticket_id} with priority {priority}"


@tool
def check_order_status(order_id: str) -> str:
    """Checks the status of an order."""
    # Mock implementation
    statuses = {
        "12345": "Shipped - Expected delivery: Dec 25",
        "67890": "Processing - Will ship tomorrow",
    }
    return statuses.get(order_id, f"Order {order_id} not found")


@tool
def reset_password(email: str) -> str:
    """Sends password reset email. REQUIRES APPROVAL."""
    return f"✅ Password reset email sent to {email}"


# TODO: Add search_knowledge_base tool here


# ============================================================================
# 3. DEFINE STATE
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_email: str
    requires_human: bool
    action_taken: str


# ============================================================================
# 4. BUILD AGENT GRAPH
# ============================================================================

def build_support_agent():
    """Builds the customer support agent system"""

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [create_ticket, check_order_status, reset_password]

    # TODO: Add agent logic
    # Hints:
    # 1. Create agent with tools
    # 2. Add conditional edges for escalation
    # 3. Add checkpointer for memory
    # 4. Handle human-in-the-loop for critical actions

    workflow = StateGraph(AgentState)

    # Add your nodes here...

    # Compile with checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    return app


# ============================================================================
# 5. MAIN - TEST YOUR AGENT
# ============================================================================

if __name__ == "__main__":
    agent = build_support_agent()

    config = {"configurable": {"thread_id": "customer_123"}}

    test_queries = [
        "Hi, I need help with my order #12345",
        "I forgot my password, can you reset it for user@example.com?",
        "What's your refund policy?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"👤 Customer: {query}")
        print(f"{'='*60}")

        result = agent.invoke({
            "messages": [("user", query)],
            "user_email": "customer@example.com",
            "requires_human": False,
            "action_taken": ""
        }, config)

        print(f"🤖 Agent: {result['messages'][-1].content}")
        if result.get("action_taken"):
            print(f"✅ Actions: {result['action_taken']}")
```

---

## ✅ Evaluation Criteria

### Basic (70 points):
- [ ] Agent responds to queries (10pts)
- [ ] RAG search works (15pts)
- [ ] At least 2 tools implemented (15pts)
- [ ] Checkpointing preserves context (15pts)
- [ ] Error handling (10pts)
- [ ] Code is clean and commented (5pts)

### Advanced (30 points):
- [ ] Human-in-the-loop for sensitive actions (10pts)
- [ ] Multi-agent setup with routing (10pts)
- [ ] LangSmith integration (5pts)
- [ ] Custom metrics/logging (5pts)

### Bonus:
- [ ] Deployment-ready (Dockerfile, API) (+10pts)
- [ ] Tests written (+5pts)
- [ ] Creative features (+5pts)

---

## 🎓 Learning Objectives

By completing this project, you will:

1. ✅ Understand how to combine RAG with agents
2. ✅ Build production-like agent systems
3. ✅ Implement proper error handling
4. ✅ Use state management with LangGraph
5. ✅ Handle real-world scenarios (escalation, memory)

---

## 💡 Hints

### For RAG:
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

docs = [Document(page_content=faq) for faq in faqs]
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

### For Human-in-the-Loop:
```python
def should_ask_human(state):
    last_message = state["messages"][-1]
    # Check if action requires approval
    if "reset_password" in str(last_message):
        return "approval"
    return "execute"
```

### For Multi-Agent:
```python
def supervisor(state):
    query = state["messages"][-1].content
    if "order" in query.lower():
        return "billing_agent"
    elif "technical" in query.lower():
        return "tech_agent"
    return "general_agent"
```

---

## 📊 Example Output

```
============================================================
👤 Customer: What's your refund policy?
============================================================
🤖 Agent: According to our policy, we offer a 30-day money-back
guarantee. Simply contact support within 30 days of purchase
and we'll process your refund.
✅ Actions: Searched knowledge base

============================================================
👤 Customer: Can you reset my password for user@example.com?
============================================================
⚠️  REQUIRES HUMAN APPROVAL: Reset password for user@example.com
✅ Approved
🤖 Agent: Password reset email has been sent to user@example.com.
Please check your inbox and follow the instructions.
✅ Actions: Password reset completed
```

---

## 🚀 Next Steps After Workshop

1. **Deploy it:**
   - Add FastAPI wrapper
   - Create Dockerfile
   - Deploy to LangGraph Cloud/AWS

2. **Enhance it:**
   - Add more tools
   - Integrate real APIs
   - Add authentication

3. **Monitor it:**
   - Set up LangSmith
   - Track costs
   - Monitor performance

4. **Scale it:**
   - Add PostgreSQL persistence
   - Implement rate limiting
   - Add caching

---

## 📚 Resources

- RAG: `module1_lcel/03_rag.py`
- Agents: `module2_agents/01_basic_agent.py`
- LangGraph: `module3_langgraph/*`
- Multi-Agent: `module4_multi_agent/*`

---

**Good luck! 🎉**

Questions? Ask the instructor or check the solutions folder after attempting yourself!
