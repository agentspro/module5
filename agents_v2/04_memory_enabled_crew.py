"""
Memory-Enabled CrewAI: Conversational Crew with Persistent Memory

This example demonstrates:
- CrewAI's memory capabilities
- Persistent context across conversations
- Learning from previous interactions
- Conversational crew pattern

Memory types in CrewAI:
- Short-term memory: Current conversation context
- Long-term memory: Persistent across sessions
- Entity memory: Information about specific entities
- Contextual memory: Task and agent-specific context

CrewAI Version: 1.4.0+
Python: 3.10-3.13
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from typing import Dict, List

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


def create_memory_enabled_crew():
    """
    Create a crew with memory capabilities for personalized assistance.

    This crew learns from interactions and maintains context across sessions.

    The crew includes:
    - Personal Assistant: Main interface with user preferences memory
    - Knowledge Curator: Organizes and recalls information
    - Context Analyzer: Understands conversation context

    Returns:
        Crew: Memory-enabled crew instance
    """

    # Define memory-aware agents
    assistant = Agent(
        role="Personal AI Assistant",
        goal=(
            "Provide personalized assistance by remembering user preferences, "
            "past conversations, and context from previous interactions"
        ),
        backstory=(
            "You are an intelligent personal assistant that learns from every "
            "interaction. You remember user preferences, their goals, past "
            "conversations, and important details they've shared. You use this "
            "knowledge to provide increasingly personalized and relevant assistance. "
            "You have been assisting users for 3 years and pride yourself on "
            "understanding their needs deeply."
        ),
        verbose=True,
        allow_delegation=True,
        llm="gpt-4o-mini"
    )

    knowledge_curator = Agent(
        role="Knowledge Curator",
        goal=(
            "Organize, store, and retrieve relevant information from past "
            "interactions and user-provided data"
        ),
        backstory=(
            "You are an expert at organizing information and making it easily "
            "retrievable. You maintain a well-structured knowledge base of user "
            "information, preferences, and past interactions. You excel at finding "
            "connections between different pieces of information and surfacing "
            "relevant context when needed."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    context_analyzer = Agent(
        role="Context Analyzer",
        goal=(
            "Analyze conversation context and identify relevant information "
            "from memory to improve responses"
        ),
        backstory=(
            "You are a specialist in understanding context and intent. You analyze "
            "current requests in light of past conversations, user preferences, "
            "and historical data. You identify when past context is relevant and "
            "ensure responses are contextually appropriate."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Define tasks that leverage memory
    context_task = Task(
        description=(
            "Analyze the current request: '{user_request}'\n\n"
            "Consider:\n"
            "1. What does the user want now?\n"
            "2. What relevant information do we know from past interactions?\n"
            "3. What preferences should we consider?\n"
            "4. What context from previous conversations is relevant?\n\n"
            "Provide a context analysis that will guide the response."
        ),
        expected_output=(
            "Detailed context analysis including:\n"
            "- Current request intent\n"
            "- Relevant past context\n"
            "- User preferences to consider\n"
            "- Recommendations for response approach"
        ),
        agent=context_analyzer
    )

    knowledge_task = Task(
        description=(
            "Based on the context analysis, retrieve and organize relevant "
            "information for: '{user_request}'\n\n"
            "Tasks:\n"
            "1. Search memory for related past conversations\n"
            "2. Identify user preferences that apply\n"
            "3. Find relevant facts or data from previous interactions\n"
            "4. Organize information in a useful structure\n\n"
            "Focus on quality and relevance."
        ),
        expected_output=(
            "Organized knowledge package with:\n"
            "- Relevant past conversations\n"
            "- Applicable user preferences\n"
            "- Key facts and data points\n"
            "- Structured for easy use in response"
        ),
        agent=knowledge_curator
    )

    response_task = Task(
        description=(
            "Create a personalized response to: '{user_request}'\n\n"
            "Use the context analysis and retrieved knowledge to provide:\n"
            "1. Direct answer to the current request\n"
            "2. Personalized based on user preferences\n"
            "3. Referenced relevant past context when appropriate\n"
            "4. Proactive suggestions based on user history\n\n"
            "Tone: {tone}\n"
            "Response style: Conversational and personalized"
        ),
        expected_output=(
            "Personalized, context-aware response that:\n"
            "- Directly addresses the request\n"
            "- Shows understanding of user preferences\n"
            "- References relevant context naturally\n"
            "- Provides value beyond just answering the question"
        ),
        agent=assistant
    )

    # Create crew with memory enabled
    crew = Crew(
        agents=[context_analyzer, knowledge_curator, assistant],
        tasks=[context_task, knowledge_task, response_task],
        process=Process.sequential,
        verbose=True,
        memory=True,  # Enable memory
        embedding_model={
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        }
    )

    return crew


def simulate_conversation():
    """
    Simulate a multi-turn conversation demonstrating memory.
    """
    print("=" * 80)
    print("MEMORY-ENABLED CREW: Personalized AI Assistant")
    print("=" * 80)
    print()
    print("This crew maintains context across multiple interactions:")
    print()
    print("FEATURES:")
    print("  - Remembers user preferences")
    print("  - Recalls past conversations")
    print("  - Learns from interactions")
    print("  - Provides personalized responses")
    print()
    print("MEMORY TYPES:")
    print("  - Short-term: Current conversation")
    print("  - Long-term: Across sessions")
    print("  - Entity: About specific topics/people")
    print("  - Contextual: Task-specific context")
    print()
    print("=" * 80)
    print()

    crew = create_memory_enabled_crew()

    # Conversation turn 1: User shares preferences
    print("TURN 1: User shares preferences")
    print("-" * 80)
    turn1 = {
        "user_request": (
            "I'm working on a multi-agent AI project using LangChain. "
            "I prefer Python and focus on production-ready code. "
            "I value clean architecture and good documentation."
        ),
        "tone": "professional and helpful"
    }
    result1 = crew.kickoff(inputs=turn1)
    print("\nResponse:", result1)
    print("\n" + "=" * 80 + "\n")

    # Conversation turn 2: Related question
    print("TURN 2: Related question (should remember context)")
    print("-" * 80)
    turn2 = {
        "user_request": (
            "What's the best way to structure my agent code?"
        ),
        "tone": "professional and helpful"
    }
    result2 = crew.kickoff(inputs=turn2)
    print("\nResponse:", result2)
    print("\n" + "=" * 80 + "\n")

    # Conversation turn 3: Follow-up
    print("TURN 3: Follow-up question")
    print("-" * 80)
    turn3 = {
        "user_request": (
            "Can you recommend testing strategies for this?"
        ),
        "tone": "professional and helpful"
    }
    result3 = crew.kickoff(inputs=turn3)
    print("\nResponse:", result3)
    print("\n" + "=" * 80 + "\n")

    print("=" * 80)
    print("CONVERSATION COMPLETE")
    print("=" * 80)
    print()
    print("Notice how the crew:")
    print("  - Remembered the LangChain + Python context")
    print("  - Applied preferences (production-ready, clean architecture)")
    print("  - Built on previous answers")
    print("  - Maintained conversation flow")
    print()
    print("=" * 80)


def example_learning_preferences():
    """
    Example showing how crew learns and applies user preferences.
    """

    # Create a preference-learning crew
    learning_agent = Agent(
        role="Preference Learning Assistant",
        goal="Learn and apply user preferences across interactions",
        backstory=(
            "You excel at understanding user preferences from their behavior "
            "and explicit statements. You apply learned preferences to improve "
            "future interactions."
        ),
        verbose=True,
        llm="gpt-4o-mini"
    )

    learn_task = Task(
        description=(
            "Process the user's statement: '{statement}'\n"
            "Extract and remember any preferences, goals, or important context."
        ),
        expected_output="Summary of learned preferences",
        agent=learning_agent
    )

    crew = Crew(
        agents=[learning_agent],
        tasks=[learn_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
        embedding_model={
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"}
        }
    )

    # Interaction 1: Learn preferences
    crew.kickoff(inputs={
        "statement": "I prefer concise code examples over long explanations"
    })

    # Interaction 2: Should apply learned preference
    crew.kickoff(inputs={
        "statement": "Show me how to create an agent"
    })


def main():
    """
    Main execution demonstrating memory-enabled crew.
    """
    # Run the conversation simulation
    simulate_conversation()

    # Uncomment to run preference learning example
    # example_learning_preferences()


if __name__ == "__main__":
    main()


"""
KEY CONCEPTS DEMONSTRATED:

1. MEMORY CONFIGURATION:
   - memory=True enables crew-level memory
   - embedder config specifies embedding model
   - Memory persists across kickoff() calls
   - Automatic context retrieval

2. MEMORY TYPES:

   A. Short-term Memory:
      - Current conversation context
      - Recent task outputs
      - Temporary working memory
      - Cleared after session

   B. Long-term Memory:
      - Persists across sessions
      - User preferences
      - Historical facts
      - Learned patterns

   C. Entity Memory:
      - Information about specific entities
      - User profiles
      - Project details
      - Relationship tracking

   D. Contextual Memory:
      - Task-specific context
      - Agent-specific knowledge
      - Conversation threads

3. MEMORY BENEFITS:

   - Personalization: Tailored responses
   - Context Awareness: Understanding history
   - Learning: Improving over time
   - Efficiency: No need to repeat information
   - Continuity: Seamless multi-turn conversations

4. MEMORY USAGE PATTERNS:

   Pattern 1 - Conversational Assistant:
   - Remember user preferences
   - Maintain conversation context
   - Learn from interactions
   - Provide personalized help

   Pattern 2 - Knowledge Base:
   - Store facts and data
   - Retrieve relevant information
   - Build knowledge over time
   - Connect related concepts

   Pattern 3 - Project Context:
   - Remember project details
   - Track progress
   - Maintain consistency
   - Recall decisions

EMBEDDING MODEL CONFIGURATION:

CrewAI uses embeddings to store and retrieve memories:

```python
embedding_model={
    "provider": "openai",  # or "cohere", "huggingface"
    "config": {
        "model": "text-embedding-3-small"
    }
}
```

Providers:
- openai: OpenAI embeddings (recommended)
- cohere: Cohere embeddings
- huggingface: HuggingFace embeddings

MEMORY BEST PRACTICES:

1. Enable Early:
   - Add memory from the start
   - Agents learn over time
   - More data = better context

2. Clear Backstories:
   - Mention memory in agent backstory
   - Explain learning capability
   - Set expectations

3. Structured Learning:
   - Explicitly store key info
   - Use dedicated learning tasks
   - Organize knowledge

4. Privacy Considerations:
   - Be mindful of sensitive data
   - Implement data retention policies
   - Allow memory clearing

MEMORY OPERATIONS:

While CrewAI handles memory automatically, you can:

1. Enable/Disable:
   ```python
   crew = Crew(..., memory=True)
   ```

2. Configure Embedding Model:
   ```python
   crew = Crew(
       ...,
       embedding_model={"provider": "openai", "config": {...}}
   )
   ```

3. Context is Automatic:
   - No manual retrieval needed
   - Crew handles context injection
   - Relevant memories auto-included

CONVERSATIONAL CREW V1:

CrewAI 0.98.0+ includes Conversational Crew:

```python
from crewai import ConversationalCrew

crew = ConversationalCrew(
    agents=[...],
    tasks=[...],
    memory=True
)

# Interactive conversation
crew.chat("What can you help me with?")
```

Features:
- Natural conversation flow
- Automatic context maintenance
- Multi-turn interactions
- User intent understanding

INTEGRATION WITH LANGMEM:

For advanced memory capabilities:

```python
from langmem import Client

# Create LangMem client
memory_client = Client()

# Use with CrewAI agents
# (Implementation depends on specific needs)
```

LangMem provides:
- Advanced memory search
- Memory organization
- Cross-conversation context
- Memory analytics

MEMORY LIMITATIONS:

1. Token Limits:
   - Memory context uses tokens
   - Very long histories may exceed limits
   - Implement summarization for long sessions

2. Relevance:
   - Not all past context is relevant
   - Embedding quality matters
   - May retrieve irrelevant info

3. Cost:
   - Embedding API calls
   - Additional tokens in prompts
   - Storage considerations

4. Privacy:
   - Persistent storage of user data
   - Compliance requirements
   - Data retention policies

DEBUGGING MEMORY:

1. Use verbose=True:
   - See what context is retrieved
   - Monitor memory usage
   - Debug relevance issues

2. Test Memory:
   - Run multi-turn conversations
   - Verify context retention
   - Check preference application

3. Monitor Embeddings:
   - Check embedding costs
   - Verify quality
   - Optimize if needed

EXPECTED BEHAVIOR:

In the conversation example:
1. Turn 1: Crew learns about LangChain project and preferences
2. Turn 2: Crew remembers context and applies preferences
3. Turn 3: Crew builds on previous responses

The memory system:
- Automatically embeds conversation history
- Retrieves relevant context
- Injects into agent prompts
- Maintains continuity

PRODUCTION CONSIDERATIONS:

1. Memory Persistence:
   - Where is memory stored?
   - How long to retain?
   - Backup strategy?

2. User Management:
   - Separate memory per user
   - User ID tracking
   - Privacy controls

3. Scaling:
   - Memory grows over time
   - Implement archiving
   - Monitor storage costs

4. Quality:
   - Regular memory validation
   - Remove outdated info
   - Update changed preferences
"""
