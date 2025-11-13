"""
Basic CrewAI Example: Sequential Process

This example demonstrates the fundamental concepts of CrewAI:
- Creating Agents with specific roles and goals
- Defining Tasks with descriptions and expected outputs
- Organizing them into a Crew with sequential process
- Using crew.kickoff() to execute the workflow

CrewAI Version: 1.4.0+
Python: 3.10-3.13
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


def create_content_creation_crew():
    """
    Create a basic crew for content creation with sequential process.

    The crew consists of:
    - Researcher: Gathers information about the topic
    - Writer: Creates engaging content based on research
    - Editor: Reviews and polishes the final content

    Returns:
        Crew: Configured CrewAI crew instance
    """

    # Define Agents with specific roles
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments and comprehensive insights about {topic}",
        backstory=(
            "You are a seasoned research analyst with a keen eye for detail. "
            "Your expertise lies in diving deep into topics, finding credible sources, "
            "and synthesizing complex information into clear, actionable insights. "
            "You have 10 years of experience in tech industry research."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    writer = Agent(
        role="Tech Content Writer",
        goal="Craft compelling and informative content about {topic} for a technical audience",
        backstory=(
            "You are an experienced tech writer known for making complex topics "
            "accessible and engaging. Your writing style is clear, concise, and "
            "backed by solid research. You have published over 100 technical articles "
            "and have a deep understanding of developer audience needs."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    editor = Agent(
        role="Senior Content Editor",
        goal="Review and polish content to ensure highest quality and accuracy",
        backstory=(
            "You are a meticulous editor with a sharp eye for grammar, flow, and "
            "technical accuracy. You ensure every piece of content meets professional "
            "standards and effectively communicates the intended message. "
            "You have edited content for top tech publications."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Define Tasks
    research_task = Task(
        description=(
            "Conduct comprehensive research on {topic}. "
            "Your analysis should include:\n"
            "- Key features and capabilities\n"
            "- Latest developments and updates (as of 2025)\n"
            "- Use cases and practical applications\n"
            "- Comparison with alternatives\n"
            "- Best practices and recommendations\n\n"
            "Focus on credible sources and current information."
        ),
        expected_output=(
            "A detailed research report with 10-15 bullet points covering "
            "all aspects of {topic}, including key insights, statistics, "
            "and actionable recommendations."
        ),
        agent=researcher
    )

    writing_task = Task(
        description=(
            "Using the research provided, write an engaging technical article "
            "about {topic}. The article should:\n"
            "- Start with a compelling introduction\n"
            "- Clearly explain the topic for a technical audience\n"
            "- Include practical examples and use cases\n"
            "- Highlight key benefits and considerations\n"
            "- End with actionable takeaways\n\n"
            "Target length: 800-1000 words.\n"
            "Tone: Professional yet accessible."
        ),
        expected_output=(
            "A well-structured technical article of 800-1000 words with "
            "clear sections, practical examples, and actionable insights."
        ),
        agent=writer
    )

    editing_task = Task(
        description=(
            "Review and edit the article for:\n"
            "- Grammar and spelling errors\n"
            "- Technical accuracy\n"
            "- Flow and readability\n"
            "- Consistency in tone and style\n"
            "- Proper formatting and structure\n\n"
            "Ensure the final output is publication-ready."
        ),
        expected_output=(
            "A polished, publication-ready article with all errors corrected, "
            "improved flow, and professional formatting. Include a brief editor's "
            "note highlighting key improvements made."
        ),
        agent=editor
    )

    # Create Crew with sequential process
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,  # Tasks execute in order
        verbose=True
    )

    return crew


def main():
    """
    Main execution function demonstrating basic CrewAI usage.
    """
    print("=" * 80)
    print("BASIC CREWAI EXAMPLE: Content Creation Crew")
    print("=" * 80)
    print()
    print("This crew consists of 3 agents working sequentially:")
    print("1. Researcher - Gathers comprehensive information")
    print("2. Writer - Creates engaging content")
    print("3. Editor - Polishes final output")
    print()
    print("=" * 80)
    print()

    # Create the crew
    crew = create_content_creation_crew()

    # Define inputs
    inputs = {
        "topic": "Multi-Agent AI Systems with LangChain and CrewAI in 2025"
    }

    print(f"Starting crew execution with topic: {inputs['topic']}")
    print()
    print("-" * 80)
    print()

    # Execute the crew
    result = crew.kickoff(inputs=inputs)

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print()
    print(result)
    print()
    print("=" * 80)
    print("Crew execution completed successfully!")
    print("=" * 80)


def example_with_different_topic():
    """
    Alternative example showing how to use the same crew with different inputs.
    """
    crew = create_content_creation_crew()

    # Example 1: AI topic
    result1 = crew.kickoff(inputs={"topic": "LangGraph 1.0 State Management"})
    print("Result 1:", result1)

    # Example 2: Different topic
    result2 = crew.kickoff(inputs={"topic": "Vector Databases for RAG Systems"})
    print("Result 2:", result2)


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to run alternative examples
    # example_with_different_topic()


"""
KEY CONCEPTS DEMONSTRATED:

1. AGENTS:
   - Created with specific roles, goals, and backstories
   - Each agent has a clear responsibility
   - verbose=True enables detailed logging
   - allow_delegation=False keeps tasks focused
   - llm parameter specifies the model to use

2. TASKS:
   - description: What needs to be done
   - expected_output: What the result should look like
   - agent: Who is responsible
   - Input variables (e.g., {topic}) are parameterized

3. CREW:
   - Combines agents and tasks
   - Process.sequential ensures tasks run in order
   - kickoff(inputs={...}) executes the workflow
   - Each task receives output from previous task

4. SEQUENTIAL PROCESS:
   - Tasks execute one after another
   - Output of each task is available to the next
   - Simple and predictable execution flow
   - Good for linear workflows

USAGE TIPS:

1. Parameterization:
   - Use {variable_name} in task descriptions
   - Pass values via kickoff(inputs={...})
   - Same crew can handle different inputs

2. Agent Design:
   - Give each agent a specific, focused role
   - Clear goals improve output quality
   - Detailed backstory provides context
   - Avoid overlapping responsibilities

3. Task Dependencies:
   - In sequential process, order matters
   - Each task builds on previous results
   - Final task produces the end result

4. Error Handling:
   - Always check for API keys
   - Use verbose=True for debugging
   - Monitor token usage for long workflows

EXPECTED OUTPUT:

The crew will:
1. Research the topic comprehensively
2. Write an engaging article based on research
3. Edit and polish the final content

Each step is visible when verbose=True.
"""
