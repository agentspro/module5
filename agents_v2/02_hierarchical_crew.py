"""
Hierarchical CrewAI Example: Manager-led Process

This example demonstrates CrewAI's hierarchical process:
- Automatic manager agent creation
- Task delegation based on agent capabilities
- Validation of outcomes by manager
- Complex project coordination

In hierarchical mode, a manager agent:
1. Plans task execution
2. Delegates to appropriate specialists
3. Reviews and validates outputs
4. Ensures quality standards

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


def create_software_development_crew():
    """
    Create a hierarchical crew for software development.

    The manager (auto-created) coordinates:
    - Requirements Analyst
    - Software Architect
    - Backend Developer
    - Frontend Developer
    - QA Engineer

    Returns:
        Crew: Hierarchical crew instance
    """

    # Define specialist agents
    requirements_analyst = Agent(
        role="Requirements Analyst",
        goal="Gather and analyze project requirements to create detailed specifications",
        backstory=(
            "You are an expert business analyst with 8 years of experience. "
            "You excel at understanding stakeholder needs and translating them "
            "into clear, actionable technical requirements. You ask the right "
            "questions and ensure nothing is overlooked."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    architect = Agent(
        role="Software Architect",
        goal="Design robust, scalable system architecture based on requirements",
        backstory=(
            "You are a senior software architect with expertise in modern "
            "cloud-native architectures, microservices, and distributed systems. "
            "You make informed decisions about technology stacks, design patterns, "
            "and system components. You have designed systems serving millions of users."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    backend_dev = Agent(
        role="Senior Backend Developer",
        goal="Implement backend services with best practices and clean code",
        backstory=(
            "You are a seasoned backend developer proficient in Python, Node.js, "
            "and database design. You write clean, maintainable code following "
            "SOLID principles. You have deep knowledge of RESTful APIs, GraphQL, "
            "and asynchronous programming."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    frontend_dev = Agent(
        role="Senior Frontend Developer",
        goal="Build intuitive, responsive user interfaces with modern frameworks",
        backstory=(
            "You are an expert frontend developer specializing in React, Vue, "
            "and modern JavaScript. You create pixel-perfect, accessible UIs "
            "with excellent UX. You understand performance optimization and "
            "state management patterns."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    qa_engineer = Agent(
        role="QA Engineer",
        goal="Ensure software quality through comprehensive testing strategies",
        backstory=(
            "You are a detail-oriented QA engineer with expertise in test "
            "automation, performance testing, and security testing. You create "
            "comprehensive test plans and catch issues before they reach production. "
            "You advocate for quality at every stage."
        ),
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Define tasks (manager will delegate appropriately)
    requirements_task = Task(
        description=(
            "Analyze the project request for {project_name} and create detailed "
            "requirements documentation. Include:\n"
            "- Functional requirements\n"
            "- Non-functional requirements (performance, security, scalability)\n"
            "- User stories\n"
            "- Acceptance criteria\n"
            "- Potential risks and constraints\n\n"
            "Ask clarifying questions if needed."
        ),
        expected_output=(
            "Comprehensive requirements document with all functional and "
            "non-functional requirements, user stories, and acceptance criteria."
        ),
        agent=requirements_analyst
    )

    architecture_task = Task(
        description=(
            "Based on requirements, design the system architecture for {project_name}. "
            "Your design should include:\n"
            "- System components and their interactions\n"
            "- Technology stack recommendations\n"
            "- Data models and database schema\n"
            "- API contracts\n"
            "- Security considerations\n"
            "- Scalability approach\n"
            "- Deployment architecture\n\n"
            "Justify your technology choices."
        ),
        expected_output=(
            "Detailed architecture document with component diagrams, technology "
            "stack, data models, API specifications, and security design."
        ),
        agent=architect
    )

    backend_task = Task(
        description=(
            "Implement the backend for {project_name} based on the architecture. "
            "Provide:\n"
            "- API endpoint specifications (RESTful or GraphQL)\n"
            "- Database schema implementation\n"
            "- Business logic implementation approach\n"
            "- Authentication/authorization strategy\n"
            "- Error handling and logging\n"
            "- Sample code for critical components\n\n"
            "Follow best practices and design patterns."
        ),
        expected_output=(
            "Backend implementation plan with API specifications, database schema, "
            "sample code, and documentation for key components."
        ),
        agent=backend_dev
    )

    frontend_task = Task(
        description=(
            "Design and implement the frontend for {project_name}. Include:\n"
            "- Component architecture\n"
            "- State management approach\n"
            "- UI/UX design principles\n"
            "- Responsive design strategy\n"
            "- API integration patterns\n"
            "- Sample code for main views\n\n"
            "Ensure accessibility and performance."
        ),
        expected_output=(
            "Frontend implementation plan with component structure, state management, "
            "sample code, and UX considerations."
        ),
        agent=frontend_dev
    )

    testing_task = Task(
        description=(
            "Create a comprehensive testing strategy for {project_name}. Cover:\n"
            "- Unit testing approach for backend and frontend\n"
            "- Integration testing strategy\n"
            "- End-to-end testing scenarios\n"
            "- Performance testing plan\n"
            "- Security testing checklist\n"
            "- Test automation recommendations\n"
            "- Quality metrics and acceptance criteria\n\n"
            "Focus on critical paths and edge cases."
        ),
        expected_output=(
            "Complete testing strategy with test plans, automation approach, "
            "and quality metrics for all testing levels."
        ),
        agent=qa_engineer
    )

    # Create hierarchical crew
    # Manager agent is auto-created by CrewAI
    crew = Crew(
        agents=[
            requirements_analyst,
            architect,
            backend_dev,
            frontend_dev,
            qa_engineer
        ],
        tasks=[
            requirements_task,
            architecture_task,
            backend_task,
            frontend_task,
            testing_task
        ],
        process=Process.hierarchical,  # Enables manager coordination
        verbose=True,
        manager_llm="gpt-4o-mini"  # LLM for the manager agent
    )

    return crew


def main():
    """
    Main execution demonstrating hierarchical crew.
    """
    print("=" * 80)
    print("HIERARCHICAL CREWAI EXAMPLE: Software Development Team")
    print("=" * 80)
    print()
    print("This crew uses hierarchical process with automatic manager coordination:")
    print()
    print("MANAGER (Auto-created)")
    print("  |")
    print("  +-- Requirements Analyst")
    print("  +-- Software Architect")
    print("  +-- Backend Developer")
    print("  +-- Frontend Developer")
    print("  +-- QA Engineer")
    print()
    print("The manager will:")
    print("  1. Plan the execution strategy")
    print("  2. Delegate tasks to appropriate specialists")
    print("  3. Review outputs for quality")
    print("  4. Coordinate handoffs between team members")
    print()
    print("=" * 80)
    print()

    # Create the crew
    crew = create_software_development_crew()

    # Define project
    inputs = {
        "project_name": "AI-powered Task Management System with Multi-Agent Support"
    }

    print(f"Starting development for: {inputs['project_name']}")
    print()
    print("-" * 80)
    print()

    # Execute with manager coordination
    result = crew.kickoff(inputs=inputs)

    print()
    print("=" * 80)
    print("FINAL PROJECT DELIVERABLES")
    print("=" * 80)
    print()
    print(result)
    print()
    print("=" * 80)
    print("Hierarchical crew execution completed!")
    print("=" * 80)


def example_smaller_project():
    """
    Example with a smaller 3-agent hierarchical crew.
    """

    # Create minimal hierarchical crew
    designer = Agent(
        role="UI/UX Designer",
        goal="Create user-friendly interface designs",
        backstory="Expert designer with 5 years experience",
        verbose=True,
        llm="gpt-4o-mini"
    )

    developer = Agent(
        role="Full-Stack Developer",
        goal="Implement the complete application",
        backstory="Versatile developer proficient in frontend and backend",
        verbose=True,
        llm="gpt-4o-mini"
    )

    design_task = Task(
        description="Design the UI for {app_name}",
        expected_output="UI mockups and design specifications",
        agent=designer
    )

    development_task = Task(
        description="Implement {app_name} based on the design",
        expected_output="Working application code",
        agent=developer
    )

    crew = Crew(
        agents=[designer, developer],
        tasks=[design_task, development_task],
        process=Process.hierarchical,
        verbose=True,
        manager_llm="gpt-4o-mini"
    )

    result = crew.kickoff(inputs={"app_name": "Weather Dashboard"})
    print(result)


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment for smaller example
    # example_smaller_project()


"""
KEY CONCEPTS DEMONSTRATED:

1. HIERARCHICAL PROCESS:
   - Manager agent auto-created by CrewAI
   - Manager plans, delegates, and validates
   - Specialists focus on their expertise
   - Dynamic task allocation

2. MANAGER RESPONSIBILITIES:
   - Analyzes task requirements
   - Assigns tasks to best-suited agents
   - Reviews outputs for quality
   - Coordinates dependencies
   - Makes strategic decisions

3. AGENT SPECIALIZATION:
   - Each agent has narrow, focused expertise
   - allow_delegation=False for specialists
   - Manager handles coordination
   - Clear separation of concerns

4. TASK DELEGATION:
   - Tasks assigned to specific agents
   - Manager can override if needed
   - Dependencies handled automatically
   - Validation at each step

HIERARCHICAL VS SEQUENTIAL:

Sequential:
  ✓ Simple, predictable flow
  ✓ All tasks execute in order
  ✗ No dynamic decision-making
  ✗ Limited flexibility

Hierarchical:
  ✓ Intelligent task routing
  ✓ Quality validation
  ✓ Adaptive execution
  ✗ More complex
  ✗ Higher token usage (manager LLM calls)

WHEN TO USE HIERARCHICAL:

1. Complex projects with multiple specialties
2. Need for quality validation
3. Dynamic task dependencies
4. Uncertain execution path
5. Large teams (5+ agents)

MANAGER CONFIGURATION:

- manager_llm: Specifies model for manager (can be different from agents)
- Manager is NOT in agents list (auto-created)
- Manager has elevated decision-making capabilities
- Manager can reassign tasks if needed

BEST PRACTICES:

1. Clear Agent Roles:
   - Each agent should have distinct expertise
   - Avoid overlapping responsibilities
   - Detailed backstories improve delegation

2. Task Descriptions:
   - Be specific about deliverables
   - Include acceptance criteria
   - Mention quality standards

3. Manager Model:
   - Use capable model for manager (GPT-4 recommended)
   - Manager makes critical decisions
   - Impacts overall quality

4. Monitoring:
   - Use verbose=True for debugging
   - Watch manager's delegation decisions
   - Review validation steps

LIMITATIONS:

1. Manager agents cannot have tools directly
   - Workaround: Create assistant agent with tools
   - Manager delegates to assistant

2. Higher cost:
   - Additional LLM calls for manager
   - More tokens for coordination

3. Complexity:
   - Harder to debug
   - Less predictable execution path

EXPECTED OUTPUT:

Manager will:
1. Analyze the project requirements
2. Delegate to Requirements Analyst
3. Review requirements, delegate to Architect
4. Review architecture, delegate to developers
5. Review implementations, delegate to QA
6. Validate final deliverables
"""
