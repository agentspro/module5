"""
Research CrewAI with Tools Integration

This example demonstrates:
- Integration with LangChain tools
- Custom tool creation for CrewAI
- Tools assignment to specific agents
- Real-world research workflow with tools

Tools enable agents to:
- Search the web
- Read files
- Perform calculations
- Access external APIs

CrewAI Version: 1.4.0+
Python: 3.10-3.13
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    FileReadTool,
    DirectoryReadTool,
)
from crewai.tools import tool
from typing import Dict, Any
import json

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# Custom tools using @tool decorator (CrewAI native API)
@tool("analyze_data")
def analyze_data(data_json: str) -> str:
    """Analyze JSON data and return statistical insights. Input should be a valid JSON string."""
    try:
        data = json.loads(data_json)

        if isinstance(data, list):
            count = len(data)
            analysis = f"Dataset contains {count} items.\n"

            # If items are numbers
            if all(isinstance(x, (int, float)) for x in data):
                avg = sum(data) / len(data)
                min_val = min(data)
                max_val = max(data)
                analysis += f"Average: {avg:.2f}\n"
                analysis += f"Min: {min_val}, Max: {max_val}\n"

            # If items are dictionaries
            elif all(isinstance(x, dict) for x in data):
                keys = set()
                for item in data:
                    keys.update(item.keys())
                analysis += f"Keys found: {', '.join(keys)}\n"

            return analysis

        elif isinstance(data, dict):
            return f"Dictionary with {len(data)} keys: {', '.join(data.keys())}"

        return "Data analyzed successfully"

    except json.JSONDecodeError:
        return "Error: Invalid JSON data"
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


@tool("calculate_metrics")
def calculate_metrics(expression: str) -> str:
    """Safely evaluate mathematical expressions. Input should be a Python expression like '2 + 2' or 'sum([1,2,3])'."""
    try:
        # Safe eval for basic math
        allowed_names = {"abs": abs, "min": min, "max": max, "sum": sum}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# File tools
file_read_tool = FileReadTool()
directory_read_tool = DirectoryReadTool()


def create_research_crew():
    """
    Create a research crew with tools for comprehensive analysis.

    The crew includes:
    - Data Researcher: Searches and gathers information
    - Data Analyst: Analyzes data with statistical tools
    - Report Writer: Compiles findings into reports

    Returns:
        Crew: Research crew with tools
    """

    # Web search tool using DuckDuckGo (no API key needed, CrewAI-native)
    from duckduckgo_search import DDGS

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web for current information using DuckDuckGo. Input should be a search query string."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No results found for '{query}'"
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get('title', 'No title')
                body = r.get('body', '')[:200]
                formatted.append(f"{i}. {title}\n   {body}")
            return "\n\n".join(formatted)
        except Exception as e:
            return f"Error searching web: {str(e)}"

    search_tool = web_search

    # Define agents with specific tools
    researcher = Agent(
        role="Senior Data Researcher",
        goal="Gather comprehensive information about {research_topic} using available tools",
        backstory=(
            "You are an expert researcher with access to powerful tools. "
            "You know how to find relevant information, read documentation, "
            "and organize findings systematically. You have 12 years of "
            "experience in data research and information retrieval."
        ),
        tools=[
            file_read_tool,
            directory_read_tool,
            search_tool,
        ],
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    analyst = Agent(
        role="Senior Data Analyst",
        goal="Analyze data and extract meaningful insights for {research_topic}",
        backstory=(
            "You are a skilled data analyst proficient in statistical analysis. "
            "You use analytical tools to process data, identify patterns, "
            "and derive actionable insights. You have expertise in both "
            "quantitative and qualitative analysis methods."
        ),
        tools=[
            analyze_data,
            calculate_metrics
        ],
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    writer = Agent(
        role="Research Report Writer",
        goal="Create comprehensive, well-structured research reports",
        backstory=(
            "You are an experienced technical writer who excels at "
            "synthesizing complex research findings into clear, actionable "
            "reports. You ensure every report is well-organized, accurate, "
            "and provides practical recommendations."
        ),
        tools=[],  # Writer doesn't need tools
        verbose=True,
        allow_delegation=False,
        llm="gpt-4o-mini"
    )

    # Define research tasks
    research_task = Task(
        description=(
            "Research {research_topic} thoroughly. Your research should:\n"
            "1. Use file reading tools to examine local documentation\n"
            "2. Gather information from available sources\n"
            "3. Identify key data points and metrics\n"
            "4. Note any gaps in information\n\n"
            "Focus on: {focus_areas}\n\n"
            "Compile all findings with source references."
        ),
        expected_output=(
            "Comprehensive research findings document with:\n"
            "- Key information about the topic\n"
            "- Relevant data and statistics\n"
            "- Source references\n"
            "- Identified gaps or areas needing more investigation"
        ),
        agent=researcher
    )

    analysis_task = Task(
        description=(
            "Analyze the research findings for {research_topic}:\n"
            "1. Use the DataAnalyzer tool for any JSON data found\n"
            "2. Use the Calculator for numerical analysis\n"
            "3. Identify trends and patterns\n"
            "4. Calculate relevant metrics\n"
            "5. Provide statistical insights\n\n"
            "Focus on: {focus_areas}\n\n"
            "Be thorough and data-driven."
        ),
        expected_output=(
            "Detailed analysis report with:\n"
            "- Statistical analysis results\n"
            "- Identified patterns and trends\n"
            "- Key metrics and calculations\n"
            "- Data-driven insights and observations"
        ),
        agent=analyst
    )

    report_task = Task(
        description=(
            "Create a comprehensive research report on {research_topic}:\n"
            "1. Executive Summary\n"
            "2. Research Methodology\n"
            "3. Key Findings (from research phase)\n"
            "4. Data Analysis (from analysis phase)\n"
            "5. Insights and Implications\n"
            "6. Recommendations\n"
            "7. Conclusion\n\n"
            "The report should be clear, well-structured, and actionable.\n"
            "Target audience: {target_audience}"
        ),
        expected_output=(
            "Publication-ready research report with all sections, "
            "professional formatting, clear insights, and actionable "
            "recommendations for the target audience."
        ),
        agent=writer
    )

    # Create crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=True
    )

    return crew


def main():
    """
    Main execution demonstrating research crew with tools.
    """
    print("=" * 80)
    print("RESEARCH CREW WITH TOOLS")
    print("=" * 80)
    print()
    print("This crew demonstrates tool integration:")
    print()
    print("RESEARCHER")
    print("  Tools: FileReadTool, DirectoryReadTool")
    print("  Purpose: Gather information from local sources")
    print()
    print("ANALYST")
    print("  Tools: DataAnalyzer, Calculator")
    print("  Purpose: Analyze data and calculate metrics")
    print()
    print("WRITER")
    print("  Tools: None (uses agent outputs)")
    print("  Purpose: Compile comprehensive report")
    print()
    print("=" * 80)
    print()

    # NOTE: This crew uses REAL tools:
    # - Web search via DuckDuckGo for current information
    # - File reading/directory tools for local data
    # - Data analysis and calculation tools
    print("Using REAL APIs (no sample/mock data):")
    print("  - DuckDuckGo Search for web research")
    print("  - File I/O tools for document analysis")
    print("  - Data analysis tools for metrics")
    print()

    # Create the crew
    crew = create_research_crew()

    # Define research parameters
    inputs = {
        "research_topic": "Multi-Agent AI Frameworks Comparison (LangChain, LangGraph, CrewAI)",
        "focus_areas": "architecture, ease of use, tool integration, use cases",
        "target_audience": "Technical leaders and AI engineers"
    }

    print(f"Starting research on: {inputs['research_topic']}")
    print(f"Focus areas: {inputs['focus_areas']}")
    print(f"Target audience: {inputs['target_audience']}")
    print()
    print("-" * 80)
    print()

    # Execute the crew
    result = crew.kickoff(inputs=inputs)

    print()
    print("=" * 80)
    print("FINAL RESEARCH REPORT")
    print("=" * 80)
    print()
    print(result)
    print()
    print("=" * 80)
    print("Research crew execution completed!")
    print("=" * 80)


def example_with_custom_tools():
    """
    Example showing custom tool creation with @tool decorator.
    """

    @tool("custom_formatter")
    def custom_formatter(text: str) -> str:
        """Format text to uppercase."""
        return text.upper()

    agent = Agent(
        role="Text Processor",
        goal="Process text with formatting tools",
        backstory="Expert in text processing",
        tools=[custom_formatter],
        verbose=True,
        llm="gpt-4o-mini"
    )

    task = Task(
        description="Format the text: {text}",
        expected_output="Formatted text",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff(inputs={"text": "hello world"})
    print(result)


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment for custom tools example
    # example_with_custom_tools()


"""
KEY CONCEPTS DEMONSTRATED:

1. TOOL INTEGRATION:
   - LangChain tools work seamlessly with CrewAI
   - CrewAI provides its own tool library (crewai_tools)
   - Custom tools created with @tool decorator (langchain_core.tools.tool)
   - Tools assigned per agent

2. TOOL TYPES:

   A. CrewAI Tools (crewai_tools):
      - FileReadTool: Read file contents
      - DirectoryReadTool: List directory contents
      - WebsiteSearchTool: Search websites
      - ScrapeWebsiteTool: Scrape web pages
      - And many more...

   B. LangChain Tools:
      - SerpAPIWrapper: Web search
      - PythonREPLTool: Execute Python
      - Tool: Custom tool wrapper

   C. Custom Tools:
      - Define your own functions
      - Wrap with Tool class
      - Specify name, description, function

3. TOOL ASSIGNMENT:
   - tools parameter when creating Agent
   - Each agent gets specific tools for their role
   - Tools are NOT shared between agents by default
   - Manager agents cannot have tools (use assistant pattern)

4. TOOL USAGE:
   - Agents automatically decide when to use tools
   - Tool descriptions guide usage
   - Results are incorporated into agent reasoning
   - Tools can be chained

CREATING CUSTOM TOOLS:

Method 1 - Using @tool decorator (RECOMMENDED):
```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    "Tool description"
    return result
```

Method 2 - Using StructuredTool (for complex schemas):
```python
from langchain_core.tools import StructuredTool

tool = StructuredTool.from_function(
    func=my_function,
    name="ToolName",
    description="What this tool does"
)
```

Method 3 - CrewAI custom tool:
```python
from crewai_tools import BaseTool

class MyTool(BaseTool):
    name: str = "ToolName"
    description: str = "Description"

    def _run(self, argument: str) -> str:
        return result
```

TOOL BEST PRACTICES:

1. Clear Descriptions:
   - Describe what the tool does
   - Specify expected input format
   - Mention any limitations

2. Error Handling:
   - Tools should handle errors gracefully
   - Return meaningful error messages
   - Don't raise uncaught exceptions

3. Appropriate Assignment:
   - Give tools to agents who need them
   - Don't overload agents with too many tools
   - Consider tool complexity

4. Testing:
   - Test tools independently first
   - Verify tool outputs are useful
   - Check error cases

AVAILABLE CREWAI TOOLS:

- FileReadTool: Read files
- DirectoryReadTool: List directories
- FileWriteTool: Write to files
- WebsiteSearchTool: Search websites
- ScrapeWebsiteTool: Scrape content
- SeleniumScrapingTool: Browser automation
- PGSearchTool: PostgreSQL search
- CodeDocsSearchTool: Search code docs
- And 20+ more...

TOOL INTEGRATION PATTERNS:

1. Research Pattern:
   - Search tool for gathering
   - Analysis tool for processing
   - No tools for writing

2. Development Pattern:
   - Code search tools
   - File read/write tools
   - Testing tools

3. Data Pattern:
   - Database query tools
   - Analysis tools
   - Visualization tools

LIMITATIONS:

1. Tool Complexity:
   - Very complex tools may confuse agents
   - Keep tools focused and simple

2. Tool Access:
   - Some tools require API keys
   - Rate limits may apply
   - Cost considerations

3. Reliability:
   - Tools can fail
   - Network issues
   - External API dependencies

EXPECTED OUTPUT:

The crew will:
1. Researcher uses FileReadTool to examine local files
2. Analyst uses DataAnalyzer and Calculator on findings
3. Writer compiles everything into final report

Each tool usage is logged when verbose=True.
"""
