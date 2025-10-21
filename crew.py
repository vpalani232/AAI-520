from news_agent import news_agent
from quant_agent import quantitative_analysis_agent
from crewai import Process, Agent, LLM, Crew, Task
from dotenv import load_dotenv

load_dotenv()

NIM_MODEL_NAME = "meta/llama-3.1-70b-instruct"
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

llm = LLM(model="nvidia_nim/meta/llama-3.1-405b-instruct", temperature=0.7)


smart_summarizer_agent = Agent(
    role="Financial Insight Synthesizer",
    goal="""Integrate data from quantitative analysis and market news into concise, structured, and decision-ready insights.""",
    backstory="""A professional financial writer and analyst specializing in summarizing technical and qualitative data
    into actionable investment insights. Ensures clarity and factual integrity.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

supervisor = Agent(
    role="Investment Research Assistant Supervisor",
    goal="""You are an Investment Research Assistant, responsible for overseeing and synthesizing financial research from specialized agents. Your role is to coordinate subagents to produce structured investment insights.

Your capabilities include:
1. Managing collaboration between subagents to retrieve and analyze financial data.
2. Synthesizing stock trends, financial reports, and market news into a structured analysis.
3. Delivering well-organized, fact-based investment insights with clear distinctions between data sources.

Available subagents:
- **news_agent**: Retrieves and summarizes the latest financial news.  
  - **Always instruct news_agent to check the knowledge base first before using external web searches**.
- **quantitative_analysis_agent**: Provides real-time and historical stock prices.  
  - For portfolio optimization, retrieve stock data via `stock_data_lookup` before calling `portfolio_optimization_action_group`.
- **smart_summarizer_agent**: Synthesizes financial data and market trends into a structured investment insight.

Core behaviors:
- Only invoke a subagent when necessary. Do not invoke agent for information not requested by user.
- Ensure responses are **well-structured, clearly formatted, and relevant to investor decision-making**.
- Differentiate between financial news, technical stock analysis, and synthesized insights.
""",
    backstory="A seasoned investment research expert responsible for orchestrating subagents to conduct a comprehensive stock analysis. This agent synthesizes market news, stock data, and smart_summarizer insights into a structured investment report.",
    verbose=True,
    allow_delegation=True,
    tools=[],  # Add your specific tools here
    llm=llm,
)
# --- Crew Setup ---
crew = Crew(
    agents=[
        news_agent,
        quantitative_analysis_agent,
        smart_summarizer_agent,
    ],
    name="Investment Research Crew",
    manager_agent=supervisor,
    process=Process.hierarchical,
    description="A multi-agent system that performs end-to-end investment research and delivers structured financial insights.",
    verbose=True,
)


def run_research_crew(user_query: str):
    """
    Takes a user query, creates a CrewAI Task for the Supervisor,
    and kicks off the hierarchical research crew.
    """
    # 1. Create the Task
    research_task = Task(
        description=user_query,
        expected_output="A structured investment report clearly differentiating between Quantitative Data, News/Sentiment, and the Final Investment Insight.",
        agent=supervisor,
        context=None,
    )

    print(f"\n\n--- Starting Investment Research Crew for Query: '{user_query}' ---")
    print("The Supervisor Agent is delegating tasks...")

    # ********************************
    # *** FIX: Assign Task to Crew ***
    # ********************************
    # Assign the dynamically created task list directly to the crew object
    crew.tasks = [research_task]

    # 2. Kick off the crew *without* passing any arguments.
    # The crew will now use the tasks stored in crew.tasks
    crew_result = crew.kickoff()

    # 3. Return the final result
    return crew_result


# Now, running the test should work:
user_request = "Conduct a full analysis on Microsoft (MSFT). I need the latest news summary, current stock price with RSI, and a clear investment recommendation for the short term."
final_report = run_research_crew(user_request)
print(final_report)
