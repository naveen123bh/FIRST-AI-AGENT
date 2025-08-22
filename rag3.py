import os
import wikipedia
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# -----------------------------
# API Keys
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables.")

# -----------------------------
# Tools
# -----------------------------
def safe_wikipedia_search(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return None

tools = [
    TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY),
    {
        "name": "WikipediaSearch",
        "description": "Search Wikipedia for factual info.",
        "func": safe_wikipedia_search
    }
]

# -----------------------------
# Initialize LLM (Ollama Mistral)
# -----------------------------
llm = ChatOpenAI(
    model="mistral",
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# Create agent executor
# -----------------------------
agent_executor = create_react_agent(llm_with_tools, tools)

# -----------------------------
# Memory and tool log
# -----------------------------
memory = []
tool_log = []  # Dynamic log of tools used

# -----------------------------
# Autonomous planning loop
# -----------------------------
print("🤖 Autonomous AI Agent ready! Type 'exit' to quit.")
while True:
    goal = input("\nEnter high-level goal: ")
    if goal.lower() in ["exit", "quit"]:
        print("Exiting agent. Goodbye! 👋")
        break

    memory.append({"role": "user", "content": goal})

    # Step 1: Planner generates sub-tasks
    planning_prompt = f"""
You are an AI planner. Break the following goal into 3-5 actionable tasks:
Goal: {goal}
Output as a numbered list.
"""
    task_plan_response = llm.invoke([{"role": "user", "content": planning_prompt}])
    task_plan = task_plan_response.content  # <-- fixed access

    print("\n[Planner created tasks]:")
    print(task_plan)

    tasks = [t.strip() for t in task_plan.split("\n") if t.strip()]

    for task in tasks:
        print(f"\n[Executing task]: {task}")

        # Initialize tool decision
        tool_used = "Unknown"

        # Try agent tools
        try:
            response = agent_executor.invoke({"messages": [("user", task)]})
            agent_messages = response.get('messages', [])
            # Attempt to log tool used if available
            tool_used = getattr(response, 'tool_name', 'Agent tool used')
        except Exception:
            agent_messages = []
            tool_used = "Error / Tool failed"

        # Fallback if no response
        if not agent_messages:
            fallback_response = llm.invoke([{"role": "user", "content": task}])
            agent_messages = [fallback_response] if hasattr(fallback_response, 'content') else [{"content": str(fallback_response)}]
            tool_used = "Fallback LLM reasoning"

        # Real-time log
        print(f"[Tool used]: {tool_used}")
        tool_log.append({"task": task, "tool_used": tool_used})

        # Store and print results
        for msg in agent_messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            memory.append({"role": "agent", "content": content})
            print("AI:", content)

    print("\n✅ Goal execution complete.\n")

# -----------------------------
# Dynamic session summary
# -----------------------------
print("\n📊 Tool Usage Summary:")
for entry in tool_log:
    print(f"Task: {entry['task']} -> Tool Used: {entry['tool_used']}")
