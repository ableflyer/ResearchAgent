from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Define the desired JSON output structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    full_explanation: str
    sources: list[str]
    tools_used: list[str]

# 1. Set up the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7
)

# --- STEP 1: RUN THE AGENT TO GET THE RESEARCH --- 

# A simple prompt for the agent to focus on research
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Use the provided tools to answer the user's query comprehensively."),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [search_tool, wiki_tool]

# Create the agent and the executor
agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = "The history of artificial intelligence"
print(f"--- Starting research for: {query} ---")

# Invoke the agent to get the raw text answer
agent_response = agent_executor.invoke({"query": query})
agent_output = agent_response.get("output", "")

print(f"--- Research complete. Raw output length: {len(agent_output)} ---")

# --- STEP 2: FORMAT THE RESEARCH INTO JSON --- 

print("--- Now, formatting the research into JSON ---")

# The parser for our desired ResearchResponse structure
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# A specific prompt to format the text into JSON
formatting_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a formatting expert. Your task is to take the provided text and format it into a valid JSON object "
     "that adheres to the following schema. Do NOT add any text before or after the JSON object.\n\n{format_instructions}"),
    ("human", "Here is the text to format:\n\n---\n{text_to_format}")
]).partial(format_instructions=parser.get_format_instructions())

# Create the formatting chain: prompt -> llm -> parser
formatting_chain = formatting_prompt | llm | parser

try:
    # Invoke the chain with the agent's output
    structured_response = formatting_chain.invoke({"text_to_format": agent_output})

    # Pretty-print the final JSON output
    print("\n--- Formatted Research Response ---")
    print(structured_response.model_dump_json(indent=2))

except Exception as e:
    print(f"\nError: Failed to parse the final JSON response.")
    print(f"Details: {e}")
    print("--- Raw Agent Output ---")
    print(agent_output)
