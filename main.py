from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool
from typing import List, Dict, Any
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Define output structures
class ResearchFindings(BaseModel):
    findings: str
    sources: List[str]
    confidence_score: float

class QualityAssessment(BaseModel):
    completeness_score: float
    accuracy_score: float
    issues_found: List[str]
    suggestions: List[str]

class FinalResearchResponse(BaseModel):
    topic: str
    summary: str
    full_explanation: str
    sources: List[str]
    tools_used: List[str]
    quality_metrics: QualityAssessment

class MultiAgentResearchSystem:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        self.tools = [search_tool, wiki_tool]
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize all specialized agents"""
        
        # 1. Research Agent - Gathers information
        research_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized research agent. Your role is to:
            1. Gather comprehensive information about the given topic
            2. Use multiple tools to find diverse sources
            3. Focus on factual accuracy and breadth of coverage
            4. Provide detailed findings with proper source attribution"""),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        self.research_agent = create_tool_calling_agent(self.llm, self.tools, research_prompt)
        self.research_executor = AgentExecutor(agent=self.research_agent, tools=self.tools, verbose=True)
        
        # 2. Analysis Agent - Processes and analyzes information
        self.analysis_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an analysis specialist. Your role is to:
            1. Analyze the research findings for patterns and insights
            2. Identify key themes and important details
            3. Structure information logically
            4. Highlight any gaps or inconsistencies in the research"""),
            ("human", "Analyze these research findings and provide insights:\n\n{research_data}")
        ])
        
        # 3. Quality Assurance Agent - Reviews and validates
        self.qa_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality assurance specialist. Your role is to:
            1. Review research for completeness and accuracy
            2. Check source reliability and diversity
            3. Identify any biases or missing perspectives
            4. Score the research quality on various metrics
            5. Provide specific suggestions for improvement
            
            Provide your assessment in the following JSON format: {format_instructions}"""),
            ("human", "Review this research:\n\nTopic: {topic}\nFindings: {findings}")
        ])
        
        # 4. Synthesis Agent - Creates final output
        self.synthesis_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a synthesis specialist. Your role is to:
            1. Combine research findings into a coherent narrative
            2. Create clear summaries and detailed explanations
            3. Ensure proper source attribution
            4. Structure the final response professionally
            
            Format your response as JSON: {format_instructions}"""),
            ("human", """Create a final research report from:
            
            Original Query: {query}
            Research Findings: {research_findings}
            Analysis: {analysis}
            Quality Assessment: {qa_assessment}""")
        ])
    
    def run_research_phase(self, query: str) -> Dict[str, Any]:
        """Phase 1: Gather research using the research agent"""
        print(f"🔍 Phase 1: Research Agent gathering information on '{query}'")
        
        research_response = self.research_executor.invoke({"query": query})
        research_output = research_response.get("output", "")
        
        # Extract tools used (simplified - you might want to track this more precisely)
        tools_used = ["search_tool", "wiki_tool"]  # You can enhance this tracking
        
        return {
            "findings": research_output,
            "tools_used": tools_used,
            "raw_response": research_response
        }
    
    def run_analysis_phase(self, research_data: str) -> str:
        """Phase 2: Analyze the research findings"""
        print("🧠 Phase 2: Analysis Agent processing findings")
        
        analysis_chain = self.analysis_agent_prompt | self.llm
        analysis_response = analysis_chain.invoke({"research_data": research_data})
        
        return analysis_response.content
    
    def run_qa_phase(self, topic: str, findings: str) -> QualityAssessment:
        """Phase 3: Quality assurance review"""
        print("✅ Phase 3: QA Agent reviewing research quality")
        
        parser = PydanticOutputParser(pydantic_object=QualityAssessment)
        qa_prompt = self.qa_agent_prompt.partial(format_instructions=parser.get_format_instructions())
        qa_chain = qa_prompt | self.llm | parser
        
        qa_response = qa_chain.invoke({
            "topic": topic,
            "findings": findings
        })
        
        return qa_response
    
    def run_synthesis_phase(self, query: str, research_findings: str, 
                          analysis: str, qa_assessment: QualityAssessment, 
                          tools_used: List[str]) -> FinalResearchResponse:
        """Phase 4: Synthesize final response"""
        print("📝 Phase 4: Synthesis Agent creating final report")
        
        parser = PydanticOutputParser(pydantic_object=FinalResearchResponse)
        synthesis_prompt = self.synthesis_agent_prompt.partial(format_instructions=parser.get_format_instructions())
        synthesis_chain = synthesis_prompt | self.llm | parser
        
        final_response = synthesis_chain.invoke({
            "query": query,
            "research_findings": research_findings,
            "analysis": analysis,
            "qa_assessment": qa_assessment.model_dump_json()
        })
        
        # Add quality metrics and tools used
        final_response.quality_metrics = qa_assessment
        final_response.tools_used = tools_used
        
        return final_response
    
    def research(self, query: str) -> FinalResearchResponse:
        """Main method to run the complete multi-agent research process"""
        print(f"🚀 Starting Multi-Agent Research System for: '{query}'")
        print("=" * 60)
        
        try:
            # Phase 1: Research
            research_result = self.run_research_phase(query)
            
            # Phase 2: Analysis
            analysis_result = self.run_analysis_phase(research_result["findings"])
            
            # Phase 3: Quality Assurance
            qa_result = self.run_qa_phase(query, research_result["findings"])
            
            # Phase 4: Synthesis
            final_result = self.run_synthesis_phase(
                query=query,
                research_findings=research_result["findings"],
                analysis=analysis_result,
                qa_assessment=qa_result,
                tools_used=research_result["tools_used"]
            )
            
            print("=" * 60)
            print("✨ Multi-Agent Research Complete!")
            return final_result
            
        except Exception as e:
            print(f"❌ Error in multi-agent system: {e}")
            raise

# Alternative: Concurrent Multi-Agent System
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ConcurrentMultiAgentSystem(MultiAgentResearchSystem):
    """Version that runs some agents concurrently for better performance"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    async def research_concurrent(self, query: str) -> FinalResearchResponse:
        """Run research with some concurrent processing"""
        print(f"🚀 Starting Concurrent Multi-Agent Research for: '{query}'")
        
        # Phase 1: Research (must be first)
        research_result = self.run_research_phase(query)
        
        # Phase 2 & 3: Run Analysis and QA concurrently
        loop = asyncio.get_event_loop()
        
        analysis_task = loop.run_in_executor(
            self.executor, 
            self.run_analysis_phase, 
            research_result["findings"]
        )
        
        qa_task = loop.run_in_executor(
            self.executor,
            self.run_qa_phase,
            query,
            research_result["findings"]
        )
        
        analysis_result, qa_result = await asyncio.gather(analysis_task, qa_task)
        
        # Phase 4: Synthesis (must be last)
        final_result = self.run_synthesis_phase(
            query=query,
            research_findings=research_result["findings"],
            analysis=analysis_result,
            qa_assessment=qa_result,
            tools_used=research_result["tools_used"]
        )
        
        return final_result

# Usage Example
def main():
    # Initialize the multi-agent system
    multi_agent_system = MultiAgentResearchSystem(api_key)
    
    # Run research
    query = input("What would you like to research about today?: ")
    result = multi_agent_system.research(query)
    
    # Display results
    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(result.model_dump_json(indent=2))

# For concurrent version:
async def main_concurrent():
    concurrent_system = ConcurrentMultiAgentSystem(api_key)
    query = "The history of artificial intelligence"
    result = await concurrent_system.research_concurrent(query)
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    # Run sequential version
    main()
    
    # Or run concurrent version
    # asyncio.run(main_concurrent())