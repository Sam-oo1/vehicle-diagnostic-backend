import os
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# Import from our modules
from utility_functions import MessageHistory, DocumentStore, PDFProcessor
from agents import (
    IntentRecognitionAgent, 
    QueryRefinementAgent,
    InformationGatheringAgent,
    DiagnosticAgent,
    ReasoningAgent,
    ModificationAgent,
    ResponseFormulationAgent
)

# Load environment variables
load_dotenv()

# Feature flags - you can modify these as needed
ENABLE_WEB_SEARCH = False  # Flag to enable/disable web search
ENABLE_PDF_PROCESSING = False  # Flag to enable/disable PDF processing


class DiagnosticFlow:
    """Orchestrates the flow between different agents"""
    
    def __init__(self):
        # Initialize LLM
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0,
            max_retries=2
        )
        
        # Initialize document store
        self.doc_store = DocumentStore()
        
        # Initialize message history
        self.message_history = MessageHistory("user_session")
        
        # Initialize all agents
        self.intent_agent = IntentRecognitionAgent("Intent Recognition", self.llm)
        self.query_agent = QueryRefinementAgent("Query Refinement", self.llm)
        self.info_agent = InformationGatheringAgent("Information Gathering", self.llm, self.doc_store)
        self.diagnostic_agent = DiagnosticAgent("Diagnostic Analysis", self.llm)
        self.reasoning_agent = ReasoningAgent("Reasoning", self.llm)
        self.modification_agent = ModificationAgent("Modification", self.llm)
        self.response_agent = ResponseFormulationAgent("Response Formulation", self.llm, self.message_history)
        
        # Configure web search if enabled
        if ENABLE_WEB_SEARCH:
            search_tool = TavilySearchResults(
                max_results=5,
                api_key=os.getenv("TAVILY_API_KEY")
            )
            self.info_agent.set_search_tool(search_tool)
        
        # Initialize PDF processor only if needed
        if ENABLE_PDF_PROCESSING:
            self.pdf_processor = PDFProcessor()
    
    def process_query(self, user_query: str, pdf_path: Optional[str] = None) -> str:
        """Process user query through the multi-agent pipeline"""
        print("\n--- Processing your query... ---")
        
        # Add PDF context if provided and PDF processing is enabled
        if ENABLE_PDF_PROCESSING and pdf_path:
            try:
                raw_pdf_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
                pdf_text = self.pdf_processor.clean_pdf_text(raw_pdf_text)
                print(f"PDF loaded successfully!")
                user_query = user_query + "\n\nContext from PDF: " + pdf_text
            except Exception as e:
                print(f"Error loading PDF: {e}")
        elif pdf_path and not ENABLE_PDF_PROCESSING:
            print("⚠ PDF processing is disabled. Ignoring PDF input.")
        
        # Initial state with user query
        state = {"user_query": user_query}
        
        # Intent Recognition
        intent_state = self.intent_agent.process(state)
        state.update(intent_state)
        
        # Special handling for modification requests
        if state["intent"] == "modify_previous":
            last_response = self.message_history.get_last_assistant_message()
            if last_response:
                state["previous_response"] = last_response.content
                
                # Get the modified version
                modification_state = self.modification_agent.process(state)
                state.update(modification_state)
                
                # Generate final response
                response_state = self.response_agent.process(state)
                state.update(response_state)
                
                # Update the last message in history instead of adding new ones
                self.message_history.replace_last_assistant_message(state["final_response"])
                
                return state["final_response"]
            else:
                return "I don't have any previous responses to modify."
        
        # Normal flow for new questions
        # Query Refinement
        refined_state = self.query_agent.process(state)
        state.update(refined_state)
        
        # Information Gathering
        info_state = self.info_agent.process(state)
        state.update(info_state)
        
        # Diagnostic Analysis
        diagnostic_state = self.diagnostic_agent.process(state)
        state.update(diagnostic_state)
        
        # Reasoning
        reasoning_state = self.reasoning_agent.process(state)
        state.update(reasoning_state)
        
        # Response Formulation
        response_state = self.response_agent.process(state)
        state.update(response_state)
        
        # Add to history
        self.message_history.add_message(HumanMessage(content=user_query))
        self.message_history.add_message(AIMessage(content=state["final_response"]))
        
        return state.get("final_response", "Sorry, there was an error processing your query.")
    
    def clear_history(self):
        """Clear conversation history"""
        self.message_history.clear()
        print("Conversation history cleared.")


def run_diagnostic_system():
    """Main function to run the vehicle diagnostic system"""
    print("\n===== Vehicle Diagnostic Assistant =====")
    print("Type 'exit' to quit, 'clear' to clear conversation history")
    print(f"Web Search: {'Enabled' if ENABLE_WEB_SEARCH else 'Disabled'}")
    print(f"PDF Processing: {'Enabled' if ENABLE_PDF_PROCESSING else 'Disabled'}")
    
    orchestrator = DiagnosticFlow()
    
    while True:
        # Get user input
        user_query = input("\nEnter your vehicle diagnostic query: ")
        
        # Check for exit command
        if user_query.lower() == 'exit':
            print("Goodbye! Your conversation history has been saved.")
            break
            
        # Check for clear history command
        if user_query.lower() == 'clear':
            orchestrator.clear_history()
            continue
        
        # Handle PDF attachment only if PDF processing is enabled
        pdf_path = None
        if ENABLE_PDF_PROCESSING:
            use_pdf = input("Do you want to attach a PDF? (y/n): ").strip().lower()
            if use_pdf == 'y':
                pdf_path = input("Enter the path to your PDF file: ")
        
        # Process the query
        final_response = orchestrator.process_query(user_query, pdf_path)
        
        print("\n Final Answer:")
        print(final_response)


if __name__ == "__main__":
    run_diagnostic_system()
