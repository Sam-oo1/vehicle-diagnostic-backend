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

# Default feature flags - you can modify these as needed
DEFAULT_ENABLE_WEB_SEARCH = False  # Flag to enable/disable web search
DEFAULT_ENABLE_PDF_PROCESSING = False  # Flag to enable/disable PDF processing


class DiagnosticFlow:
    """Orchestrates the flow between different agents"""
    
    def __init__(self, default_enable_web_search=DEFAULT_ENABLE_WEB_SEARCH, 
                 default_enable_pdf_processing=DEFAULT_ENABLE_PDF_PROCESSING):
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
        
        # Set default flags
        self.default_enable_web_search = default_enable_web_search
        self.default_enable_pdf_processing = default_enable_pdf_processing
        
        # Create search tool (will be used only when needed)
        self.search_tool = TavilySearchResults(
            max_results=5,
            api_key=os.getenv("TAVILY_API_KEY")
        ) if os.getenv("TAVILY_API_KEY") else None
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()
    
    def process_query(self, 
                     user_query: str, 
                     enable_web_search: Optional[bool] = None,
                     enable_pdf_processing: Optional[bool] = None,
                     pdf_content: Optional[bytes] = None) -> str:
        """Process user query through the multi-agent pipeline
        
        Args:
            user_query (str): User's diagnostic query
            enable_web_search (bool, optional): Override default web search setting for this query
            enable_pdf_processing (bool, optional): Override default PDF processing for this query
            pdf_content (bytes, optional): PDF file content as bytes if applicable
            
        Returns:
            str: Final response to the user
        """
        # Use provided flags or fall back to defaults
        use_web_search = enable_web_search if enable_web_search is not None else self.default_enable_web_search
        use_pdf_processing = enable_pdf_processing if enable_pdf_processing is not None else self.default_enable_pdf_processing
        
        # Configure search tool for this query
        if use_web_search:
            if self.search_tool:
                self.info_agent.set_search_tool(self.search_tool)
            else:
                print("Web search was enabled but Tavily API key is missing.")
        else:
            self.info_agent.set_search_tool(None)
            
        # Add PDF context if provided and PDF processing is enabled
        if use_pdf_processing and pdf_content:
            try:
                pdf_text = self.pdf_processor.extract_text_from_bytes(pdf_content)
                pdf_text = self.pdf_processor.clean_pdf_text(pdf_text)
                user_query = user_query + "\n\nContext from PDF: " + pdf_text
            except Exception as e:
                print(f"Error processing PDF: {e}")
        
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
        return {"status": "success", "message": "Conversation history cleared."}


# Example Flask API implementation
def create_flask_api():
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    diagnostics = DiagnosticFlow()
    
    @app.route('/api/diagnostic', methods=['POST'])
    def diagnostic_endpoint():
        data = request.json
        
        # Get required parameters
        user_query = data.get('query', '')
        
        # Get optional parameters with their explicit boolean values
        enable_web_search = data.get('enable_web_search')  # Will be True, False, or None
        enable_pdf_processing = data.get('enable_pdf_processing')  # Will be True, False, or None
        
        # Handle PDF if provided
        pdf_content = None
        if 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            pdf_content = pdf_file.read()
        
        # Process the query with explicit flag values
        response = diagnostics.process_query(
            user_query,
            enable_web_search=enable_web_search,
            enable_pdf_processing=enable_pdf_processing,
            pdf_content=pdf_content
        )
        
        return jsonify({
            'response': response,
            'web_search_used': enable_web_search if enable_web_search is not None else DEFAULT_ENABLE_WEB_SEARCH,
            'pdf_processing_used': enable_pdf_processing if enable_pdf_processing is not None else DEFAULT_ENABLE_PDF_PROCESSING
        })
    
    @app.route('/api/clear-history', methods=['POST'])
    def clear_history_endpoint():
        result = diagnostics.clear_history()
        return jsonify(result)
    
    return app


# Example FastAPI implementation
def create_fastapi_app():
    from fastapi import FastAPI, File, UploadFile, Form
    from pydantic import BaseModel
    from typing import Optional
    
    app = FastAPI()
    diagnostics = DiagnosticFlow()
    
    class DiagnosticRequest(BaseModel):
        query: str
        enable_web_search: Optional[bool] = None
        enable_pdf_processing: Optional[bool] = None
    
    @app.post("/api/diagnostic")
    async def process_diagnostic(request: DiagnosticRequest, pdf_file: Optional[UploadFile] = File(None)):
        # Handle PDF if provided
        pdf_content = None
        if pdf_file:
            pdf_content = await pdf_file.read()
        
        # Process the query with explicit flag values
        response = diagnostics.process_query(
            request.query,
            enable_web_search=request.enable_web_search,
            enable_pdf_processing=request.enable_pdf_processing,
            pdf_content=pdf_content
        )
        
        return {
            'response': response,
            'web_search_used': request.enable_web_search if request.enable_web_search is not None else DEFAULT_ENABLE_WEB_SEARCH,
            'pdf_processing_used': request.enable_pdf_processing if request.enable_pdf_processing is not None else DEFAULT_ENABLE_PDF_PROCESSING
        }
    
    @app.post("/api/clear-history")
    async def clear_history():
        result = diagnostics.clear_history()
        return result
    
    return app


if __name__ == "__main__":
    # This is just for demonstration purposes
    # In a real application, you would use the API implementations above
    
    # Example of how to use the DiagnosticFlow class directly
    diagnostics = DiagnosticFlow()
    
    # Example usage with explicit flag values (no user input)
    sample_query = "What could cause my car to make a grinding noise when braking?"
    response = diagnostics.process_query(
        sample_query,
        enable_web_search=True,  # Explicitly enable web search for this query
        enable_pdf_processing=False  # Explicitly disable PDF processing for this query
    )
    
    print("\nSample query:")
    print(sample_query)
    print("\nResponse:")
    print(response)
    
    # To start the Flask API
    # app = create_flask_api()
    # app.run(debug=True, port=5000)
    
    # To start the FastAPI
    # import uvicorn
    # app = create_fastapi_app()
    # uvicorn.run(app, host="0.0.0.0", port=8000)