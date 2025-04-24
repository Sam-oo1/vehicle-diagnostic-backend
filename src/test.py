import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from flask import Flask, request, jsonify
from flask_cors import CORS  # For handling CORS issues
import logging
from functools import lru_cache

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Default feature flags
DEFAULT_ENABLE_WEB_SEARCH = False
DEFAULT_ENABLE_PDF_PROCESSING = False


class DiagnosticFlow:
    """Orchestrates the flow between different agents for diagnostic processing"""
    
    def __init__(self, default_enable_web_search=DEFAULT_ENABLE_WEB_SEARCH, 
                 default_enable_pdf_processing=DEFAULT_ENABLE_PDF_PROCESSING):
        # Initialize LLM with a cached factory
        self.llm = self._get_llm_instance()
        
        # Initialize components
        self.doc_store = DocumentStore()
        self.message_history = MessageHistory("user_session")
        
        # Initialize agents with dependency injection
        self._initialize_agents()
        
        # Set default flags
        self.default_enable_web_search = default_enable_web_search
        self.default_enable_pdf_processing = default_enable_pdf_processing
        
        # Initialize search tool lazily
        self._search_tool = None
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()
        
        logger.info("DiagnosticFlow initialized successfully")
    
    @property
    def search_tool(self):
        """Lazy initialization of search tool"""
        if self._search_tool is None and os.getenv("TAVILY_API_KEY"):
            self._search_tool = TavilySearchResults(
                max_results=5,
                api_key=os.getenv("TAVILY_API_KEY")
            )
        return self._search_tool
    
    @lru_cache(maxsize=1)
    def _get_llm_instance(self):
        """Get an LLM instance with caching for performance"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
        genai.configure(api_key=api_key)
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0,
            max_retries=2
        )
    
    def _initialize_agents(self):
        """Initialize all agent components"""
        self.intent_agent = IntentRecognitionAgent("Intent Recognition", self.llm)
        self.query_agent = QueryRefinementAgent("Query Refinement", self.llm)
        self.info_agent = InformationGatheringAgent("Information Gathering", self.llm, self.doc_store)
        self.diagnostic_agent = DiagnosticAgent("Diagnostic Analysis", self.llm)
        self.reasoning_agent = ReasoningAgent("Reasoning", self.llm)
        self.modification_agent = ModificationAgent("Modification", self.llm)
        self.response_agent = ResponseFormulationAgent("Response Formulation", self.llm, self.message_history)
    
    def process_query(self, 
                     user_query: str, 
                     enable_web_search: Optional[bool] = None,
                     enable_pdf_processing: Optional[bool] = None,
                     pdf_content: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Process user query through the multi-agent pipeline
        
        Args:
            user_query (str): User's diagnostic query
            enable_web_search (bool, optional): Override default web search setting for this query
            enable_pdf_processing (bool, optional): Override default PDF processing for this query
            pdf_content (bytes, optional): PDF file content as bytes if applicable
            
        Returns:
            Dict[str, Any]: Dictionary containing response and processing flags
        """
        try:
            # Use provided flags or fall back to defaults
            use_web_search = enable_web_search if enable_web_search is not None else self.default_enable_web_search
            use_pdf_processing = enable_pdf_processing if enable_pdf_processing is not None else self.default_enable_pdf_processing
            
            # Configure search tool for this query
            if use_web_search:
                if self.search_tool:
                    self.info_agent.set_search_tool(self.search_tool)
                else:
                    logger.warning("Web search was enabled but Tavily API key is missing")
                    use_web_search = False
            else:
                self.info_agent.set_search_tool(None)
                
            # Add PDF context if provided and PDF processing is enabled
            pdf_text = None
            if use_pdf_processing and pdf_content:
                try:
                    pdf_text = self.pdf_processor.extract_text_from_bytes(pdf_content)
                    pdf_text = self.pdf_processor.clean_pdf_text(pdf_text)
                    user_query = user_query + "\n\nContext from PDF: " + pdf_text
                except Exception as e:
                    logger.error(f"Error processing PDF: {e}")
                    use_pdf_processing = False
            
            # Initial state with user query
            state = {"user_query": user_query}
            
            # Intent Recognition
            intent_state = self.intent_agent.process(state)
            state.update(intent_state)
            
            # Special handling for modification requests
            response = None
            if state.get("intent") == "modify_previous":
                response = self._handle_modification_intent(state)
            else:
                # Normal flow for new questions
                response = self._process_standard_flow(state)
            
            return {
                "response": response,
                "web_search_used": use_web_search,
                "pdf_processing_used": use_pdf_processing,
                "pdf_text_extracted": pdf_text is not None
            }
            
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            return {
                "response": "Sorry, there was an error processing your query.",
                "error": str(e),
                "web_search_used": False,
                "pdf_processing_used": False
            }
    
    def _handle_modification_intent(self, state: Dict[str, Any]) -> str:
        """Handle modification intent specifically"""
        last_response = self.message_history.get_last_assistant_message()
        if not last_response:
            return "I don't have any previous responses to modify."
            
        state["previous_response"] = last_response.content
        
        # Get the modified version
        modification_state = self.modification_agent.process(state)
        state.update(modification_state)
        
        # Generate final response
        response_state = self.response_agent.process(state)
        state.update(response_state)
        
        # Update the last message in history instead of adding new ones
        final_response = state.get("final_response", "Sorry, there was an error modifying the response.")
        self.message_history.replace_last_assistant_message(final_response)
        
        return final_response
    
    def _process_standard_flow(self, state: Dict[str, Any]) -> str:
        """Process a standard (new question) flow"""
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
        user_query = state.get("user_query", "")
        final_response = state.get("final_response", "Sorry, there was an error processing your query.")
        
        self.message_history.add_message(HumanMessage(content=user_query))
        self.message_history.add_message(AIMessage(content=final_response))
        
        return final_response
    
    def clear_history(self) -> Dict[str, str]:
        """Clear conversation history"""
        self.message_history.clear()
        logger.info("Conversation history cleared")
        return {"status": "success", "message": "Conversation history cleared."}


# API Implementation
def create_api():
    """Create and configure Flask API with the DiagnosticFlow"""
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Create a singleton instance of DiagnosticFlow
    diagnostics = DiagnosticFlow()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Simple health check endpoint"""
        return jsonify({
            'status': 'ok',
            'version': '1.0.0',
        })
    
    @app.route('/api/query', methods=['POST'])
    def diagnostic_endpoint():
        """Main endpoint for processing diagnostic queries"""
        try:
            # Handle JSON data
            if request.content_type and 'application/json' in request.content_type:
                data = request.json
                user_query = data.get('query', '')
                enable_web_search = data.get('searchEnabled')
                enable_pdf_processing = data.get('pdfEnabled')
                pdf_content = None
            # Handle form data with potential file upload
            else:
                user_query = request.form.get('query', '')
                enable_web_search = request.form.get('searchEnabled')
                enable_pdf_processing = request.form.get('pdfEnabled')
                pdf_content = None
                
                # Convert string values to boolean if needed
                if isinstance(enable_web_search, str):
                    enable_web_search = enable_web_search.lower() == 'true'
                if isinstance(enable_pdf_processing, str):
                    enable_pdf_processing = enable_pdf_processing.lower() == 'true'
                
            # Handle PDF if provided
            if 'pdf_file' in request.files:
                pdf_file = request.files['pdf_file']
                if pdf_file.filename:
                    pdf_content = pdf_file.read()
            
            if not user_query:
                return jsonify({
                    'error': 'Missing required parameter: query',
                    'status': 'error'
                }), 400
            
            # Process the query with explicit flag values
            result = diagnostics.process_query(
                user_query,
                enable_web_search=enable_web_search,
                enable_pdf_processing=enable_pdf_processing,
                pdf_content=pdf_content
            )
            
            return jsonify(result)
            
        except Exception as e:
            logger.exception(f"Error in diagnostic endpoint: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500
    
    @app.route('/api/clear-history', methods=['POST'])
    def clear_history_endpoint():
        """Endpoint to clear conversation history"""
        try:
            result = diagnostics.clear_history()
            return jsonify(result)
        except Exception as e:
            logger.exception(f"Error clearing history: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500
    
    return app


# Run the application if executed directly
if __name__ == "__main__":
    api = create_api()
    port = int(os.getenv("PORT", 5000))
    
    logger.info(f"Starting Diagnostic API server on port {port}")
    api.run(host="0.0.0.0", port=port, debug=False)