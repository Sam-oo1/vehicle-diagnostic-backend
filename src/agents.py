import json
from typing import Dict, List, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.tools.tavily_search import TavilySearchResults

class Agent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, llm: BaseChatModel):
        self.name = name
        self.llm = llm
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        raise NotImplementedError("Each agent must implement this method")


class IntentRecognitionAgent(Agent):
    """Agent responsible for recognizing user intent, especially for meta-requests"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_query = input_data.get("user_query", "")
        
        prompt = f"""You are an intent recognition system for vehicle diagnostics. 
        
        Analyze the following user query and determine what the user wants:
        
        "{user_query}"
        
        Choose exactly one of the following intent categories:
        1. "new_question" - User is asking a new vehicle diagnostic question
        2. "modify_previous" - User wants to modify a previous response (make it shorter, longer, simpler, etc.)
        3. "followup" - User is asking a follow-up question about previous information
        4. "meta_command" - User is giving a system command (like 'clear history')
        
        Also extract any specific modification request if the intent is "modify_previous" (e.g., "make shorter", "simplify", etc.)
        
        Return your answer as a JSON object with fields "intent" and "modification_type" (if applicable).
        """
        
        try:
            response = self.llm.invoke(prompt)
            intent_text = response.content
            
            # Try to parse as JSON
            try:
                intent_data = json.loads(intent_text)
                intent = intent_data.get("intent", "new_question")
                modification_type = intent_data.get("modification_type", "")
            except json.JSONDecodeError:
                # If not valid JSON, attempt to extract intent from text
                intent = "new_question"
                modification_type = ""
                
                if "modify_previous" in intent_text.lower():
                    intent = "modify_previous"
                    # Try to extract modification type
                    if "shorter" in intent_text.lower():
                        modification_type = "shorter"
                    elif "longer" in intent_text.lower():
                        modification_type = "longer"
                    elif "simpl" in intent_text.lower():
                        modification_type = "simplify"
                elif "followup" in intent_text.lower():
                    intent = "followup"
                elif "meta_command" in intent_text.lower():
                    intent = "meta_command"
            
            print(f"\n✓ Intent recognized: {intent}")
            return {
                "intent": intent, 
                "modification_type": modification_type
            }
        except Exception as e:
            print(f"Error in intent recognition: {e}")
            return {"intent": "new_question", "modification_type": ""}


class QueryRefinementAgent(Agent):
    """Agent responsible for refining user queries"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_query = input_data.get("user_query", "")
        intent = input_data.get("intent", "new_question")
        
        # Skip refinement for modification requests
        if intent == "modify_previous":
            return {"refined_query": user_query}
        
        prompt = f"""You are a vehicle diagnostic expert. Refine the following vehicle diagnostic user query to make it specific, structured, and clear for troubleshooting:
        
        "{user_query}"
        
        Focus ONLY on vehicle diagnostic issues. Ignore any unrelated queries.
        Format the refined query to include:
        1. The specific symptom or issue (with details about when it occurs)
        2. Relevant vehicle information (if provided)
        3. Any potential causes the user might be concerned about
        4. Clear questions about diagnosis or repair options
        5. If the querry is irrespective of any car give general response that is car agnostic.
        
        Return ONLY the refined version of the query.
        """
        
        try:
            response = self.llm.invoke(prompt)
            refined_query = response.content
            print("\n✓ Query refined")
            return {"refined_query": refined_query}
        except Exception as e:
            print(f"Error in query refinement: {e}")
            return {"refined_query": user_query}


class InformationGatheringAgent(Agent):
    """Agent responsible for gathering information from multiple sources"""
    
    def __init__(self, name: str, llm: BaseChatModel, doc_store):
        super().__init__(name, llm)
        self.doc_store = doc_store
        self.search_tool = None
    
    def set_search_tool(self, search_tool):
        self.search_tool = search_tool
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        intent = input_data.get("intent", "new_question")
        
        # Skip information gathering for modification requests
        if intent == "modify_previous":
            return {"search_results": "", "internal_documents": ""}
            
        refined_query = input_data.get("refined_query", "")
        
        # Get search results if web search is enabled
        formatted_results = ""
        if self.search_tool:
            try:
                search_results = self.search_tool.invoke(refined_query)
                if isinstance(search_results, list):
                    for i, result in enumerate(search_results, 1):
                        if isinstance(result, dict):
                            formatted_results += f"Result {i}: {result.get('title', 'No title')}\n"
                            formatted_results += f"{result.get('content', 'No content')}\n\n"
                print("\n✓ Web search completed")
            except Exception as e:
                print(f"Error in search: {e}")
                formatted_results = "No search results available due to an error."
        else:
            print("\n⚠ Web search is disabled")
            formatted_results = "Web search is currently disabled."
        
        # Get internal documents
        try:
            retrieved_docs = self.doc_store.retrieve(refined_query)
            print("\n✓ Internal documents retrieved ")
        except Exception as e:
            print(f"Error in document retrieval: {e}")
            retrieved_docs = "No internal documents available."
        
        return {
            "search_results": formatted_results,
            "internal_documents": retrieved_docs
        }


class DiagnosticAgent(Agent):
    """Agent responsible for initial diagnostic analysis"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        intent = input_data.get("intent", "new_question")
        
        # Skip diagnostic analysis for modification requests
        if intent == "modify_previous":
            return {"diagnostic_response": ""}
            
        refined_query = input_data.get("refined_query", "")
        search_results = input_data.get("search_results", "")
        internal_documents = input_data.get("internal_documents", "")
        
        prompt = f"""
        You are an **BMW expert car diagnostic assistant** who is assisting an expert car mechanic or technician in their work.
        my car suite only includes BMW cars if you encounter any other car brand, simply return a response saying you only cater to BMW cars.
        You know nothing about other cars or car brands and if you need any eg. for the query to propose use model name as **BMW X1** or **X2** or **X3**.
        And the Internal document i am providng you is the data from forums of BMW bimmers and other specific models. So you only need to 

        User query:
        {refined_query}
        
        Search result:
        {search_results}
        
        **only diagnose the data based on these Retrieval Output and this output is from my RAG:**:
        {internal_documents}
        
        Based on all this information, provide a clear and helpful diagnostic response.
        """
        try:
            response = self.llm.invoke(prompt)
            diagnostic_response = response.content
            print("\n✓ Initial diagnostic response generated")
            return {"diagnostic_response": diagnostic_response}
        except Exception as e:
            print(f"Error generating diagnostic response: {e}")
            return {"diagnostic_response": "I'm having trouble generating a response based on the information available."}


class ReasoningAgent(Agent):
    """Agent responsible for evaluating and improving the diagnostic response"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        intent = input_data.get("intent", "new_question")
        
        # Handle modification requests differently
        if intent == "modify_previous":
            return {"reasoning_feedback": ""}
        
        refined_query = input_data.get("refined_query", "")
        diagnostic_response = input_data.get("diagnostic_response", "")
        search_results = input_data.get("search_results", "")
        internal_documents = input_data.get("internal_documents", "")
        
        prompt = f"""
        You are an BMW expert vehicle diagnostics analyst and an expert mechanic and vehicle technician.
        my car suite only includes BMW cars if you encounter any other car brand, simply return a response saying you only cater to BMW cars.
        You know nothing about other cars or car brands. 
        And the Internal document i am providng you is the data from forums of BMW bimmers and other specific models. So you only need to give the response based on the Retrieval Output from my RAG
        the retrieval output is data and my vehicle is BMW X1 dont use **HONDA civic**
        Analyze the diagnostic response based on the original user query and available supporting information.

        Query: {refined_query}

        LLM Response: {diagnostic_response}
        
        Web Search Output: {search_results}
        
        Retrieval Output: {internal_documents}

        ---

        Your task:
        - Evaluate the accuracy of the LLM response
        - Point out any issues or missing considerations
        - Provide step-by-step reasoning
        - Provide a final verdict on the LLM response ("Correct", "Partially correct", "Incorrect")
        - Generate a final human-readable response for the user that is **clear and concise**.
        
        Please write a clear, complete, and professional final answer that:
        - Fixes any mistakes
        - Adds any missing details
        - Is easy to understand for a non-technical person
        - Reflects expert-level insight

        Make it sound human and informative and don't make it much longer, keep it in appropriate length so that it is not boring to read but covers all necessary things.
        """
        
        try:
            response = self.llm.invoke(prompt)
            reasoning_feedback = response.content
            print("\n✓ Response evaluation completed of reasoning agent")
            return {"reasoning_feedback": reasoning_feedback}
        except Exception as e:
            print(f"Error in reasoning: {e}")
            return {"reasoning_feedback": "Unable to evaluate the response due to an error."}


class ModificationAgent(Agent):
    """Agent responsible for modifying previous responses"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        intent = input_data.get("intent", "")
        
        if intent != "modify_previous":
            return {"modified_response": ""}
            
        user_query = input_data.get("user_query", "")
        modification_type = input_data.get("modification_type", "")
        previous_response = input_data.get("previous_response", "")
        
        prompt = f"""
        You are an expert at modifying text based on user requests. The user has asked to modify a previous response. 
        
        Previous response:
        {previous_response}
        
        User's modification request:
        {user_query}
        
        Detected modification type: {modification_type}
        
        Please modify the previous response according to the user's request. Maintain the technical accuracy 
        and expert knowledge of the original response, while applying the requested changes.
        
        If the modification request is to make it shorter, focus on removing unnecessary details while keeping 
        the essential diagnostic information. If it's to make it simpler, reduce technical jargon but keep the 
        core diagnostic advice.
        
        Return only the modified response without any explanation of what you changed.
        """
        
        try:
            response = self.llm.invoke(prompt)
            modified_response = response.content
            print("\n✓ Previous response modified")
            return {"modified_response": modified_response}
        except Exception as e:
            print(f"Error modifying previous response: {e}")
            return {"modified_response": "I'm having trouble modifying the previous response."}


class ResponseFormulationAgent(Agent):
    """Agent responsible for formulating the final response"""
    
    def __init__(self, name: str, llm: BaseChatModel, message_history):
        super().__init__(name, llm)
        self.message_history = message_history
        
        self.prompt_template = PromptTemplate.from_template("""
        You are a professional vehicle diagnostics assistant assisting an expert car mechanic or technician.

        Your task is to generate a final, verified, and user-friendly answer for a vehicle mechanic, using:
        - The original question: {query}
        - The initial LLM response: {diagnostic_response}
        - The detailed evaluation provided by a reasoning agent: {reasoning_feedback}
        - Previous conversation context: {chat_history}

        ---

        Please write a clear, complete, and professional final answer that:
        - Fixes any mistakes
        - Adds any missing details
        - Is easy to understand for a non-technical person
        - Reflects expert-level insight
        - Maintains continuity with previous conversation (if relevant)

        Make it sound human and informative and don't make it much longer, keep it in appropriate length so that it is not boring to read but covers all necessary things.
                                                        
        Give the Response in strictly markdown format and the response should be around 200 to 300 words.
        The Markdown format should follow hierarchial headings and body text format.
        """)
    
    def _format_chat_history(self) -> str:
        """Format chat history as a string"""
        formatted_history = ""
        messages = self.message_history.get_messages()
        
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                formatted_history += f"User: {msg.content}\n\n"
            elif isinstance(msg, AIMessage):
                formatted_history += f"Assistant: {msg.content}\n\n"
        
        return formatted_history
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        intent = input_data.get("intent", "new_question")
        user_query = input_data.get("user_query", "")
        
        # Handle modification requests
        if intent == "modify_previous":
            modified_response = input_data.get("modified_response", "")
            if modified_response:
                return {"final_response": modified_response}
        
        refined_query = input_data.get("refined_query", "")
        diagnostic_response = input_data.get("diagnostic_response", "")
        reasoning_feedback = input_data.get("reasoning_feedback", "")
        
        chat_history = self._format_chat_history()
        
        # Prepare the input for the template
        prompt_input = {
            "query": refined_query,
            "diagnostic_response": diagnostic_response,
            "reasoning_feedback": reasoning_feedback,
            "chat_history": chat_history
        }
        
        # Process the prompt
        try:
            prompt = self.prompt_template.format(**prompt_input)
            response = self.llm.invoke(prompt)
            final_response = response.content
            
            return {"final_response": final_response}
        except Exception as e:
            print(f"Error generating final response: {e}")
            error_response = "I apologize, but I'm having trouble generating a final response."
            return {"final_response": error_response}
