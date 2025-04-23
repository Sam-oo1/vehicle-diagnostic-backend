import os
import re
import json
import PyPDF2
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class MessageHistory:
    """Handles conversation history persistence"""
    
    def __init__(self, session_id: str, history_file="conversation_history.json"):
        self.session_id = session_id
        self.messages = []
        self.history_file = history_file
        self._load_history()
    
    def _load_history(self):
        """Load existing conversation history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    for msg in history:
                        if msg["role"] == "user":
                            self.messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            self.messages.append(AIMessage(content=msg["content"]))
                        elif msg["role"] == "system":
                            self.messages.append(SystemMessage(content=msg["content"]))
            except json.JSONDecodeError:
                print("Error loading history file. Starting with empty history.")
    
    def add_message(self, message):
        """Add a new message to history and save to file"""
        self.messages.append(message)
        self._save_history()
    
    def _save_history(self):
        """Save messages to file"""
        messages_json = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                continue
            messages_json.append({"role": role, "content": msg.content})
        
        with open(self.history_file, 'w') as f:
            json.dump(messages_json, f, indent=2)
    
    def get_messages(self):
        """Return all messages in history"""
        return self.messages
    
    def get_last_assistant_message(self) -> Optional[AIMessage]:
        """Get the most recent assistant message"""
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage):
                return msg
        return None
    
    def replace_last_assistant_message(self, new_content: str):
        """Replace the content of the last assistant message"""
        for i in range(len(self.messages) - 1, -1, -1):
            if isinstance(self.messages[i], AIMessage):
                self.messages[i] = AIMessage(content=new_content)
                self._save_history()
                return True
        return False
    
    def clear(self):
        """Clear all history"""
        self.messages = []
        if os.path.exists(self.history_file):
            os.remove(self.history_file)


class PDFProcessor:
    """Handles PDF document processing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file: str) -> str:
        """Extract text content from a PDF file"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    
    @staticmethod
    def clean_pdf_text(text: str) -> str:
        """Clean and format extracted PDF text"""
        # Replace multiple newlines and tabs with a single space
        text = re.sub(r'[\r\n\t]+', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove unnecessary punctuation (keep . % / ° :)
        text = re.sub(r'[^\w\s.%/°:]', '', text)
        # Strip leading/trailing whitespace
        return text.strip()


class DocumentStore:
    """Manages vector storage and retrieval of documents"""
    
    def __init__(self, vector_db_path="./chroma_db"):
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize or load the vector database"""
        try:
            self.db = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
            print("Vector database loaded successfully")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating an empty vector store...")
            self.db = Chroma(embedding_function=self.embeddings)
            self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the vector store"""
        try:
            self.db.add_texts(texts=texts, metadatas=metadatas)
            self.db.persist()
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def retrieve(self, query: str) -> str:
        """Retrieve relevant documents based on query"""
        try:
            docs = self.retriever.invoke(query)
            if docs:
                return "\n\n".join(d.page_content for d in docs)
            else:
                return "No relevant documents found in the knowledge base."
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return "Error retrieving documents from the knowledge base."
