"""
Presentation Planning Module: Plan presentations from lightweight content
This module now calls lightweight planner module functionality, providing efficient presentation plan generation
"""
import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("env.local"):
    load_dotenv("env.local")

# Import lightweight planner
from .lightweight_planner import LightweightPlanner, generate_lightweight_presentation_plan

# Try to import OpenAI-related packages
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class PresentationPlanner:
    """
    Presentation Planner - Lightweight Wrapper
    
    This class is now a wrapper for LightweightPlanner, maintaining backward compatibility
    """
    def __init__(
        self, 
        raw_content_path: str, 
        output_dir: str = "output",
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        language: str = "zh"
    ):
        """
        Initialize presentation planner
        
        Args:
            raw_content_path: Lightweight content JSON file path
            output_dir: Output directory
            model_name: Language model name to use
            temperature: Randomness level of model generation
            api_key: OpenAI API key
            language: Output language, zh for Chinese, en for English
        """
        # Create lightweight planner instance
        self.lightweight_planner = LightweightPlanner(
            lightweight_content_path=raw_content_path,
            output_dir=output_dir,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            language=language
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Compatibility attributes
        self.raw_content_path = raw_content_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.language = language
        
        # Presentation plan data
        self.presentation_plan = {}
        self.paper_info = {}
        self.key_content = {}
        self.slides_plan = []
        self.conversation_history = []
    
    def generate_presentation_plan(self) -> Dict[str, Any]:
        """
        Generate presentation plan
        
        Returns:
            Dict: Presentation plan
        """
        self.logger.info("Using lightweight planner to generate presentation plan...")
        
        # Call lightweight planner
        self.presentation_plan = self.lightweight_planner.generate_presentation_plan()
        
        if self.presentation_plan:
            # Update compatibility attributes
            self.paper_info = self.presentation_plan.get("paper_info", {})
            self.key_content = self.presentation_plan.get("key_content", {})
            self.slides_plan = self.presentation_plan.get("slides_plan", [])
            
            self.logger.info("Presentation plan generation completed")
        else:
            self.logger.error("Presentation plan generation failed")
        
        return self.presentation_plan
    
    def save_presentation_plan(self, presentation_plan, output_file=None):
        """
        Save presentation plan to JSON file
        
        Args:
            presentation_plan: Presentation plan
            output_file: Output file path, if None use default path
            
        Returns:
            str: Saved file path
        """
        return self.lightweight_planner.save_presentation_plan(presentation_plan, output_file)
    
    def interactive_refinement(self, initial_feedback=None) -> Dict[str, Any]:
        """
        Multi-turn interaction with user to optimize presentation plan
        
        Args:
            initial_feedback: User's initial feedback
            
        Returns:
            Dict: Optimized presentation plan
        """
        result = self.lightweight_planner.interactive_refinement(initial_feedback)
        
        # Update local attributes
        if result:
            self.presentation_plan = result
            self.paper_info = result.get("paper_info", {})
            self.key_content = result.get("key_content", {})
            self.slides_plan = result.get("slides_plan", [])
        
        return result
    
    def continue_conversation(self, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Continue conversation with user, update presentation plan
        
        Args:
            user_message: User's message
            
        Returns:
            Tuple: (Model response, Updated presentation plan)
        """
        response, updated_plan = self.lightweight_planner.continue_conversation(user_message)
        
        # Update local attributes
        if updated_plan:
            self.presentation_plan = updated_plan
            self.paper_info = updated_plan.get("paper_info", {})
            self.key_content = updated_plan.get("key_content", {})
            self.slides_plan = updated_plan.get("slides_plan", [])
        
        return response, updated_plan
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history
        
        Returns:
            List: Conversation history records
        """
        return self.lightweight_planner.get_conversation_history()

def generate_presentation_plan(raw_content_path, output_dir="output", model_name="gpt-4o", api_key=None, language="zh", user_feedback=None):
    """
    Generate presentation plan from lightweight content (convenience function)
    
    Args:
        raw_content_path: Lightweight content JSON file path
        output_dir: Output directory
        model_name: Language model name to use
        api_key: OpenAI API key
        language: Output language, zh for Chinese, en for English
        user_feedback: User's initial feedback (optional)
        
    Returns:
        tuple: (Presentation plan, Saved file path, Planner instance)
    """
    # Direct call to lightweight planner convenience function
    presentation_plan, plan_path, lightweight_planner = generate_lightweight_presentation_plan(
        lightweight_content_path=raw_content_path,
        output_dir=output_dir,
        model_name=model_name,
        api_key=api_key,
        language=language,
        user_feedback=user_feedback
    )
    
    # Create wrapper instance for compatibility
    if lightweight_planner:
        wrapper = PresentationPlanner(
            raw_content_path=raw_content_path,
            output_dir=output_dir,
            model_name=model_name,
            api_key=api_key,
            language=language
        )
        wrapper.lightweight_planner = lightweight_planner
        wrapper.presentation_plan = presentation_plan
        if presentation_plan:
            wrapper.paper_info = presentation_plan.get("paper_info", {})
            wrapper.key_content = presentation_plan.get("key_content", {})
            wrapper.slides_plan = presentation_plan.get("slides_plan", [])
        
        return presentation_plan, plan_path, wrapper
    
    return presentation_plan, plan_path, None
