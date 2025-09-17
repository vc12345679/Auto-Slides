"""
Repair Agent: Automatic Content Repair Based on Verification Results
Automatically fixes issues identified by the Verification Agent
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

# Try to import LangChain packages and our unified interface
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
    
    # Import our unified LLM interface and parameter system
    from modules.llm_interface import LLMInterface
    from config.llm_params import TaskType
    UNIFIED_INTERFACE_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    UNIFIED_INTERFACE_AVAILABLE = False


class RepairAgent:
    """
    Repair Agent for automatic content repair based on verification results
    
    This agent automatically fixes issues identified by the Verification Agent:
    - Factual errors and inconsistencies
    - Missing key information
    - Data inaccuracies
    - Hallucinated content removal
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,  # Low temperature for precise repairs
        api_key: Optional[str] = None,
        language: str = "zh"
    ):
        """
        Initialize the Repair Agent
        
        Args:
            model_name: Language model to use for repairs
            temperature: Model temperature (low for precise repairs)
            api_key: OpenAI API key
            language: Output language for repair operations
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Try to load API key from .env file
        if not api_key:
            try:
                from dotenv import load_dotenv
                import os
                load_dotenv()
                api_key = os.environ.get("OPENAI_API_KEY")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load .env file: {e}")
        
        self.api_key = api_key
        self.language = language
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize unified LLM interface if available
        if UNIFIED_INTERFACE_AVAILABLE:
            self.llm_interface = LLMInterface(model_name, api_key)
            self.logger.info("Unified LLM interface initialized for repair with task-optimized parameters")
        else:
            self.llm_interface = None
            self.logger.warning("Unified LLM interface not available, using fallback methods")
        
        # Initialize language model (fallback)
        self._init_model()
        
        # Repair statistics
        self.repair_stats = {}
    
    def _init_model(self):
        """Initialize the language model"""
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning("LangChain not available, repair functionality disabled")
            self.llm = None
            return
        
        if not self.api_key:
            self.logger.warning("No OpenAI API key provided, repair functionality disabled")
            self.llm = None
            return
        
        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key
            )
            self.logger.info(f"Repair Agent initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize language model: {str(e)}")
            self.llm = None
    
    def repair_presentation_plan(
        self,
        presentation_plan_path: str,
        verification_report_path: str,
        output_dir: str = "output"
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Repair presentation plan based on verification results
        
        Args:
            presentation_plan_path: Path to original presentation plan JSON
            verification_report_path: Path to verification report JSON
            output_dir: Directory to save repaired plan
            
        Returns:
            Tuple of (repair_success, repair_report, repaired_plan_path)
        """
        if not self.llm:
            self.logger.error("Language model not available for repair")
            return False, {"error": "Language model not initialized"}, ""
        
        self.logger.info("Starting automatic presentation plan repair...")
        
        try:
            # Load content files
            presentation_plan = self._load_json_file(presentation_plan_path)
            verification_report = self._load_json_file(verification_report_path)
            
            if not presentation_plan or not verification_report:
                return False, {"error": "Failed to load content files"}, ""
            
            # Initialize repair report
            repair_report = {
                "repair_timestamp": self._get_timestamp(),
                "original_plan_path": presentation_plan_path,
                "verification_report_path": verification_report_path,
                "repairs_applied": {},
                "repair_summary": {}
            }
            
            # Extract verification results
            verification_results = verification_report.get("verification_results", {})
            
            # Make a copy of the presentation plan for modification
            repaired_plan = presentation_plan.copy()
            
            # 1. Repair Factual Inconsistencies
            self.logger.info("Repairing factual inconsistencies...")
            factual_repairs = self._repair_factual_inconsistencies(
                repaired_plan, verification_results.get("factual_consistency", {})
            )
            repair_report["repairs_applied"]["factual_consistency"] = factual_repairs
            
            # 2. Add Missing Key Information
            self.logger.info("Adding missing key information...")
            missing_info_repairs = self._add_missing_key_information(
                repaired_plan, verification_results.get("key_information_preservation", {})
            )
            repair_report["repairs_applied"]["missing_information"] = missing_info_repairs
            
            # 3. Fix Data Inaccuracies
            self.logger.info("Fixing data inaccuracies...")
            data_repairs = self._fix_data_inaccuracies(
                repaired_plan, verification_results.get("data_accuracy", {})
            )
            repair_report["repairs_applied"]["data_accuracy"] = data_repairs
            
            # 4. Remove Hallucinated Content
            self.logger.info("Removing hallucinated content...")
            hallucination_repairs = self._remove_hallucinated_content(
                repaired_plan, verification_results.get("hallucination_detection", {})
            )
            repair_report["repairs_applied"]["hallucination_removal"] = hallucination_repairs
            
            # Generate repair summary
            repair_summary = self._generate_repair_summary(repair_report["repairs_applied"])
            repair_report["repair_summary"] = repair_summary
            
            # Save repaired presentation plan
            os.makedirs(output_dir, exist_ok=True)
            repaired_plan_path = os.path.join(output_dir, "repaired_presentation_plan.json")
            with open(repaired_plan_path, 'w', encoding='utf-8') as f:
                json.dump(repaired_plan, f, ensure_ascii=False, indent=2)
            
            # Save repair report
            repair_report_path = os.path.join(output_dir, "repair_report.json")
            with open(repair_report_path, 'w', encoding='utf-8') as f:
                json.dump(repair_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Repair completed. Repaired plan saved to: {repaired_plan_path}")
            self.logger.info(f"Repair report saved to: {repair_report_path}")
            
            # Determine repair success
            repair_success = any(repairs for repairs in repair_report["repairs_applied"].values())
            
            return repair_success, repair_report, repaired_plan_path
            
        except Exception as e:
            self.logger.error(f"Repair process failed: {str(e)}")
            return False, {"error": str(e)}, ""
    
    def _repair_factual_inconsistencies(self, presentation_plan: Dict, factual_results: Dict) -> List[Dict]:
        """Repair factual inconsistencies identified by verification"""
        repairs = []
        inconsistencies = factual_results.get("inconsistencies", [])
        
        for inconsistency in inconsistencies:
            if inconsistency.get("severity") in ["medium", "high", "critical"]:
                try:
                    # Generate corrected content
                    corrected_content = self._generate_corrected_content(
                        inconsistency.get("presentation_content", ""),
                        inconsistency.get("original_content", ""),
                        inconsistency.get("description", "")
                    )
                    
                    if corrected_content:
                        # Find and replace in presentation plan
                        replacement_made = self._replace_content_in_plan(
                            presentation_plan,
                            inconsistency.get("presentation_content", ""),
                            corrected_content
                        )
                        
                        if replacement_made:
                            repairs.append({
                                "type": "factual_correction",
                                "original": inconsistency.get("presentation_content", ""),
                                "corrected": corrected_content,
                                "reason": inconsistency.get("description", ""),
                                "severity": inconsistency.get("severity", "unknown")
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Failed to repair factual inconsistency: {e}")
        
        return repairs
    
    def _add_missing_key_information(self, presentation_plan: Dict, preservation_results: Dict) -> List[Dict]:
        """Add missing key information to presentation"""
        repairs = []
        missing_info = preservation_results.get("missing_key_info", [])
        
        for missing_item in missing_info:
            if missing_item.get("importance") in ["high", "critical"]:
                try:
                    # Generate content for missing information
                    additional_content = self._generate_missing_content(
                        missing_item.get("missing_content", ""),
                        missing_item.get("category", ""),
                        presentation_plan
                    )
                    
                    if additional_content:
                        # Add to appropriate slide
                        slide_added = self._add_content_to_appropriate_slide(
                            presentation_plan,
                            additional_content,
                            missing_item.get("category", "")
                        )
                        
                        if slide_added:
                            repairs.append({
                                "type": "missing_information_added",
                                "content": additional_content,
                                "category": missing_item.get("category", ""),
                                "importance": missing_item.get("importance", ""),
                                "slide_location": slide_added
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Failed to add missing information: {e}")
        
        return repairs
    
    def _fix_data_inaccuracies(self, presentation_plan: Dict, data_results: Dict) -> List[Dict]:
        """Fix data inaccuracies in presentation"""
        repairs = []
        data_inconsistencies = data_results.get("data_inconsistencies", [])
        
        for inconsistency in data_inconsistencies:
            if inconsistency.get("severity") in ["medium", "high", "critical"]:
                try:
                    original_value = inconsistency.get("original_value", "")
                    presentation_value = inconsistency.get("presentation_value", "")
                    
                    if original_value and presentation_value and original_value != presentation_value:
                        # Replace incorrect data with correct data
                        replacement_made = self._replace_content_in_plan(
                            presentation_plan,
                            presentation_value,
                            original_value
                        )
                        
                        if replacement_made:
                            repairs.append({
                                "type": "data_correction",
                                "incorrect_value": presentation_value,
                                "correct_value": original_value,
                                "location": inconsistency.get("location", ""),
                                "inconsistency_type": inconsistency.get("type", "")
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Failed to fix data inaccuracy: {e}")
        
        return repairs
    
    def _remove_hallucinated_content(self, presentation_plan: Dict, hallucination_results: Dict) -> List[Dict]:
        """Remove hallucinated content from presentation"""
        repairs = []
        potential_hallucinations = hallucination_results.get("potential_hallucinations", [])
        
        for hallucination in potential_hallucinations:
            if hallucination.get("severity") in ["high", "critical"]:
                try:
                    hallucinated_content = hallucination.get("content", "")
                    
                    # Remove or replace hallucinated content
                    if hallucinated_content:
                        # Try to find a factual replacement
                        replacement_content = self._generate_factual_replacement(
                            hallucinated_content,
                            hallucination.get("explanation", "")
                        )
                        
                        if replacement_content:
                            replacement_made = self._replace_content_in_plan(
                                presentation_plan,
                                hallucinated_content,
                                replacement_content
                            )
                        else:
                            # If no replacement, remove the content
                            replacement_made = self._remove_content_from_plan(
                                presentation_plan,
                                hallucinated_content
                            )
                        
                        if replacement_made:
                            repairs.append({
                                "type": "hallucination_removal",
                                "removed_content": hallucinated_content,
                                "replacement": replacement_content or "[REMOVED]",
                                "reason": hallucination.get("explanation", ""),
                                "severity": hallucination.get("severity", "")
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Failed to remove hallucinated content: {e}")
        
        return repairs
    
    def _generate_corrected_content(self, incorrect_content: str, original_content: str, description: str) -> str:
        """Generate corrected content based on original content"""
        try:
            correction_prompt = f"""
请根据原始论文内容修正演示文稿中的表述错误。

**错误描述**: {description}

**原始论文内容**: {original_content}

**演示文稿错误内容**: {incorrect_content}

请提供修正后的准确表述，保持学术严谨性：
"""
            
            if self.llm_interface:
                # 使用修复任务专用的LLM调用方法
                result = self.llm_interface.call_for_repair(
                    "", correction_prompt, json_mode=False
                )
            else:
                response = self.llm.invoke([HumanMessage(content=correction_prompt)])
                result = response.content
            
            return result.strip() if result else ""
            
        except Exception as e:
            self.logger.error(f"Failed to generate corrected content: {e}")
            return ""
    
    def _generate_missing_content(self, missing_description: str, category: str, presentation_plan: Dict) -> str:
        """Generate content for missing key information"""
        try:
            generation_prompt = f"""
根据缺失信息描述，为演示文稿生成相应的内容。

**缺失信息类别**: {category}
**缺失内容描述**: {missing_description}

**当前演示文稿结构**: {json.dumps(presentation_plan.get('slides_plan', [])[:3], ensure_ascii=False, indent=2)}...

请生成简洁、准确的内容来补充这个缺失信息：
"""
            
            if self.llm_interface:
                # 使用修复任务专用的LLM调用方法  
                result = self.llm_interface.call_for_repair(
                    "", generation_prompt, json_mode=False
                )
            else:
                response = self.llm.invoke([HumanMessage(content=generation_prompt)])
                result = response.content
            
            return result.strip() if result else ""
            
        except Exception as e:
            self.logger.error(f"Failed to generate missing content: {e}")
            return ""
    
    def _generate_factual_replacement(self, hallucinated_content: str, explanation: str) -> str:
        """Generate factual replacement for hallucinated content"""
        try:
            replacement_prompt = f"""
为被识别为幻觉的内容提供事实性替代内容。

**幻觉内容**: {hallucinated_content}
**问题说明**: {explanation}

请提供准确的、基于事实的替代内容，如果无法确定准确信息则返回空字符串：
"""
            
            if self.llm_interface:
                # 使用修复任务专用的LLM调用方法
                result = self.llm_interface.call_for_repair(
                    "", replacement_prompt, json_mode=False
                )
            else:
                response = self.llm.invoke([HumanMessage(content=replacement_prompt)])
                result = response.content
            
            # If result contains uncertainty indicators, return empty
            uncertainty_indicators = ["不确定", "可能", "也许", "不清楚", "无法确认"]
            if any(indicator in result for indicator in uncertainty_indicators):
                return ""
            
            return result.strip() if result else ""
            
        except Exception as e:
            self.logger.error(f"Failed to generate factual replacement: {e}")
            return ""
    
    def _replace_content_in_plan(self, presentation_plan: Dict, old_content: str, new_content: str) -> bool:
        """Replace content in presentation plan"""
        try:
            slides = presentation_plan.get("slides_plan", [])
            replacement_made = False
            
            for slide in slides:
                # Check in slide title
                if old_content in slide.get("title", ""):
                    slide["title"] = slide["title"].replace(old_content, new_content)
                    replacement_made = True
                
                # Check in slide content
                content_list = slide.get("content", [])
                for i, content_item in enumerate(content_list):
                    if old_content in content_item:
                        content_list[i] = content_item.replace(old_content, new_content)
                        replacement_made = True
            
            return replacement_made
            
        except Exception as e:
            self.logger.error(f"Failed to replace content in plan: {e}")
            return False
    
    def _remove_content_from_plan(self, presentation_plan: Dict, content_to_remove: str) -> bool:
        """Remove content from presentation plan"""
        try:
            slides = presentation_plan.get("slides_plan", [])
            removal_made = False
            
            for slide in slides:
                # Check in slide content
                content_list = slide.get("content", [])
                original_length = len(content_list)
                
                # Remove content items that contain the problematic content
                slide["content"] = [item for item in content_list if content_to_remove not in item]
                
                if len(slide["content"]) < original_length:
                    removal_made = True
            
            return removal_made
            
        except Exception as e:
            self.logger.error(f"Failed to remove content from plan: {e}")
            return False
    
    def _add_content_to_appropriate_slide(self, presentation_plan: Dict, content: str, category: str) -> str:
        """Add content to the most appropriate slide based on category"""
        try:
            slides = presentation_plan.get("slides_plan", [])
            
            # Define category to slide type mapping
            category_mapping = {
                "contributions": ["introduction", "overview", "summary"],
                "methodology": ["method", "approach", "implementation"],
                "results": ["result", "experiment", "evaluation"],
                "conclusions": ["conclusion", "summary", "future"]
            }
            
            target_slide_types = category_mapping.get(category, [])
            
            # Find the most appropriate slide
            for slide in slides:
                slide_title = slide.get("title", "").lower()
                if any(slide_type in slide_title for slide_type in target_slide_types):
                    slide.setdefault("content", []).append(content)
                    return f"Slide {slide.get('slide_number', 'unknown')}: {slide.get('title', '')}"
            
            # If no appropriate slide found, add to the last slide
            if slides:
                slides[-1].setdefault("content", []).append(content)
                return f"Slide {slides[-1].get('slide_number', 'unknown')}: {slides[-1].get('title', '')}"
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to add content to slide: {e}")
            return ""
    
    def _generate_repair_summary(self, repairs_applied: Dict) -> Dict[str, Any]:
        """Generate summary of all repairs applied"""
        summary = {
            "total_repairs": 0,
            "repair_types": {},
            "critical_fixes": 0,
            "overall_improvement": "Minimal"
        }
        
        for repair_type, repairs in repairs_applied.items():
            repair_count = len(repairs) if repairs else 0
            summary["total_repairs"] += repair_count
            summary["repair_types"][repair_type] = repair_count
            
            # Count critical fixes
            if repairs:
                critical_count = len([r for r in repairs if r.get("severity") in ["high", "critical"]])
                summary["critical_fixes"] += critical_count
        
        # Determine overall improvement level
        if summary["total_repairs"] >= 10:
            summary["overall_improvement"] = "Significant"
        elif summary["total_repairs"] >= 5:
            summary["overall_improvement"] = "Moderate"
        elif summary["total_repairs"] >= 1:
            summary["overall_improvement"] = "Minor"
        
        return summary
    
    def _load_json_file(self, file_path: str) -> Optional[Dict]:
        """Load and parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {file_path}: {str(e)}")
            return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reporting"""
        from datetime import datetime
        return datetime.now().isoformat()


def repair_presentation_content(
    presentation_plan_path: str,
    verification_report_path: str,
    output_dir: str = "output",
    model_name: str = "gpt-4o",
    api_key: Optional[str] = None,
    language: str = "zh"
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Convenient function for presentation content repair
    
    Args:
        presentation_plan_path: Path to presentation plan JSON
        verification_report_path: Path to verification report JSON
        output_dir: Directory to save repaired plan
        model_name: Language model to use
        api_key: OpenAI API key
        language: Output language
        
    Returns:
        Tuple of (repair_success, repair_report, repaired_plan_path)
    """
    agent = RepairAgent(
        model_name=model_name,
        api_key=api_key,
        language=language
    )
    
    return agent.repair_presentation_plan(
        presentation_plan_path=presentation_plan_path,
        verification_report_path=verification_report_path,
        output_dir=output_dir
    )
