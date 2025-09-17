"""
Verification Agent: Bidirectional Content Verification and Hallucination Detection
Ensures consistency between original paper content and generated presentation slides
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

class VerificationAgent:
    """
    Verification Agent for bidirectional content validation and hallucination detection
    
    This agent performs comprehensive verification between original paper content
    and generated presentation slides to ensure accuracy and consistency.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,  # Low temperature for precise verification
        api_key: Optional[str] = None,
        language: str = "en"
    ):
        """
        Initialize the Verification Agent
        
        Args:
            model_name: Language model to use for verification
            temperature: Model temperature (low for precise verification)
            api_key: OpenAI API key
            language: Output language for verification reports
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # 尝试加载.env文件中的API密钥
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
            self.logger.info("Unified LLM interface initialized for verification with task-optimized parameters")
        else:
            self.llm_interface = None
            self.logger.warning("Unified LLM interface not available, using fallback methods")
        
        # Initialize language model (fallback)
        self._init_model()
        
        # Verification metrics
        self.verification_results = {}
    
    def _init_model(self):
        """Initialize the language model"""
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning("LangChain not available, verification functionality disabled")
            self.llm = None
            return
        
        if not self.api_key:
            self.logger.warning("No OpenAI API key provided, verification functionality disabled")
            self.llm = None
            return
        
        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key
            )
            self.logger.info(f"Verification Agent initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize language model: {str(e)}")
            self.llm = None
    
    def verify_presentation_plan(
        self,
        original_content_path: str,
        presentation_plan_path: str,
        output_dir: str = "output"
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Comprehensive verification of presentation plan against original content
        
        Args:
            original_content_path: Path to original extracted content JSON
            presentation_plan_path: Path to generated presentation plan JSON
            output_dir: Directory to save verification report
            
        Returns:
            Tuple of (verification_passed, verification_report, report_path)
        """
        if not self.llm:
            self.logger.error("Language model not available for verification")
            return False, {"error": "Language model not initialized"}, ""
        
        self.logger.info("Starting comprehensive presentation verification...")
        
        try:
            # Load content files
            original_content = self._load_json_file(original_content_path)
            presentation_plan = self._load_json_file(presentation_plan_path)
            
            if not original_content or not presentation_plan:
                return False, {"error": "Failed to load content files"}, ""
            
            # Perform verification checks
            verification_report = {
                "verification_timestamp": self._get_timestamp(),
                "original_content_path": original_content_path,
                "presentation_plan_path": presentation_plan_path,
                "verification_results": {}
            }
            
            # 1. Factual Consistency Check
            self.logger.info("Performing factual consistency verification...")
            consistency_result = self._verify_factual_consistency(original_content, presentation_plan)
            verification_report["verification_results"]["factual_consistency"] = consistency_result
            
            # 2. Hallucination Detection
            self.logger.info("Performing hallucination detection...")
            hallucination_result = self._detect_hallucinations(original_content, presentation_plan)
            verification_report["verification_results"]["hallucination_detection"] = hallucination_result
            
            # 3. Key Information Preservation
            self.logger.info("Verifying key information preservation...")
            preservation_result = self._verify_key_information_preservation(original_content, presentation_plan)
            verification_report["verification_results"]["key_information_preservation"] = preservation_result
            
            # 4. Quantitative Data Accuracy
            self.logger.info("Verifying quantitative data accuracy...")
            data_accuracy_result = self._verify_quantitative_data(original_content, presentation_plan)
            verification_report["verification_results"]["data_accuracy"] = data_accuracy_result
            
            # Generate overall assessment
            overall_assessment = self._generate_overall_assessment(verification_report["verification_results"])
            verification_report["overall_assessment"] = overall_assessment
            
            # Save verification report
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "verification_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(verification_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Verification report saved to: {report_path}")
            
            # Determine if verification passed
            verification_passed = overall_assessment["passed"]
            
            return verification_passed, verification_report, report_path
            
        except Exception as e:
            self.logger.error(f"Verification process failed: {str(e)}")
            return False, {"error": str(e)}, ""
    
    def _verify_factual_consistency(self, original_content: Dict, presentation_plan: Dict) -> Dict[str, Any]:
        """Verify factual consistency between original content and presentation"""
        
        # Extract original text for comparison
        original_text = self._extract_original_text(original_content)
        
        # Extract presentation content
        presentation_content = self._extract_presentation_content(presentation_plan)
        
        # Create verification prompt
        verification_prompt = self._create_factual_consistency_prompt(original_text, presentation_content)
        
        try:
            # Use task-optimized parameters for fact checking
            if self.llm_interface:
                result = self.llm_interface.call_for_fact_checking(
                    "", verification_prompt, json_mode=False
                )
                result = self._parse_verification_response(result)
            else:
                response = self.llm.invoke([HumanMessage(content=verification_prompt)])
                result = self._parse_verification_response(response.content)
            
            return {
                "status": "completed",
                "consistency_score": result.get("consistency_score", 0),
                "inconsistencies": result.get("inconsistencies", []),
                "detailed_analysis": result.get("detailed_analysis", ""),
                "recommendations": result.get("recommendations", [])
            }
            
        except Exception as e:
            self.logger.error(f"Factual consistency verification failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _detect_hallucinations(self, original_content: Dict, presentation_plan: Dict) -> Dict[str, Any]:
        """Detect potential hallucinations in presentation content with pre-validation"""
        
        # Extract content for hallucination detection
        original_text = self._extract_original_text(original_content)
        presentation_content = self._extract_presentation_content(presentation_plan)
        
        # PRE-VALIDATE numerical claims to prevent false positives
        pre_validation = self._pre_validate_numerical_claims(original_text, presentation_content)
        self.logger.info(f"Pre-validation found {len(pre_validation['validated_comparisons'])} comparisons")
        
        # SEMANTIC VALIDATION: Check context and meaning of comparisons
        semantic_validations = []
        for comp in pre_validation['validated_comparisons']:
            if comp['both_exist']:
                semantic_result = self._validate_semantic_context(original_text, comp['comparison'])
                semantic_validations.append(semantic_result)
                
                # Log semantic issues
                if not semantic_result['semantic_valid']:
                    self.logger.warning(f"Semantic issues found in '{comp['comparison']}': {semantic_result['issues']}")
        
        pre_validation['semantic_validations'] = semantic_validations
        
        # Create hallucination detection prompt with pre-validation context
        detection_prompt = self._create_hallucination_detection_prompt_with_prevalidation(
            original_text, presentation_content, pre_validation
        )
        
        try:
            # Use ultra-precise parameters for hallucination detection
            if self.llm_interface:
                result = self.llm_interface.call_for_hallucination_detection(
                    "", detection_prompt, json_mode=False
                )
                result = self._parse_verification_response(result)
            else:
                response = self.llm.invoke([HumanMessage(content=detection_prompt)])
                result = self._parse_verification_response(response.content)
            
            return {
                "status": "completed",
                "hallucination_detected": result.get("hallucination_detected", False),
                "potential_hallucinations": result.get("potential_hallucinations", []),
                "confidence_score": result.get("confidence_score", 0),
                "detailed_analysis": result.get("detailed_analysis", ""),
                "severity_level": result.get("severity_level", "low")
            }
            
        except Exception as e:
            self.logger.error(f"Hallucination detection failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _verify_key_information_preservation(self, original_content: Dict, presentation_plan: Dict) -> Dict[str, Any]:
        """Verify that key information from original content is preserved"""
        
        # Extract enhanced content if available
        enhanced_content = original_content.get("enhanced_content", {})
        
        # Extract key information from original content
        key_info = {
            "contributions": enhanced_content.get("presentation_sections", {}).get("solution_overview", ""),
            "methodology": enhanced_content.get("presentation_sections", {}).get("technical_approach", ""),
            "results": enhanced_content.get("presentation_sections", {}).get("evidence_proof", ""),
            "conclusions": enhanced_content.get("presentation_sections", {}).get("impact_significance", "")
        }
        
        # Extract presentation slides
        slides_content = []
        for slide in presentation_plan.get("slides_plan", []):
            slides_content.append({
                "title": slide.get("title", ""),
                "content": slide.get("content", [])
            })
        
        # Create key information preservation prompt
        preservation_prompt = self._create_key_info_preservation_prompt(key_info, slides_content)
        
        try:
            # Use verification parameters for key information preservation
            if self.llm_interface:
                result = self.llm_interface.call_for_verification(
                    "", preservation_prompt, json_mode=False
                )
                result = self._parse_verification_response(result)
            else:
                response = self.llm.invoke([HumanMessage(content=preservation_prompt)])
                result = self._parse_verification_response(response.content)
            
            return {
                "status": "completed",
                "preservation_score": result.get("preservation_score", 0),
                "missing_key_info": result.get("missing_key_info", []),
                "well_preserved_info": result.get("well_preserved_info", []),
                "detailed_analysis": result.get("detailed_analysis", ""),
                "improvement_suggestions": result.get("improvement_suggestions", [])
            }
            
        except Exception as e:
            self.logger.error(f"Key information preservation verification failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _verify_quantitative_data(self, original_content: Dict, presentation_plan: Dict) -> Dict[str, Any]:
        """Verify accuracy of quantitative data and statistics"""
        
        # Extract tables and quantitative data from original content
        original_tables = original_content.get("enhanced_content", {}).get("tables", [])
        
        # Extract quantitative claims from presentation slides
        slides_with_data = []
        for slide in presentation_plan.get("slides_plan", []):
            if slide.get("includes_table") or any("%" in content or any(char.isdigit() for char in content) for content in slide.get("content", [])):
                slides_with_data.append(slide)
        
        if not original_tables and not slides_with_data:
            return {
                "status": "completed",
                "data_accuracy_score": 100,
                "message": "No quantitative data to verify"
            }
        
        # Create quantitative data verification prompt
        data_verification_prompt = self._create_data_verification_prompt(original_tables, slides_with_data)
        
        try:
            # Use verification parameters for data accuracy checking
            if self.llm_interface:
                result = self.llm_interface.call_for_verification(
                    "", data_verification_prompt, json_mode=False
                )
                result = self._parse_verification_response(result)
            else:
                response = self.llm.invoke([HumanMessage(content=data_verification_prompt)])
                result = self._parse_verification_response(response.content)
            
            return {
                "status": "completed",
                "data_accuracy_score": result.get("data_accuracy_score", 0),
                "data_inconsistencies": result.get("data_inconsistencies", []),
                "verified_data_points": result.get("verified_data_points", []),
                "detailed_analysis": result.get("detailed_analysis", ""),
                "critical_errors": result.get("critical_errors", [])
            }
            
        except Exception as e:
            self.logger.error(f"Quantitative data verification failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_overall_assessment(self, verification_results: Dict) -> Dict[str, Any]:
        """Generate overall assessment based on all verification results"""
        
        # Calculate overall scores
        scores = []
        critical_issues = []
        warnings = []
        
        # Factual consistency
        if verification_results.get("factual_consistency", {}).get("status") == "completed":
            consistency_score = verification_results["factual_consistency"].get("consistency_score", 0)
            scores.append(consistency_score)
            if consistency_score < 70:
                critical_issues.append("Low factual consistency score")
            elif consistency_score < 85:
                warnings.append("Moderate factual consistency concerns")
        
        # Hallucination detection
        if verification_results.get("hallucination_detection", {}).get("status") == "completed":
            if verification_results["hallucination_detection"].get("hallucination_detected"):
                severity = verification_results["hallucination_detection"].get("severity_level", "low")
                if severity in ["high", "critical"]:
                    critical_issues.append("High-severity hallucinations detected")
                else:
                    warnings.append("Potential hallucinations detected")
        
        # Key information preservation
        if verification_results.get("key_information_preservation", {}).get("status") == "completed":
            preservation_score = verification_results["key_information_preservation"].get("preservation_score", 0)
            scores.append(preservation_score)
            if preservation_score < 70:
                critical_issues.append("Poor key information preservation")
            elif preservation_score < 85:
                warnings.append("Some key information may be missing")
        
        # Data accuracy
        if verification_results.get("data_accuracy", {}).get("status") == "completed":
            data_score = verification_results["data_accuracy"].get("data_accuracy_score", 0)
            scores.append(data_score)
            if data_score < 80:
                critical_issues.append("Quantitative data accuracy issues")
            elif data_score < 95:
                warnings.append("Minor data accuracy concerns")
        
        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Determine pass/fail
        verification_passed = len(critical_issues) == 0 and overall_score >= 75
        
        return {
            "passed": verification_passed,
            "overall_score": round(overall_score, 2),
            "critical_issues": critical_issues,
            "warnings": warnings,
            "recommendation": "APPROVED" if verification_passed else "NEEDS_REVISION",
            "summary": self._generate_assessment_summary(overall_score, critical_issues, warnings)
        }
    
    def _load_json_file(self, file_path: str) -> Optional[Dict]:
        """Load and parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {file_path}: {str(e)}")
            return None
    
    def _extract_original_text(self, original_content: Dict) -> str:
        """Extract text content from original content structure"""
        text_parts = []
        
        # Prioritize enhanced content sections first (most relevant)
        enhanced_content = original_content.get("enhanced_content", {})
        if "presentation_sections" in enhanced_content:
            for section, content in enhanced_content["presentation_sections"].items():
                text_parts.append(f"[{section.upper()}] {content}")
        
        # Add table content if available
        if "tables" in enhanced_content:
            for table in enhanced_content["tables"]:
                table_text = f"TABLE: {table.get('title', '')} - {table.get('markdown_content', '')}"
                text_parts.append(table_text)
        
        # Add equations if available
        if "equations" in enhanced_content:
            for eq in enhanced_content["equations"]:
                eq_text = f"EQUATION: {eq.get('description', '')} - {eq.get('latex', '')}"
                text_parts.append(eq_text)
        
        # Get full text as additional context (but lower priority)
        if "full_text" in original_content:
            # Truncate full text to avoid API limits while keeping enhanced content complete
            full_text = original_content["full_text"][:5000]
            text_parts.append(f"[FULL_TEXT_EXCERPT] {full_text}")
        
        return " ".join(text_parts)
    
    def _extract_presentation_content(self, presentation_plan: Dict) -> str:
        """Extract content from presentation plan"""
        content_parts = []
        
        for slide in presentation_plan.get("slides_plan", []):
            content_parts.append(slide.get("title", ""))
            content_parts.extend(slide.get("content", []))
        
        return " ".join(content_parts)
    
    def _pre_validate_numerical_claims(self, original_text: str, presentation_content: str) -> Dict[str, Any]:
        """
        Pre-validate numerical claims before LLM analysis to prevent false positives
        
        Args:
            original_text: Original paper content
            presentation_content: Presentation content to verify
            
        Returns:
            Dict containing pre-validation results
        """
        import re
        
        validation_results = {
            "validated_numbers": [],
            "validated_comparisons": [],
            "potential_issues": []
        }
        
        # Extract numerical comparisons from presentation (e.g., "from X to Y")
        comparison_pattern = r'(?:from|reduction.*?from|\(from)\s*([0-9.]+)\s*(?:to|steps?\s+to)\s*([0-9.]+)'
        comparisons = re.findall(comparison_pattern, presentation_content, re.IGNORECASE)
        
        for from_val, to_val in comparisons:
            # Clean values (remove trailing punctuation)
            from_val = from_val.rstrip('.,;')
            to_val = to_val.rstrip('.,;')
            
            # Check if both numbers exist in original text
            from_exists = from_val in original_text
            to_exists = to_val in original_text
            
            # Also check the complete comparison phrase
            comparison_phrases = [
                f"from {from_val} to {to_val}",
                f"({from_val} to {to_val})",
                f"{from_val} to {to_val}"
            ]
            comparison_exists = any(phrase in original_text for phrase in comparison_phrases)
            
            validation_results["validated_comparisons"].append({
                "comparison": f"{from_val} to {to_val}",
                "from_value_exists": from_exists,
                "to_value_exists": to_val in original_text,
                "both_exist": from_exists and to_exists,
                "comparison_phrase_exists": comparison_exists
            })
        
        # Extract standalone numbers from presentation
        number_pattern = r'\b([0-9]+\.?[0-9]*%?k?)\b'
        presentation_numbers = re.findall(number_pattern, presentation_content)
        
        for number in set(presentation_numbers):  # Remove duplicates
            exists_in_original = number in original_text
            validation_results["validated_numbers"].append({
                "number": number,
                "exists_in_original": exists_in_original
            })
        
        return validation_results
    
    def _validate_semantic_context(self, original_text: str, comparison: str, context_window: int = 200) -> Dict[str, Any]:
        """
        Validate that numerical comparisons have correct semantic context
        
        Args:
            original_text: Original paper content
            comparison: Numerical comparison like "9.1 to 5.7"
            context_window: Characters around the comparison to check
            
        Returns:
            Dict containing semantic validation results
        """
        semantic_validation = {
            "comparison": comparison,
            "semantic_valid": False,
            "context_found": False,
            "original_context": "",
            "improvement_direction_correct": None,
            "issues": []
        }
        
        # Find the comparison in original text with context
        import re
        
        # Look for the exact comparison pattern
        pattern = re.escape(comparison).replace(r'\ to\ ', r'\s+to\s+')
        match = re.search(pattern, original_text, re.IGNORECASE)
        
        if match:
            start = max(0, match.start() - context_window)
            end = min(len(original_text), match.end() + context_window)
            context = original_text[start:end]
            
            semantic_validation["context_found"] = True
            semantic_validation["original_context"] = context
            
            # Check for improvement indicators
            improvement_words = ['reduction', 'reduced', 'decrease', 'improved', 'better', 'enhancement']
            degradation_words = ['increase', 'worse', 'degradation', 'higher']
            
            context_lower = context.lower()
            
            # Extract the numbers
            numbers = comparison.split(' to ')
            if len(numbers) == 2:
                try:
                    from_val = float(numbers[0])
                    to_val = float(numbers[1])
                    
                    # Determine if this should be an improvement (lower is better) or performance gain (higher is better)
                    is_improvement = any(word in context_lower for word in improvement_words)
                    is_degradation = any(word in context_lower for word in degradation_words)
                    
                    if is_improvement:
                        # For improvements, lower should be better
                        if to_val < from_val:
                            semantic_validation["improvement_direction_correct"] = True
                        else:
                            semantic_validation["improvement_direction_correct"] = False
                            semantic_validation["issues"].append("Claims improvement but numbers suggest degradation")
                    
                    elif is_degradation:
                        # For degradation mentions, higher values might be expected
                        if to_val > from_val:
                            semantic_validation["improvement_direction_correct"] = True
                        else:
                            semantic_validation["improvement_direction_correct"] = False
                            semantic_validation["issues"].append("Claims degradation but numbers suggest improvement")
                    
                    # Check for metric consistency (steps, time, etc. should generally decrease for improvements)
                    metric_indicators = ['steps', 'time', 'seconds', 'duration', 'cost']
                    if any(metric in context_lower for metric in metric_indicators) and is_improvement:
                        if to_val >= from_val:
                            semantic_validation["issues"].append("Metric should decrease for improvement")
                    
                    semantic_validation["semantic_valid"] = len(semantic_validation["issues"]) == 0
                    
                except ValueError:
                    semantic_validation["issues"].append("Could not parse numerical values")
        else:
            semantic_validation["issues"].append("Comparison not found in original text")
        
        return semantic_validation
    
    def _create_factual_consistency_prompt(self, original_text: str, presentation_content: str) -> str:
        """Create prompt for factual consistency verification with conservative logic"""
        return f"""
You are an expert fact-checker specializing in academic content verification. Your role is to identify ONLY genuine factual inconsistencies between the original paper and presentation.

**STRICT VALIDATION PROTOCOL:**
1. **Number Verification**: For every numerical claim in the presentation:
   - Search the original text for EXACT numbers
   - If both numbers in a comparison (e.g., "X improved from A to B") exist in original, mark as CONSISTENT
   - Only flag as inconsistent if numbers are genuinely absent or contradictory

2. **Experimental Claims**: For performance/result claims:
   - Verify against original experimental sections and tables
   - Accept valid interpretations and reasonable summaries
   - Only flag clear contradictions or unsupported claims

3. **Conservative Scoring**: 
   - Start with assumption that content is consistent
   - Only reduce score for verified, significant inconsistencies
   - Minor presentation differences should NOT affect consistency score

4. **Evidence Requirement**: Every flagged inconsistency must include:
   - Exact quote from original showing contradiction
   - Specific explanation of why it's inconsistent
   - Severity level (only flag medium/high severity issues)

**Original Paper Content:**
{original_text[:12000]}

**Generated Presentation Content:**
{presentation_content[:6000]}

Apply conservative fact-checking. Remember: Accurate data should receive high consistency scores, even if presentation format differs from original.

{{
  "consistency_score": <score from 0-100>,
  "inconsistencies": [
    {{
      "type": "factual_error|misrepresentation|omission",
      "description": "Description of the inconsistency",
      "severity": "low|medium|high|critical",
      "original_content": "Relevant original content",
      "presentation_content": "Problematic presentation content"
    }}
  ],
  "detailed_analysis": "Overall analysis of consistency",
  "recommendations": ["List of specific recommendations for improvement"]
}}

Focus on:
1. Factual accuracy of claims and statements
2. Correct representation of research findings
3. Accurate citation of numbers and statistics
4. Proper context preservation
"""
    
    def _create_hallucination_detection_prompt_with_prevalidation(
        self, original_text: str, presentation_content: str, pre_validation: Dict[str, Any]
    ) -> str:
        """Create hallucination detection prompt with pre-validation context"""
        
        # Build pre-validation summary
        pre_val_summary = "**PRE-VALIDATION RESULTS:**\n"
        
        validated_comparisons = pre_validation.get("validated_comparisons", [])
        if validated_comparisons:
            pre_val_summary += "Verified numerical comparisons:\n"
            for comp in validated_comparisons:
                if comp["both_exist"]:
                    pre_val_summary += f"✅ '{comp['comparison']}' - BOTH numbers found in original\n"
                else:
                    pre_val_summary += f"❌ '{comp['comparison']}' - Missing: from={comp['from_value_exists']}, to={comp['to_value_exists']}\n"
        
        # Add semantic validation results
        semantic_validations = pre_validation.get("semantic_validations", [])
        if semantic_validations:
            pre_val_summary += "\n**SEMANTIC VALIDATION:**\n"
            for sem_val in semantic_validations:
                comparison = sem_val["comparison"]
                if sem_val["semantic_valid"]:
                    pre_val_summary += f"✅ '{comparison}' - Semantically correct in context\n"
                else:
                    issues = "; ".join(sem_val["issues"])
                    pre_val_summary += f"⚠️ '{comparison}' - Semantic issues: {issues}\n"
        
        validated_numbers = pre_validation.get("validated_numbers", [])
        verified_numbers = [n["number"] for n in validated_numbers if n["exists_in_original"]]
        if verified_numbers:
            pre_val_summary += f"✅ Verified standalone numbers: {', '.join(verified_numbers)}\n"
        
        return f"""
You are an expert hallucination detector. You must identify ONLY clearly fabricated content that contradicts the original paper.

{pre_val_summary}

**CRITICAL INSTRUCTION**: The pre-validation above shows numbers that were programmatically verified to exist in the original text. DO NOT flag these as hallucinations unless they are used in completely wrong context.

**VALIDATION RULES:**
1. If a numerical comparison was pre-validated (✅), it should NOT be flagged as hallucination
2. Only flag content that is clearly fabricated or contradicts the original
3. When in doubt, do NOT flag as hallucination
4. **IMPORTANT**: Claims like "highest average score" or "lowest hallucination rate" are VALID if the actual data supports them - verify table data before flagging
5. **TABLE DATA PRIORITY**: Always defer to actual numerical data in tables when evaluating performance claims

**Original Paper Content:**
{original_text[:12000]}

**Generated Presentation Content:**
{presentation_content[:6000]}

Analyze carefully, respecting the pre-validation results. Focus only on genuine fabrications or contradictions.

{{
  "hallucination_detected": <true/false>,
  "potential_hallucinations": [
    {{
      "content": "Potentially hallucinated content",
      "type": "fabricated_data|unsupported_claim|invented_reference|exaggerated_claim",
      "severity": "low|medium|high|critical",
      "explanation": "Why this might be a hallucination",
      "evidence_check": "What evidence should exist but doesn't"
    }}
  ],
  "confidence_score": <score from 0-100>,
  "detailed_analysis": "Detailed analysis of potential hallucinations",
  "severity_level": "low|medium|high|critical"
}}

Look specifically for:
1. Claims not supported by the original paper
2. Fabricated statistics or numbers
3. Invented references or citations
4. Exaggerated or overstated findings
5. Technical details not present in original

**BEFORE FLAGGING PERFORMANCE CLAIMS:**
- For claims like "highest score" or "lowest rate", MANUALLY verify against table data
- If table shows VTI: 2.90 and others are lower (2.69, 2.60, 1.99), then "highest score" is FACTUALLY CORRECT
- If table shows VTI: 0.51 and others are higher (0.56, 0.58, 0.62), then "lowest rate" is FACTUALLY CORRECT
- Only flag if claim contradicts the actual numbers in tables
"""
    
    def _create_key_info_preservation_prompt(self, key_info: Dict, slides_content: List) -> str:
        """Create prompt for key information preservation verification"""
        return f"""
You are an academic content analyst. Please evaluate how well the key information from the research paper has been preserved in the presentation slides.

**Key Information from Original Paper:**
{json.dumps(key_info, indent=2)}

**Presentation Slides Content:**
{json.dumps(slides_content, indent=2)}

Please analyze information preservation and provide results in JSON format:

{{
  "preservation_score": <score from 0-100>,
  "missing_key_info": [
    {{
      "category": "contributions|methodology|results|conclusions",
      "missing_content": "Description of missing information",
      "importance": "low|medium|high|critical"
    }}
  ],
  "well_preserved_info": [
    {{
      "category": "contributions|methodology|results|conclusions", 
      "preserved_content": "Description of well-preserved information"
    }}
  ],
  "detailed_analysis": "Analysis of information preservation quality",
  "improvement_suggestions": ["Specific suggestions for better preservation"]
}}

Evaluate:
1. Whether main contributions are clearly presented
2. If methodology is adequately explained
3. Whether key results are included
4. If conclusions are properly conveyed
5. Overall completeness of information transfer
"""
    
    def _create_data_verification_prompt(self, original_tables: List, slides_with_data: List) -> str:
        """Create prompt for quantitative data verification"""
        return f"""
You are a data accuracy specialist. Please verify the accuracy of quantitative data, statistics, and numerical claims in the presentation slides against the original research data.

**Original Tables and Data:**
{json.dumps(original_tables, indent=2)[:4000]}

**Presentation Slides with Quantitative Data:**
{json.dumps(slides_with_data, indent=2)[:4000]}

Please verify data accuracy and provide analysis in JSON format:

{{
  "data_accuracy_score": <score from 0-100>,
  "data_inconsistencies": [
    {{
      "type": "incorrect_number|misplaced_decimal|wrong_unit|calculation_error",
      "original_value": "Value from original data",
      "presentation_value": "Value in presentation",
      "location": "Where the error occurs",
      "severity": "low|medium|high|critical"
    }}
  ],
  "verified_data_points": [
    {{
      "data_point": "Correctly represented data",
      "verification_status": "accurate"
    }}
  ],
  "detailed_analysis": "Overall analysis of data accuracy",
  "critical_errors": ["List of any critical data errors"]
}}

Check specifically for:
1. Correct numerical values and statistics
2. Proper units and decimal places
3. Accurate percentages and ratios
4. Correct table data representation
5. Mathematical consistency
"""
    
    def _parse_verification_response(self, response_content: str) -> Dict[str, Any]:
        """Parse JSON response from verification prompts"""
        try:
            # Extract JSON from response if wrapped in markdown
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = response_content.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse verification response: {str(e)}")
            return {"error": "Failed to parse response", "raw_response": response_content}
    
    def _generate_assessment_summary(self, overall_score: float, critical_issues: List, warnings: List) -> str:
        """Generate human-readable assessment summary"""
        if overall_score >= 90:
            quality = "Excellent"
        elif overall_score >= 80:
            quality = "Good"  
        elif overall_score >= 70:
            quality = "Acceptable"
        elif overall_score >= 60:
            quality = "Needs Improvement"
        else:
            quality = "Poor"
        
        summary = f"Overall Quality: {quality} (Score: {overall_score:.1f}/100)"
        
        if critical_issues:
            summary += f"\nCritical Issues: {len(critical_issues)} found"
        if warnings:
            summary += f"\nWarnings: {len(warnings)} identified"
        
        if not critical_issues and not warnings:
            summary += "\nNo significant issues detected."
        
        return summary
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reporting"""
        from datetime import datetime
        return datetime.now().isoformat()


def verify_presentation_content(
    original_content_path: str,
    presentation_plan_path: str,
    output_dir: str = "output",
    model_name: str = "gpt-4o",
    api_key: Optional[str] = None,
    language: str = "en"
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Convenient function for presentation content verification
    
    Args:
        original_content_path: Path to original extracted content JSON
        presentation_plan_path: Path to generated presentation plan JSON
        output_dir: Directory to save verification report
        model_name: Language model to use
        api_key: OpenAI API key
        language: Output language
        
    Returns:
        Tuple of (verification_passed, verification_report, report_path)
    """
    agent = VerificationAgent(
        model_name=model_name,
        api_key=api_key,
        language=language
    )
    
    return agent.verify_presentation_plan(
        original_content_path=original_content_path,
        presentation_plan_path=presentation_plan_path,
        output_dir=output_dir
    )
