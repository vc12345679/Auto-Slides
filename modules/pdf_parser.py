"""
PDF Parser Module: Responsible for parsing PDF files and extracting basic information
This module now calls lightweight extractor functionality for efficient content extraction
and uses LLM for presentation-oriented content enhancement
"""
import os
import json
import logging
import re
from typing import Dict, Any, Optional
from .lightweight_extractor import extract_lightweight_content

# Import LLM-related packages
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import enhancement prompts
from prompts import SUMMARIZE_TEXT_FOR_PRESENTATION_PROMPT, EXTRACT_TABLES_AND_EQUATIONS_PROMPT

def enhance_content_with_llm(lightweight_content: Dict[str, Any], model_name: str = "gpt-4o", api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhance content using LLM, reorganize and structure content from presentation perspective
    Now divided into two steps: 1) Extract tables and formulas 2) Summarize text content
    
    Args:
        lightweight_content: Basic content from lightweight extraction
        model_name: Language model name to use
        api_key: OpenAI API key
        
    Returns:
        Dict: Enhanced content
    """
    logger = logging.getLogger(__name__)
    
    if not OPENAI_AVAILABLE:
        logger.warning("Cannot import OpenAI packages, skipping LLM enhancement")
        return lightweight_content
    
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not provided, skipping LLM enhancement")
            return lightweight_content
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.2,
            openai_api_key=api_key
        )
        
        # Get full text
        full_text = lightweight_content.get("full_text", "")
        if not full_text:
            logger.warning("No full_text found, skipping LLM enhancement")
            return lightweight_content
        
        logger.info("Starting LLM content enhancement...")
        
        # Step 1: Extract tables and formulas
        logger.info("Step 1: Extracting tables and formulas...")
        tables_equations_result = _extract_tables_and_equations(llm, full_text)
        
        # Step 2: Summarize text content
        logger.info("Step 2: Summarizing presentation content...")
        presentation_summary = _summarize_for_presentation(llm, full_text)
        
        # Merge results
        enhanced_content = lightweight_content.copy()
        enhanced_content["enhanced_content"] = presentation_summary
        
        # If tables and formulas were successfully extracted, add to results
        if tables_equations_result:
            if "tables" in tables_equations_result:
                enhanced_content["enhanced_content"]["tables"] = tables_equations_result["tables"]
            if "equations" in tables_equations_result:
                enhanced_content["enhanced_content"]["equations"] = tables_equations_result["equations"]
        
        logger.info("LLM content enhancement completed")
        return enhanced_content
        
    except Exception as e:
        logger.error(f"Error during LLM enhancement: {str(e)}")
        return lightweight_content


def _extract_tables_and_equations(llm, full_text: str) -> Optional[Dict]:
    """
    Step 1: Specifically extract tables and formulas
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 导入特殊字符处理模块
        from modules.special_char_handler import preprocess_content_for_llm, postprocess_content_from_llm, validate_special_chars_in_output
        
        # 预处理文本以保护特殊字符
        protected_text = preprocess_content_for_llm(full_text)
        logger.debug("已对特殊字符进行预处理保护")
        
        prompt = ChatPromptTemplate.from_template(EXTRACT_TABLES_AND_EQUATIONS_PROMPT)
        chain = prompt | llm
        response = chain.invoke({"full_text": protected_text})
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 恢复特殊字符
        response_text = postprocess_content_from_llm(response_text)
        
        # Extract JSON part
        json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text.strip()
        
        # Parse JSON
        result = json.loads(json_str)
        
        # 验证特殊字符是否丢失
        if result.get('tables'):
            for table in result['tables']:
                markdown_content = table.get('markdown_content', '')
                lost_chars = validate_special_chars_in_output(full_text, markdown_content)
                if lost_chars:
                    logger.warning(f"表格 {table.get('id', 'unknown')} 中丢失特殊字符: {lost_chars}")
        
        logger.info(f"Successfully extracted {len(result.get('tables', []))} tables and {len(result.get('equations', []))} equations")
        return result
        
    except Exception as e:
        logger.warning(f"Error extracting tables and formulas: {str(e)}")
        return None


def _summarize_for_presentation(llm, full_text: str) -> Dict:
    """
    Step 2: Summarize presentation content
    """
    logger = logging.getLogger(__name__)
    
    try:
        prompt = ChatPromptTemplate.from_template(SUMMARIZE_TEXT_FOR_PRESENTATION_PROMPT)
        chain = prompt | llm
        response = chain.invoke({"full_text": full_text})
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON part
        json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text.strip()
        
        # Parse JSON
        result = json.loads(json_str)
        logger.info("Presentation content summarization completed")
        return result
        
    except Exception as e:
        logger.error(f"Error summarizing presentation content: {str(e)}")
        # Return basic structure to avoid complete failure
        return {
            "presentation_sections": {
                "background_context": "Content summarization failed",
                "problem_motivation": "Content summarization failed", 
                "solution_overview": "Content summarization failed",
                "technical_approach": "Content summarization failed",
                "evidence_proof": "Content summarization failed",
                "impact_significance": "Content summarization failed"
            },
            "key_narratives": {
                "field_importance": [],
                "problem_scenarios": [],
                "solution_benefits": [],
                "breakthrough_results": []
            }
        }

def extract_pdf_content(pdf_path, output_dir="output", cleanup_temp=False, enable_llm_enhancement=True, model_name="gpt-4o", api_key=None):
    """
    Extract PDF content (including text, images, metadata etc.) with optional LLM enhancement
    
    Args:
        pdf_path: PDF file path
        output_dir: Output directory
        cleanup_temp: Whether to clean up temporary files
        enable_llm_enhancement: Whether to enable LLM enhancement processing
        model_name: Language model name to use
        api_key: OpenAI API key
        
    Returns:
        tuple: (extracted content, content save file path)
    """
    logging.info(f"Starting PDF content extraction: {pdf_path}")
    
    # Call lightweight extractor module functionality
    lightweight_content, lightweight_content_path = extract_lightweight_content(pdf_path, output_dir, cleanup_temp)
    
    if not lightweight_content:
        logging.error("PDF content extraction failed")
        return None, None
    
    # If LLM enhancement is enabled, perform enhancement processing
    if enable_llm_enhancement:
        logging.info("Starting LLM enhancement processing...")
        enhanced_content = enhance_content_with_llm(lightweight_content, model_name, api_key)
        
        # Save enhanced content
        enhanced_content_path = lightweight_content_path.replace('.json', '_enhanced.json')
        try:
            with open(enhanced_content_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_content, f, ensure_ascii=False, indent=2)
            logging.info(f"Enhanced content saved to: {enhanced_content_path}")
            return enhanced_content, enhanced_content_path
        except Exception as e:
            logging.error(f"Error saving enhanced content: {str(e)}")
            # If save fails, return original content
            return lightweight_content, lightweight_content_path
    else:
        logging.info(f"PDF content extracted and saved to: {lightweight_content_path}")
        return lightweight_content, lightweight_content_path
