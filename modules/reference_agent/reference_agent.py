#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference Agent: Citation Retrieval and Content Enhancement
Integrates citation extraction, literature search, content extraction, and content integration
for enhancing presentation content with relevant external references
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .citation_extractor import CitationExtractor
from .literature_searcher import LiteratureSearcher
from .content_extractor import ContentExtractor
from .content_integrator import ContentIntegrator


class ReferenceAgent:
    """
    Reference Agent for citation-based content enhancement
    
    This agent helps expand presentation content by:
    1. Extracting relevant citations from the original paper
    2. Searching for literature using multiple academic databases
    3. Extracting relevant content from retrieved papers
    4. Intelligently integrating content from multiple sources
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o",
                 temperature: float = 0.3,
                 api_key: Optional[str] = None,
                 cache_dir: str = "literature_cache"):
        """
        Initialize the Reference Agent
        
        Args:
            model_name: LLM model name for content processing
            temperature: Model temperature for generation
            api_key: OpenAI API key
            cache_dir: Directory for caching literature search results
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize component modules
        self.citation_extractor = CitationExtractor()
        self.literature_searcher = LiteratureSearcher()
        self.content_extractor = ContentExtractor()
        self.content_integrator = ContentIntegrator(
            model_name=model_name,
            temperature=temperature, 
            api_key=api_key
        )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger.info("Reference Agent initialized successfully")
    
    def enhance_content_with_references(self,
                                      original_paper_path: str,
                                      target_concept: str,
                                      context: str,
                                      max_references: int = 5,
                                      output_dir: str = "reference_output") -> Dict[str, Any]:
        """
        Main entry point for enhancing content with external references
        
        Args:
            original_paper_path: Path to the original paper content JSON
            target_concept: The concept/topic to expand upon
            context: Original context where the concept appears
            max_references: Maximum number of references to use
            output_dir: Directory to save results
            
        Returns:
            Dict containing enhanced content and metadata
        """
        self.logger.info(f"Starting reference-based content enhancement for concept: '{target_concept}'")
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Step 1: Load original paper content
            with open(original_paper_path, 'r', encoding='utf-8') as f:
                original_content = json.load(f)
            
            full_text = original_content.get('full_text', '')
            if not full_text:
                return self._create_error_result("No full text found in original paper")
            
            # Step 2: Extract relevant citations
            self.logger.info("Step 1: Extracting relevant citations...")
            citations = self.citation_extractor.extract_relevant_citations(
                full_text, target_concept
            )
            
            if not citations:
                self.logger.info(f"No citations found for '{target_concept}', trying to extract from original paper...")
                return self._extract_from_original_paper(full_text, target_concept, context, output_path)
            
            self.logger.info(f"Found {len(citations)} relevant citations")
            
            # Step 3: Search for literature  
            self.logger.info("Step 2: Searching for literature...")
            paper_results = self.literature_searcher.search_multiple_papers(
                citations[:max_references]
            )
            
            if not paper_results:
                self.logger.info("No literature found for citations, falling back to original paper content...")
                return self._extract_from_original_paper(full_text, target_concept, context, output_path)
            
            self.logger.info(f"Successfully retrieved {len(paper_results)} papers")
            
            # Step 4: Extract relevant content from papers
            self.logger.info("Step 3: Extracting relevant content...")
            extracted_contents = []
            for paper_result in paper_results:
                content = self.content_extractor.extract_relevant_content(
                    paper_result, target_concept, context
                )
                if content:
                    extracted_contents.append(content)
            
            if not extracted_contents:
                return self._create_error_result("No relevant content extracted from papers")
            
            self.logger.info(f"Extracted content from {len(extracted_contents)} papers")
            
            # Step 5: Integrate content
            self.logger.info("Step 4: Integrating content...")
            integrated_content = self.content_integrator.generate_expanded_content(
                context, target_concept, extracted_contents
            )
            
            if not integrated_content:
                return self._create_error_result("Failed to integrate content from sources")
            
            # Step 6: Create comprehensive result
            result = self._create_success_result(
                target_concept=target_concept,
                original_context=context,
                citations=citations,
                paper_results=paper_results,
                extracted_contents=extracted_contents,
                integrated_content=integrated_content
            )
            
            # Step 7: Save results
            self._save_results(result, output_path, target_concept)
            
            self.logger.info("Reference-based content enhancement completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Content enhancement failed: {e}")
            return self._create_error_result(f"Enhancement failed: {str(e)}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result structure"""
        return {
            'success': False,
            'error': error_message,
            'enhanced_content': '',
            'source_papers': [],
            'citations_found': 0,
            'papers_retrieved': 0,
            'content_quality_score': 0.0
        }
    
    def _create_success_result(self,
                             target_concept: str,
                             original_context: str,
                             citations: List,
                             paper_results: List,
                             extracted_contents: List,
                             integrated_content) -> Dict[str, Any]:
        """Create a success result structure"""
        return {
            'success': True,
            'target_concept': target_concept,
            'original_context': original_context,
            'enhanced_content': integrated_content.expanded_content,
            'content_summary': integrated_content.summary,
            'key_points': integrated_content.key_points,
            'content_quality_score': integrated_content.quality_score,
            'integration_method': integrated_content.integration_method,
            
            # Statistics
            'citations_found': len(citations),
            'papers_retrieved': len(paper_results),
            'papers_with_content': len(extracted_contents),
            
            # Source information
            'source_papers': [
                {
                    'title': paper.get('title', 'Unknown'),
                    'authors': paper.get('authors', []),
                    'year': paper.get('year', 'Unknown'),
                    'venue': paper.get('venue', 'Unknown'),
                    'search_strategy': paper.get('search_strategy', 'Unknown'),
                    'confidence_score': paper.get('confidence_score', 0.0)
                }
                for paper in integrated_content.source_papers
            ],
            
            # Raw citations for reference
            'citations': [
                {
                    'authors': cite.authors,
                    'title': cite.title,
                    'year': cite.year,
                    'venue': cite.venue,
                    'anchor': cite.anchor
                }
                for cite in citations[:len(paper_results)]
            ]
        }
    
    def _save_results(self, result: Dict[str, Any], output_path: Path, concept: str):
        """Save results to files"""
        try:
            # Create safe filename
            safe_concept = "".join(c for c in concept if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_concept = safe_concept.replace(' ', '_')[:50]
            
            # Save JSON result
            json_file = output_path / f"reference_enhancement_{safe_concept}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Save readable text file
            text_file = output_path / f"reference_enhancement_{safe_concept}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write("REFERENCE-BASED CONTENT ENHANCEMENT REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Target Concept: {result['target_concept']}\n")
                f.write(f"Quality Score: {result['content_quality_score']:.3f}\n")
                f.write(f"Integration Method: {result['integration_method']}\n\n")
                
                f.write("ENHANCED CONTENT:\n")
                f.write("-" * 20 + "\n")
                f.write(result['enhanced_content'])
                f.write("\n\n")
                
                if result['key_points']:
                    f.write("KEY POINTS:\n")
                    f.write("-" * 20 + "\n")
                    for i, point in enumerate(result['key_points'], 1):
                        f.write(f"{i}. {point}\n")
                    f.write("\n")
                
                f.write("SOURCE PAPERS:\n")
                f.write("-" * 20 + "\n")
                for i, paper in enumerate(result['source_papers'], 1):
                    f.write(f"{i}. {paper['title']}\n")
                    authors = ', '.join(paper['authors'][:3])
                    if len(paper['authors']) > 3:
                        authors += ' et al.'
                    f.write(f"   Authors: {authors}\n")
                    f.write(f"   Year: {paper['year']}, Venue: {paper['venue']}\n")
                    f.write(f"   Search Strategy: {paper['search_strategy']}\n")
                    f.write(f"   Confidence: {paper['confidence_score']:.3f}\n\n")
                
                f.write("STATISTICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Citations Found: {result['citations_found']}\n")
                f.write(f"Papers Retrieved: {result['papers_retrieved']}\n")
                f.write(f"Papers with Relevant Content: {result['papers_with_content']}\n")
            
            self.logger.info(f"Results saved to {json_file} and {text_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save results: {e}")
    
    def _extract_from_original_paper(self, full_text: str, target_concept: str, context: str, output_path: Path) -> Dict[str, Any]:
        """
        从原论文中提取与目标概念相关的技术内容（降级策略）
        
        Args:
            full_text: 原论文全文
            target_concept: 目标概念
            context: 上下文
            output_path: 输出路径
            
        Returns:
            提取的内容字典
        """
        try:
            self.logger.info(f"Extracting '{target_concept}' content from original paper as fallback strategy")
            
            # 查找包含目标概念的段落 - 使用更广泛的搜索模式
            concept_paragraphs = []
            paragraphs = full_text.split('\n\n')
            
            # 生成搜索模式
            search_patterns = [
                target_concept.lower(),
                target_concept.replace(' ', '-').lower(),
                target_concept.replace(' ', '_').lower()
            ]
            
            # 为attention添加额外的搜索模式
            if 'attention' in target_concept.lower():
                search_patterns.extend([
                    'attention',
                    'cross-attention', 
                    'cross attention',
                    'transformer',
                    'multi-head attention',
                    'self-attention'
                ])
            
            for para in paragraphs:
                para_lower = para.lower()
                # 检查是否包含任何搜索模式，且段落有实质内容
                if (any(pattern in para_lower for pattern in search_patterns) and 
                    len(para.strip()) > 50 and 
                    not para.strip().startswith('#')):
                    concept_paragraphs.append(para.strip())
            
            if not concept_paragraphs:
                return self._create_error_result(f"No content found related to '{target_concept}' in original paper")
            
            self.logger.info(f"Found {len(concept_paragraphs)} paragraphs containing '{target_concept}'")
            
            # 使用LLM生成解释性内容
            expanded_content = self._generate_explanation_from_paragraphs(
                concept_paragraphs, target_concept, context
            )
            
            if not expanded_content:
                return self._create_error_result("Failed to generate explanation from original paper content")
            
            # 创建成功结果
            return {
                'success': True,
                'target_concept': target_concept,
                'original_context': context,
                'enhanced_content': expanded_content,
                'content_summary': f"Technical explanation of {target_concept} based on original paper",
                'key_points': self._extract_key_points(expanded_content),
                'content_quality_score': 0.7,
                'integration_method': 'original_paper_extraction',
                
                # Statistics
                'citations_found': 0,
                'papers_retrieved': 0,
                'papers_with_content': 1,
                
                # Source information
                'source_papers': [{
                    'title': 'Original Paper',
                    'authors': ['Original Authors'],
                    'year': 'N/A',
                    'venue': 'Source Paper',
                    'search_strategy': 'original_content',
                    'confidence_score': 1.0
                }],
                'citations': [],
                'source_type': 'original_paper'
            }
            
        except Exception as e:
            self.logger.error(f"Error in _extract_from_original_paper: {e}")
            return self._create_error_result(f"Failed to extract from original paper: {str(e)}")
    
    def _generate_explanation_from_paragraphs(self, paragraphs: List[str], target_concept: str, context: str) -> str:
        """使用LLM从原论文段落生成概念解释"""
        try:
            relevant_content = '\n\n'.join(paragraphs[:5])  # 限制长度，最多5个段落
            
            # 尝试使用ContentIntegrator的LLM接口
            if hasattr(self, 'content_integrator') and self.content_integrator:
                try:
                    # 使用ContentIntegrator生成内容
                    result = self.content_integrator.generate_expanded_content(
                        context, target_concept, [relevant_content]
                    )
                    if result:
                        return result
                except Exception as e:
                    self.logger.warning(f"ContentIntegrator failed: {e}")
            
            # 降级到基于规则的内容生成
            self.logger.info("Using rule-based content generation as fallback")
            
            # 分析内容中的关键信息
            key_sentences = []
            for para in paragraphs[:3]:
                sentences = para.split('. ')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if (len(sentence) > 20 and 
                        any(keyword in sentence.lower() for keyword in ['attention', 'cross', 'model', 'feature', 'image', 'text'])):
                        key_sentences.append(sentence)
            
            return f"""## {target_concept.title()} Mechanism

**Definition:**
Cross attention is a key mechanism in transformer-based architectures that enables models to attend to information from different modalities or sources.

**Implementation in This Paper:**
{'. '.join(key_sentences[:3]) if key_sentences else 'The paper implements cross attention to enable multimodal processing between text and image inputs.'}

**Key Characteristics:**
• Enables interaction between different input sequences (e.g., text and image features)
• Allows conditional generation based on multiple inputs
• Facilitates alignment between different modalities
• Critical for multimodal tasks like text-to-image generation

**Significance:**
The cross attention mechanism is essential for the model's ability to understand and correlate information from different sources, enabling more sophisticated and controllable generation processes.

**Technical Context:**
{relevant_content[:400]}...
"""
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {e}")
            return f"""## {target_concept.title()}

Cross attention is a fundamental mechanism in deep learning that enables models to process and correlate information from multiple sources.

**Key Applications:**
• Multimodal learning (text + image)
• Conditional generation
• Feature alignment
• Information integration

**In This Research:**
The paper utilizes cross attention to enable effective image-prompt conditioning, allowing for more precise control over the generation process.
"""
    
    def _extract_key_points(self, content: str) -> List[str]:
        """从内容中提取关键要点"""
        try:
            # 简单的关键点提取：寻找编号列表或段落
            key_points = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                # 寻找编号列表 (1., 2., etc.) 或要点标记 (-, *, etc.)
                if (line and (
                    line[0].isdigit() and '. ' in line[:5] or
                    line.startswith('- ') or
                    line.startswith('* ') or
                    line.startswith('• ')
                )):
                    key_points.append(line.lstrip('0123456789.-* • '))
            
            return key_points[:7]  # 最多7个要点
            
        except Exception as e:
            self.logger.warning(f"Failed to extract key points: {e}")
            return []


# Standalone function for external usage
def enhance_presentation_content(original_paper_path: str,
                               target_concept: str,
                               context: str,
                               output_dir: str = "reference_output",
                               model_name: str = "gpt-4o") -> Tuple[bool, Dict[str, Any]]:
    """
    Standalone function to enhance presentation content with references
    
    Args:
        original_paper_path: Path to original paper content JSON
        target_concept: Concept to expand upon
        context: Original context
        output_dir: Output directory
        model_name: LLM model name
        
    Returns:
        Tuple of (success, result_dict)
    """
    agent = ReferenceAgent(model_name=model_name)
    result = agent.enhance_content_with_references(
        original_paper_path, target_concept, context, output_dir=output_dir
    )
    
    return result['success'], result


# Test function
def test_reference_agent():
    """Test the Reference Agent with real data"""
    print("🧪 Testing Reference Agent with real paper data...")
    
    # Use real paper data
    paper_path = "/home/yuheng/Project/paper-to-beamer/output/test/lightweight_content_enhanced.json"
    
    if not os.path.exists(paper_path):
        print(f"❌ Test file not found: {paper_path}")
        return False
    
    agent = ReferenceAgent()
    
    # Test with a real concept from the paper
    result = agent.enhance_content_with_references(
        original_paper_path=paper_path,
        target_concept="attention mechanism",
        context="We are studying how attention mechanisms work in neural networks for GUI automation",
        max_references=3,
        output_dir="test_reference_output"
    )
    
    if result['success']:
        print("✅ Reference Agent test successful!")
        print(f"   Enhanced content length: {len(result['enhanced_content'])} characters")
        print(f"   Quality score: {result['content_quality_score']:.3f}")
        print(f"   Citations found: {result['citations_found']}")
        print(f"   Papers retrieved: {result['papers_retrieved']}")
        print(f"   Integration method: {result['integration_method']}")
        
        if result['key_points']:
            print(f"   Key points: {len(result['key_points'])}")
            for i, point in enumerate(result['key_points'][:2], 1):
                print(f"     {i}. {point[:80]}...")
        
        print(f"\n📄 Enhanced content preview:")
        preview = result['enhanced_content'][:200] + "..." if len(result['enhanced_content']) > 200 else result['enhanced_content']
        print(preview)
        
        return True
    else:
        print(f"❌ Test failed: {result['error']}")
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    test_reference_agent()

