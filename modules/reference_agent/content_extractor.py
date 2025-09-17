#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content Extractor Module for Reference Agent
内容提取模块，从检索到的文献中提取相关内容
"""

import os
import logging
import tempfile
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# 导入现有的PDF解析模块
import sys
sys.path.append('../..')
from modules.lightweight_extractor import LightweightExtractor

from .literature_searcher import PaperResult


@dataclass
class ExtractedContent:
    """提取的内容数据类"""
    paper_info: Dict[str, Any]
    relevant_sections: List[str]
    key_sentences: List[str]
    confidence_score: float
    extraction_method: str
    full_text: str = ""
    abstract: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'paper_info': self.paper_info,
            'relevant_sections': self.relevant_sections,
            'key_sentences': self.key_sentences,
            'confidence_score': self.confidence_score,
            'extraction_method': self.extraction_method,
            'full_text': self.full_text,
            'abstract': self.abstract
        }


class ContentExtractor:
    """内容提取器"""
    
    def __init__(self, llm_interface=None):
        self.logger = logging.getLogger(__name__)
        self.llm_interface = llm_interface
        self.temp_dir = Path(tempfile.gettempdir()) / "reference_pdfs"
        self.temp_dir.mkdir(exist_ok=True)
    
    def extract_relevant_content(self, 
                               paper_result: PaperResult,
                               target_concept: str,
                               original_context: str,
                               max_sections: int = 3) -> Optional[ExtractedContent]:
        """
        从论文结果中提取与目标概念相关的内容
        
        Args:
            paper_result: 论文检索结果
            target_concept: 目标概念
            original_context: 原始上下文
            max_sections: 最大提取段落数
            
        Returns:
            ExtractedContent: 提取的内容
        """
        self.logger.info(f"开始提取内容: {paper_result.title[:50]}...")
        
        try:
            # 优先尝试PDF全文（信息更完整）
            if paper_result.has_pdf_access():
                try:
                    return self._extract_from_pdf(paper_result, target_concept, original_context, max_sections)
                except Exception as pdf_error:
                    self.logger.warning(f"PDF extraction failed, falling back to abstract: {pdf_error}")
                    # 如果PDF失败，回退到摘要
                    if paper_result.abstract:
                        return self._extract_from_abstract(paper_result, target_concept, original_context)
            elif paper_result.abstract:
                # 使用摘要
                return self._extract_from_abstract(paper_result, target_concept, original_context)
            else:
                self.logger.warning("No content available for extraction")
                return None
                
        except Exception as e:
            self.logger.error(f"内容提取失败: {e}")
            return None
    
    def _extract_from_pdf(self, 
                         paper_result: PaperResult,
                         target_concept: str,
                         original_context: str,
                         max_sections: int) -> Optional[ExtractedContent]:
        """从PDF中提取内容"""
        try:
            # 下载PDF
            pdf_path = self._download_pdf(paper_result.pdf_url, paper_result.paper_id)
            if not pdf_path:
                return None
            
            # 使用现有的PDF解析器（使用绝对路径）
            abs_pdf_path = pdf_path.absolute()
            abs_output_dir = self.temp_dir.absolute()
            
            # 确保在正确的工作目录下运行
            import os
            original_cwd = os.getcwd()
            try:
                # 切换到项目根目录以正确加载模型
                project_root = Path(__file__).parent.parent.parent
                os.chdir(project_root)
                
                extractor = LightweightExtractor(str(abs_pdf_path), output_dir=str(abs_output_dir))
                content = extractor.extract_content()
            finally:
                # 恢复原工作目录
                os.chdir(original_cwd)
            
            if not content or not content.get('full_text'):
                self.logger.warning("PDF解析失败或无文本内容")
                return None
            
            full_text = content['full_text']
            
            # 提取相关段落
            relevant_sections = self._find_relevant_sections(full_text, target_concept, max_sections)
            
            # 使用LLM提取关键句子（如果可用）
            key_sentences = []
            if self.llm_interface and relevant_sections:
                key_sentences = self._extract_key_sentences_with_llm(
                    relevant_sections, target_concept, original_context
                )
            else:
                # 简单的关键词匹配
                key_sentences = self._extract_key_sentences_simple(relevant_sections, target_concept)
            
            # 计算相关性分数
            confidence_score = self._calculate_relevance_score(relevant_sections, target_concept)
            
            return ExtractedContent(
                paper_info=paper_result.to_dict(),
                relevant_sections=relevant_sections,
                key_sentences=key_sentences,
                confidence_score=confidence_score,
                extraction_method="pdf_full_text",
                full_text=full_text,
                abstract=paper_result.abstract
            )
            
        except Exception as e:
            self.logger.error(f"PDF内容提取失败: {e}")
            return None
        finally:
            # 清理临时文件
            if 'pdf_path' in locals() and pdf_path and pdf_path.exists():
                try:
                    pdf_path.unlink()
                except:
                    pass
    
    def _extract_from_abstract(self, 
                             paper_result: PaperResult,
                             target_concept: str,
                             original_context: str) -> Optional[ExtractedContent]:
        """从摘要中提取内容"""
        try:
            abstract = paper_result.abstract
            if not abstract:
                return None
            
            # 检查摘要与目标概念的相关性
            if target_concept.lower() not in abstract.lower():
                self.logger.info("摘要与目标概念相关性较低")
                return None
            
            # 分句处理
            sentences = self._split_sentences(abstract)
            relevant_sentences = [s for s in sentences if target_concept.lower() in s.lower()]
            
            # 计算相关性分数
            confidence_score = len(relevant_sentences) / len(sentences) if sentences else 0
            
            return ExtractedContent(
                paper_info=paper_result.to_dict(),
                relevant_sections=[abstract],
                key_sentences=relevant_sentences,
                confidence_score=confidence_score,
                extraction_method="abstract_only",
                abstract=abstract
            )
            
        except Exception as e:
            self.logger.error(f"摘要内容提取失败: {e}")
            return None
    
    def _download_pdf(self, pdf_url: str, paper_id: str) -> Optional[Path]:
        """下载PDF文件"""
        try:
            self.logger.info(f"下载PDF: {pdf_url}")
            
            # 创建临时文件名
            filename = f"{paper_id[:10]}_{hash(pdf_url) % 10000}.pdf"
            pdf_path = self.temp_dir / filename
            
            # 如果文件已存在，直接返回
            if pdf_path.exists():
                return pdf_path
            
            # 下载文件
            response = requests.get(pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"PDF下载成功: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            self.logger.error(f"PDF下载失败: {e}")
            return None
    
    def _find_relevant_sections(self, 
                              full_text: str, 
                              target_concept: str, 
                              max_sections: int) -> List[str]:
        """查找相关段落"""
        try:
            # 按段落分割
            paragraphs = self._split_paragraphs(full_text)
            
            # 查找包含目标概念的段落
            relevant_paragraphs = []
            concept_lower = target_concept.lower()
            
            for paragraph in paragraphs:
                if len(paragraph.strip()) < 50:  # 过滤太短的段落
                    continue
                    
                if concept_lower in paragraph.lower():
                    # 计算相关性得分
                    score = self._calculate_paragraph_relevance(paragraph, target_concept)
                    relevant_paragraphs.append((paragraph, score))
            
            # 按相关性排序，取前N个
            relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
            
            return [p[0] for p in relevant_paragraphs[:max_sections]]
            
        except Exception as e:
            self.logger.error(f"查找相关段落失败: {e}")
            return []
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """分割段落"""
        # 按双换行符分割段落
        paragraphs = text.split('\n\n')
        
        # 清理和过滤
        cleaned_paragraphs = []
        for p in paragraphs:
            cleaned = p.strip().replace('\n', ' ')
            if len(cleaned) > 50:  # 只保留足够长的段落
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        import re
        
        # 简单的句子分割
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _calculate_paragraph_relevance(self, paragraph: str, target_concept: str) -> float:
        """计算段落与目标概念的相关性"""
        try:
            paragraph_lower = paragraph.lower()
            concept_lower = target_concept.lower()
            
            # 基础分数：包含目标概念
            score = 0.0
            if concept_lower in paragraph_lower:
                score += 0.5
            
            # 额外分数：概念出现频率
            concept_count = paragraph_lower.count(concept_lower)
            score += min(concept_count * 0.1, 0.3)
            
            # 额外分数：段落长度适中
            length = len(paragraph)
            if 100 <= length <= 500:
                score += 0.2
            elif 500 < length <= 1000:
                score += 0.1
            
            return score
            
        except Exception as e:
            self.logger.error(f"计算段落相关性失败: {e}")
            return 0.0
    
    def _extract_key_sentences_with_llm(self, 
                                      sections: List[str],
                                      target_concept: str,
                                      original_context: str) -> List[str]:
        """使用LLM提取关键句子"""
        try:
            if not self.llm_interface or not sections:
                return []
            
            # 构建提示词
            sections_text = '\n\n'.join(sections)
            
            prompt = f"""
            从以下文献段落中提取与概念"{target_concept}"相关的关键句子。
            
            原始上下文：
            {original_context[:200]}...
            
            文献段落：
            {sections_text[:2000]}...
            
            请提取3-5个最相关的关键句子，每个句子应该：
            1. 直接包含或解释"{target_concept}"
            2. 提供有价值的信息
            3. 与原始上下文相关
            
            请用以下格式输出：
            1. [第一个关键句子]
            2. [第二个关键句子]
            ...
            """
            
            # 调用LLM（使用现有的extraction方法）
            response = self.llm_interface.call_for_extraction(
                system_prompt="你是一个专业的学术文献分析专家，擅长从文献中提取关键信息。",
                user_prompt=prompt
            )
            
            # 解析响应
            key_sentences = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or line.startswith('-')):
                    sentence = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if sentence:
                        key_sentences.append(sentence)
            
            return key_sentences[:5]  # 最多返回5个
            
        except Exception as e:
            self.logger.error(f"LLM提取关键句子失败: {e}")
            return []
    
    def _extract_key_sentences_simple(self, sections: List[str], target_concept: str) -> List[str]:
        """简单的关键句子提取"""
        try:
            key_sentences = []
            concept_lower = target_concept.lower()
            
            for section in sections:
                sentences = self._split_sentences(section)
                
                for sentence in sentences:
                    if concept_lower in sentence.lower() and len(sentence) > 30:
                        key_sentences.append(sentence)
                        
                        if len(key_sentences) >= 5:  # 最多5个
                            break
                
                if len(key_sentences) >= 5:
                    break
            
            return key_sentences
            
        except Exception as e:
            self.logger.error(f"简单提取关键句子失败: {e}")
            return []
    
    def _calculate_relevance_score(self, sections: List[str], target_concept: str) -> float:
        """计算内容相关性分数"""
        try:
            if not sections:
                return 0.0
            
            total_score = 0.0
            concept_lower = target_concept.lower()
            
            for section in sections:
                section_lower = section.lower()
                
                # 基础分数：包含概念
                if concept_lower in section_lower:
                    total_score += 0.3
                
                # 额外分数：概念出现频率
                concept_count = section_lower.count(concept_lower)
                total_score += min(concept_count * 0.1, 0.2)
                
                # 额外分数：内容质量（长度适中）
                if 100 <= len(section) <= 1000:
                    total_score += 0.1
            
            # 归一化到0-1区间
            return min(total_score / len(sections), 1.0)
            
        except Exception as e:
            self.logger.error(f"计算相关性分数失败: {e}")
            return 0.0


# 测试函数
def test_content_extractor():
    """测试内容提取器"""
    print("🧪 测试内容提取器...")
    
    # 创建测试用的PaperResult
    test_paper = PaperResult(
        paper_id="test_paper",
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer"],
        year="2017",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        venue="NIPS",
        confidence_score=0.8,
        search_strategy="test"
    )
    
    extractor = ContentExtractor()
    
    # 测试从摘要提取
    print("\n📄 测试从摘要提取内容...")
    result = extractor.extract_relevant_content(
        test_paper, 
        "attention mechanism", 
        "我们正在研究注意力机制在神经网络中的应用"
    )
    
    if result:
        print("✅ 提取成功!")
        print(f"   提取方法: {result.extraction_method}")
        print(f"   相关性分数: {result.confidence_score:.3f}")
        print(f"   相关段落数: {len(result.relevant_sections)}")
        print(f"   关键句子数: {len(result.key_sentences)}")
        
        if result.key_sentences:
            print("   关键句子:")
            for i, sentence in enumerate(result.key_sentences[:2], 1):
                print(f"     {i}. {sentence[:80]}...")
        
        return True
    else:
        print("❌ 提取失败")
        return False


if __name__ == "__main__":
    test_content_extractor()
