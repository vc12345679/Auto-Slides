#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Citation Extractor Module for Reference Agent
从markdown文本中提取和解析引用信息
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Citation:
    """引用信息数据类"""
    anchor: str  # 页面锚点，如 "page-9-0"
    authors: List[str]  # 作者列表
    title: str  # 论文标题
    year: str  # 发表年份
    venue: str  # 发表场所/期刊
    doi: str = ""  # DOI
    url: str = ""  # URL
    context: str = ""  # 引用上下文
    arxiv_id: str = ""  # arXiv ID
    
    def get_cache_key(self) -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{self.authors}_{self.title}_{self.year}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'anchor': self.anchor,
            'authors': self.authors,
            'title': self.title,
            'year': self.year,
            'venue': self.venue,
            'doi': self.doi,
            'url': self.url,
            'context': self.context,
            'arxiv_id': self.arxiv_id
        }


class CitationExtractor:
    """引用提取器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 引用模式匹配
        self.citation_patterns = [
            # 标准模式: [\\(Author,](#page-x-y) [year\\)](#page-x-y)
            r'\[\\?\([^)]+\)\]\(#[^)]+\)\s*\[\\?\([^)]+\)\]\(#[^)]+\)',
            # 数字引用: [\\[18\\]](#page-x-y) 或 [18](#page-x-y)
            r'\[\\*\[?\d+\\*\]?\]\(#page-\d+-\d+\)',
            # 多引用: [\\[14,](#page-x-y) [15\\]](#page-x-y)
            r'\[\\*\[?\d+,?\s*\\*\]?\]\(#page-\d+-\d+\)',
            # 简化模式: [Author et al., year](#page-x-y)
            r'\[[^]]+\]\(#page-\d+-\d+\)',
            # 单个引用: [\\(Author, year\\)](#page-x-y)
            r'\[\\?\([^)]+\)\]\(#page-\d+-\d+\)'
        ]
        
    def extract_relevant_citations(self, 
                                 full_text: str, 
                                 target_concept: str,
                                 context_window: int = 500) -> List[Citation]:
        """
        从原文中提取与目标概念相关的引用
        
        Args:
            full_text: 完整的markdown文本
            target_concept: 目标概念/关键词
            context_window: 上下文窗口大小
            
        Returns:
            List[Citation]: 相关引用列表
        """
        self.logger.info(f"开始提取与 '{target_concept}' 相关的引用")
        
        # 1. 找到包含目标概念的段落
        relevant_paragraphs = self._find_concept_paragraphs(full_text, target_concept, context_window)
        self.logger.info(f"找到 {len(relevant_paragraphs)} 个相关段落")
        
        # 2. 从相关段落中提取引用
        citations = []
        for paragraph in relevant_paragraphs:
            paragraph_citations = self._extract_citations_from_text(paragraph, full_text)
            citations.extend(paragraph_citations)
        
        # 3. 去重和清理
        unique_citations = self._deduplicate_citations(citations)
        self.logger.info(f"提取到 {len(unique_citations)} 个唯一引用")
        
        return unique_citations
    
    def _find_concept_paragraphs(self, full_text: str, target_concept: str, context_window: int) -> List[str]:
        """找到包含目标概念的段落（增强版）"""
        paragraphs = []
        
        # 生成相关概念的搜索模式
        search_patterns = self._generate_concept_patterns(target_concept)
        self.logger.info(f"生成了 {len(search_patterns)} 个搜索模式: {search_patterns}")
        
        # 按句子分割文本
        sentences = re.split(r'[.!?]+', full_text)
        
        for i, sentence in enumerate(sentences):
            # 检查句子是否包含任何相关模式
            for pattern in search_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # 构建上下文窗口
                    start_idx = max(0, i - 2)  # 前2句
                    end_idx = min(len(sentences), i + 3)  # 后2句
                    
                    context = '. '.join(sentences[start_idx:end_idx])
                    if context not in paragraphs:  # 避免重复
                        paragraphs.append(context)
                    break  # 找到一个匹配就够了
        
        return paragraphs
    
    def _generate_concept_patterns(self, target_concept: str) -> List[str]:
        """生成概念搜索的多种模式"""
        patterns = []
        
        # 基础概念（原始形式）
        patterns.append(re.escape(target_concept))
        
        # 处理连字符变体
        if ' ' in target_concept:
            # "cross attention" -> "cross-attention"
            hyphenated = target_concept.replace(' ', '-')
            patterns.append(re.escape(hyphenated))
            
            # "cross attention" -> "cross_attention"  
            underscored = target_concept.replace(' ', '_')
            patterns.append(re.escape(underscored))
        
        # 处理单词变体和扩展
        concept_words = target_concept.lower().split()
        
        if len(concept_words) >= 2:
            # 生成部分匹配模式
            for word in concept_words:
                if len(word) > 3:  # 只处理较长的单词
                    # "attention" -> "attention mechanism", "attention layer", etc.
                    patterns.append(f"{re.escape(word)}\\s+\\w+")
                    patterns.append(f"\\w+\\s+{re.escape(word)}")
        
        # 特殊情况处理
        if "attention" in target_concept.lower():
            patterns.extend([
                r"attention\s+mechanism",
                r"attention\s+layer",
                r"self-attention",
                r"multi-head\s+attention",
                r"scaled\s+dot-product\s+attention"
            ])
        
        if "cross" in target_concept.lower():
            patterns.extend([
                r"cross-modal",
                r"cross-domain", 
                r"cross-attention\s+mechanism",
                r"decoupled\s+cross-attention"
            ])
        
        # 技术论文常见模式
        base_concept = target_concept.replace('-', '').replace('_', ' ')
        patterns.extend([
            f"{re.escape(base_concept)}\\s+approach",
            f"{re.escape(base_concept)}\\s+method",
            f"{re.escape(base_concept)}\\s+technique",
            f"{re.escape(base_concept)}\\s+strategy",
            f"propose.*{re.escape(base_concept)}",
            f"using.*{re.escape(base_concept)}",
            f"based.*{re.escape(base_concept)}"
        ])
        
        # 上下文相关扩展（对于attention相关概念）
        if any(word in target_concept.lower() for word in ['attention', 'transformer', 'neural']):
            patterns.extend([
                r"transformer",
                r"neural\s+network",
                r"encoder-decoder",
                r"sequence-to-sequence",
                r"seq2seq",
                r"bert",
                r"gpt",
                r"vaswani.*attention",
                r"attention.*all.*need"  # "Attention is All You Need"的引用模式
            ])
        
        self.logger.debug(f"为 '{target_concept}' 生成的搜索模式: {patterns}")
        return list(set(patterns))  # 去重
    
    def _extract_citations_from_text(self, text: str, full_text: str) -> List[Citation]:
        """从文本中提取引用"""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            
            for match in matches:
                citation = self._parse_citation_match(match.group(), full_text)
                if citation:
                    citation.context = text[:100] + "..." if len(text) > 100 else text
                    citations.append(citation)
        
        return citations
    
    def _parse_citation_match(self, citation_text: str, full_text: str) -> Optional[Citation]:
        """解析单个引用匹配"""
        try:
            # 提取页面锚点
            anchor_match = re.search(r'#(page-\d+-\d+)', citation_text)
            if not anchor_match:
                return None
            
            anchor = anchor_match.group(1)
            
            # 在参考文献部分查找完整信息
            reference_info = self._find_reference_by_anchor(full_text, anchor)
            
            if not reference_info:
                # 如果找不到参考文献，尝试从引用文本本身解析
                reference_info = self._parse_inline_citation(citation_text)
            
            if reference_info:
                return Citation(
                    anchor=anchor,
                    authors=reference_info.get('authors', []),
                    title=reference_info.get('title', ''),
                    year=reference_info.get('year', ''),
                    venue=reference_info.get('venue', ''),
                    doi=reference_info.get('doi', ''),
                    url=reference_info.get('url', ''),
                    arxiv_id=reference_info.get('arxiv_id', '')
                )
            
        except Exception as e:
            self.logger.warning(f"解析引用时出错: {e}")
            
        return None
    
    def _find_reference_by_anchor(self, full_text: str, anchor: str) -> Optional[Dict[str, Any]]:
        """根据锚点在参考文献部分查找完整信息"""
        try:
            # 查找参考文献部分 - 支持多种格式
            references_patterns = [
                r'# References\s*(.*?)(?=#|\Z)',  # # References
                r'## References\s*(.*?)(?=##|\Z)',  # ## References  
                r'References\s*(.*?)(?=\n#|\Z)'     # References
            ]
            
            references_text = None
            for pattern in references_patterns:
                references_match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
                if references_match:
                    references_text = references_match.group(1)
                    break
            
            if not references_text:
                self.logger.warning("未找到参考文献部分")
                return None
            
            # 查找对应锚点的引用
            anchor_pattern = rf'<span id="{anchor}"></span>(.*?)(?=<span id=|$)'
            anchor_match = re.search(anchor_pattern, references_text, re.DOTALL)
            
            if anchor_match:
                reference_text = anchor_match.group(1).strip()
                # 清理HTML标签
                reference_text = re.sub(r'<[^>]+>', '', reference_text)
                return self._parse_reference_text(reference_text)
            else:
                self.logger.warning(f"未找到锚点 {anchor} 对应的参考文献")
            
        except Exception as e:
            self.logger.warning(f"查找参考文献时出错: {e}")
            
        return None
    
    def _parse_reference_text(self, reference_text: str) -> Dict[str, Any]:
        """解析参考文献文本"""
        info = {
            'authors': [],
            'title': '',
            'year': '',
            'venue': '',
            'doi': '',
            'url': '',
            'arxiv_id': ''
        }
        
        try:
            # 提取年份
            year_match = re.search(r'\b(19|20)\d{2}\b', reference_text)
            if year_match:
                info['year'] = year_match.group()
            
            # 提取DOI
            doi_match = re.search(r'doi:\s*([^\s,]+)', reference_text, re.IGNORECASE)
            if doi_match:
                info['doi'] = doi_match.group(1)
            
            # 提取arXiv ID
            arxiv_match = re.search(r'arxiv:(\d+\.\d+)', reference_text, re.IGNORECASE)
            if arxiv_match:
                info['arxiv_id'] = arxiv_match.group(1)
            
            # 提取URL
            url_match = re.search(r'https?://[^\s,\]]+', reference_text)
            if url_match:
                info['url'] = url_match.group()
            
            # 解析作者和标题（改进处理）
            # 通常格式：作者名. 年份. [标题](URL) *期刊*, 信息.
            
            # 提取标题（在方括号中）
            title_match = re.search(r'\[([^\]]+)\]', reference_text)
            if title_match:
                info['title'] = title_match.group(1)
            
            # 解析作者（年份之前的部分）
            year_pos = reference_text.find(info['year']) if info['year'] else -1
            if year_pos > 0:
                authors_part = reference_text[:year_pos].strip()
                # 移除末尾的年份和标点
                authors_part = re.sub(r'\.\s*$', '', authors_part)
                if authors_part:
                    info['authors'] = [authors_part]
            
            # 提取期刊/会议（在*号之间）
            venue_match = re.search(r'\*([^*]+)\*', reference_text)
            if venue_match:
                info['venue'] = venue_match.group(1)
            
        except Exception as e:
            self.logger.warning(f"解析参考文献文本时出错: {e}")
        
        return info
    
    def _parse_inline_citation(self, citation_text: str) -> Optional[Dict[str, Any]]:
        """从内联引用文本中解析基本信息"""
        info = {
            'authors': [],
            'title': '',
            'year': '',
            'venue': '',
            'doi': '',
            'url': '',
            'arxiv_id': ''
        }
        
        try:
            # 提取年份
            year_match = re.search(r'\b(19|20)\d{2}\b', citation_text)
            if year_match:
                info['year'] = year_match.group()
            
            # 提取作者（从括号中）
            author_match = re.search(r'\[\\?\(([^)]+)\)', citation_text)
            if author_match:
                author_text = author_match.group(1)
                # 移除年份
                author_text = re.sub(r'\b(19|20)\d{2}\b', '', author_text).strip(' ,')
                info['authors'] = [author_text]
            
        except Exception as e:
            self.logger.warning(f"解析内联引用时出错: {e}")
        
        return info if info['authors'] or info['year'] else None
    
    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """去重引用列表"""
        seen = set()
        unique_citations = []
        
        for citation in citations:
            # 使用锚点作为去重键
            if citation.anchor not in seen:
                seen.add(citation.anchor)
                unique_citations.append(citation)
        
        return unique_citations


# 测试函数
def test_citation_extractor():
    """测试引用提取器"""
    print("🧪 测试引用提取器...")
    
    extractor = CitationExtractor()
    
    # 创建测试数据
    test_text = """
    Recent advancements in Large Language Models (LLMs) have led to the development of 
    intelligent agents [\\(OpenAI,](#page-9-0) [2024\\)](#page-9-0). These models demonstrate 
    strong reasoning capabilities [\\(Huang](#page-9-2) [et al.,](#page-9-2) [2022\\)](#page-9-2).
    
    ## References
    
    <span id="page-9-0"></span>OpenAI. 2024. [Gpt-4 technical report.](https://arxiv.org/abs/2303.08774) *Preprint*, arXiv:2303.08774.
    
    <span id="page-9-2"></span>Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2022. [Large language models can self-improve.](https://arxiv.org/abs/2210.11610) *Preprint*, arXiv:2210.11610.
    """
    
    # 测试提取功能
    citations = extractor.extract_relevant_citations(test_text, "Large Language Models")
    
    print(f"✅ 提取到 {len(citations)} 个引用:")
    for i, citation in enumerate(citations, 1):
        print(f"  {i}. {citation.authors} ({citation.year})")
        print(f"     标题: {citation.title}")
        print(f"     锚点: {citation.anchor}")
        print(f"     arXiv: {citation.arxiv_id}")
        print()
    
    return len(citations) > 0


if __name__ == "__main__":
    test_citation_extractor()
