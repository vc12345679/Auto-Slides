#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content Integrator Module for Reference Agent
内容整合模块，将多篇文献的内容智能整合生成扩展材料
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from .content_extractor import ExtractedContent

# Load environment variables
if os.path.exists("../../.env"):
    load_dotenv("../../.env")
elif os.path.exists("../../env.local"):
    load_dotenv("../../env.local")

# Import prompts
import sys
sys.path.append('../..')
from prompts.reference_content_integration import (
    CONTENT_INTEGRATION_SYSTEM_PROMPT,
    create_content_integration_user_prompt,
    SIMPLE_INTEGRATION_TEMPLATE
)

# 导入LangChain组件
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class IntegratedContent:
    """整合后的内容"""
    expanded_content: str
    source_papers: List[Dict[str, Any]]
    integration_method: str
    quality_score: float
    summary: str = ""
    key_points: List[str] = None
    
    def __post_init__(self):
        if self.key_points is None:
            self.key_points = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'expanded_content': self.expanded_content,
            'source_papers': self.source_papers,
            'integration_method': self.integration_method,
            'quality_score': self.quality_score,
            'summary': self.summary,
            'key_points': self.key_points
        }


class ContentIntegrator:
    """内容整合器"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
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
                self.logger.warning(f"Failed to load .env file: {e}")
        
        self.api_key = api_key
        
        # 初始化LangChain模型
        self._init_model()
    
    def _init_model(self):
        """初始化语言模型"""
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning("LangChain not available, content integration functionality disabled")
            self.llm = None
            return
        
        if not self.api_key:
            self.logger.warning("No OpenAI API key provided, content integration functionality disabled")
            self.llm = None
            return
        
        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key
            )
            self.logger.info(f"Content Integrator initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize language model: {str(e)}")
            self.llm = None
    
    def generate_expanded_content(self, 
                                original_context: str,
                                target_concept: str,
                                extracted_contents: List[ExtractedContent],
                                max_length: int = 2000) -> Optional[IntegratedContent]:
        """
        整合多篇文献内容生成扩展材料
        
        Args:
            original_context: 原始上下文
            target_concept: 目标概念
            extracted_contents: 提取的内容列表
            max_length: 最大长度
            
        Returns:
            IntegratedContent: 整合后的内容
        """
        self.logger.info(f"开始整合 {len(extracted_contents)} 篇文献的内容")
        
        if not extracted_contents:
            self.logger.warning("没有可整合的内容")
            return None
        
        try:
            # 按相关性排序
            sorted_contents = sorted(
                extracted_contents, 
                key=lambda x: x.confidence_score, 
                reverse=True
            )
            
            # 限制使用前3篇最相关的文献
            top_contents = sorted_contents[:3]
            
            # 优先尝试LLM整合，失败时回退到简单整合
            if self.llm:
                try:
                    result = self._integrate_with_llm(
                        original_context, target_concept, top_contents, max_length
                    )
                    if result:
                        return result
                except Exception as e:
                    self.logger.warning(f"LLM整合失败，回退到简单整合: {e}")
            
            # 使用简单规则进行整合
            return self._integrate_simple(
                original_context, target_concept, top_contents, max_length
            )
                
        except Exception as e:
            self.logger.error(f"内容整合失败: {e}")
            return None
    
    def _integrate_with_llm(self, 
                          original_context: str,
                          target_concept: str,
                          contents: List[ExtractedContent],
                          max_length: int) -> Optional[IntegratedContent]:
        """使用LLM进行智能整合"""
        try:
            # 构建整合提示词
            system_prompt = CONTENT_INTEGRATION_SYSTEM_PROMPT
            
            # 构建文献信息
            literature_info = []
            for i, content in enumerate(contents, 1):
                paper_info = content.paper_info
                title = paper_info.get('title', 'Unknown title')[:60]
                authors = paper_info.get('authors', ['Unknown author'])
                year = paper_info.get('year', 'Unknown year')
                
                lit_section = [
                    f"Literature {i}: {title}... ({', '.join(authors[:2])}{'et al.' if len(authors) > 2 else ''}, {year})"
                ]
                
                if content.key_sentences:
                    lit_section.append("Key content:")
                    for j, sentence in enumerate(content.key_sentences[:3], 1):
                        lit_section.append(f"  - {sentence}")
                
                literature_info.append("\n".join(lit_section))
            
            literature_text = "\n\n".join(literature_info)
            
            user_prompt = create_content_integration_user_prompt(
                original_context, target_concept, literature_text, max_length
            )
            
            # 调用LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析响应
            result = self._parse_llm_response(response_text, contents)
            
            # 验证质量
            quality_score = self._validate_content_quality(result['expanded_content'], target_concept)
            
            return IntegratedContent(
                expanded_content=result['expanded_content'],
                source_papers=[c.paper_info for c in contents],
                integration_method="llm_intelligent",
                quality_score=quality_score,
                summary=result.get('summary', ''),
                key_points=result.get('key_points', [])
            )
            
        except Exception as e:
            self.logger.error(f"LLM整合失败: {e}")
            return None
    
    def _integrate_simple(self, 
                        original_context: str,
                        target_concept: str,
                        contents: List[ExtractedContent],
                        max_length: int) -> Optional[IntegratedContent]:
        """使用简单规则进行整合"""
        try:
            # 收集所有关键句子
            all_sentences = []
            source_papers = []
            
            for content in contents:
                # 添加文献信息
                paper_info = content.paper_info
                source_papers.append(paper_info)
                
                # 添加关键句子
                for sentence in content.key_sentences[:2]:  # 每篇文献最多2个句子
                    all_sentences.append(sentence)
            
            # 构建扩展内容（使用英文模板）
            key_points_text = "\n".join([
                f"{i}. {sentence}" for i, sentence in enumerate(all_sentences[:5], 1)
            ])
            
            expanded_content = SIMPLE_INTEGRATION_TEMPLATE.format(
                target_concept=target_concept,
                key_points=key_points_text
            )
            
            # 控制长度
            if len(expanded_content) > max_length:
                expanded_content = expanded_content[:max_length-3] + "..."
            
            # 计算质量分数
            quality_score = self._validate_content_quality(expanded_content, target_concept)
            
            return IntegratedContent(
                expanded_content=expanded_content,
                source_papers=source_papers,
                integration_method="simple_concatenation",
                quality_score=quality_score,
                summary=f"Integrated research from {len(contents)} papers about {target_concept}",
                key_points=all_sentences[:5]
            )
            
        except Exception as e:
            self.logger.error(f"简单整合失败: {e}")
            return None
    

    
    def _parse_llm_response(self, response: str, contents: List[ExtractedContent]) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            result = {
                'expanded_content': '',
                'key_points': [],
                'summary': ''
            }
            
            lines = response.split('\n')
            current_section = None
            content_lines = []
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('# 扩展内容'):
                    current_section = 'content'
                    content_lines = []
                elif line.startswith('# 关键要点'):
                    if current_section == 'content':
                        result['expanded_content'] = '\n'.join(content_lines).strip()
                    current_section = 'points'
                elif line.startswith('# 内容总结'):
                    current_section = 'summary'
                elif line:
                    if current_section == 'content':
                        content_lines.append(line)
                    elif current_section == 'points':
                        if line.startswith(('1.', '2.', '3.', '4.', '5.', '-')):
                            point = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                            if point:
                                result['key_points'].append(point)
                    elif current_section == 'summary':
                        result['summary'] = line
                        break
            
            # 如果没有正确解析到扩展内容，使用整个响应
            if not result['expanded_content']:
                result['expanded_content'] = response.strip()
            
            return result
            
        except Exception as e:
            self.logger.error(f"解析LLM响应失败: {e}")
            return {
                'expanded_content': response.strip(),
                'key_points': [],
                'summary': '内容整合完成'
            }
    
    def _validate_content_quality(self, content: str, target_concept: str) -> float:
        """验证内容质量"""
        try:
            score = 0.0
            
            if not content:
                return 0.0
            
            # 长度合理性
            length = len(content)
            if 200 <= length <= 2000:
                score += 0.3
            elif 100 <= length < 200 or 2000 < length <= 3000:
                score += 0.2
            elif length < 100:
                score += 0.1
            
            # 包含目标概念
            if target_concept.lower() in content.lower():
                score += 0.3
            
            # 结构完整性（包含多个句子）
            sentences = content.split('.')
            if len(sentences) >= 3:
                score += 0.2
            elif len(sentences) >= 2:
                score += 0.1
            
            # 学术性词汇
            academic_words = ['研究', '表明', '发现', '分析', '结果', '方法', '理论', '模型', '实验']
            academic_count = sum(1 for word in academic_words if word in content)
            score += min(academic_count * 0.05, 0.2)
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"质量验证失败: {e}")
            return 0.5  # 默认中等质量


# 测试函数
def test_content_integrator():
    """测试内容整合器"""
    print("🧪 测试内容整合器...")
    
    # 创建测试用的提取内容
    test_contents = [
        ExtractedContent(
            paper_info={
                'title': 'Attention Is All You Need',
                'authors': ['Vaswani', 'Shazeer'],
                'year': '2017'
            },
            relevant_sections=['Transformer架构完全基于注意力机制'],
            key_sentences=[
                'Transformer使用多头注意力机制来处理序列',
                '自注意力允许模型关注输入序列的不同位置'
            ],
            confidence_score=0.9,
            extraction_method='test'
        ),
        ExtractedContent(
            paper_info={
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                'authors': ['Devlin', 'Chang'],
                'year': '2019'
            },
            relevant_sections=['BERT使用双向Transformer编码器'],
            key_sentences=[
                'BERT通过掩码语言模型预训练Transformer',
                '双向注意力机制提高了语言理解能力'
            ],
            confidence_score=0.8,
            extraction_method='test'
        )
    ]
    
    integrator = ContentIntegrator()
    
    # 测试简单整合（不依赖LLM）
    print("\n📝 测试简单内容整合...")
    result = integrator.generate_expanded_content(
        original_context="我们正在研究Transformer架构中注意力机制的工作原理",
        target_concept="注意力机制",
        extracted_contents=test_contents
    )
    
    if result:
        print("✅ 整合成功!")
        print(f"   整合方法: {result.integration_method}")
        print(f"   质量分数: {result.quality_score:.3f}")
        print(f"   源文献数: {len(result.source_papers)}")
        print(f"   关键要点数: {len(result.key_points)}")
        print(f"   内容长度: {len(result.expanded_content)}")
        print("\n📄 扩展内容预览:")
        print(result.expanded_content[:200] + "..." if len(result.expanded_content) > 200 else result.expanded_content)
        
        if result.key_points:
            print("\n🔑 关键要点:")
            for i, point in enumerate(result.key_points[:3], 1):
                print(f"   {i}. {point}")
        
        return True
    else:
        print("❌ 整合失败")
        return False


if __name__ == "__main__":
    test_content_integrator()
