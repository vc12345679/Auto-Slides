#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能图片匹配器
基于关键词匹配和规则优化算法为幻灯片智能分配图片
"""

import logging
from typing import Dict, List, Any, Optional, Tuple


class IntelligentFigureMatcher:
    """智能图片匹配器，用于优化图片和幻灯片的匹配"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 幻灯片类型关键词
        self.slide_type_keywords = {
            'background': ['background', 'introduction', 'motivation', 'problem', 'challenge'],
            'method': ['methodology', 'approach', 'method', 'algorithm', 'architecture', 'framework'],
            'result': ['result', 'experiment', 'evaluation', 'performance', 'comparison'],
            'ablation': ['ablation', 'analysis', 'component', 'impact'],
            'conclusion': ['conclusion', 'future', 'limitation', 'summary']
        }
        
        # 图片类型关键词（优先级从高到低）
        self.figure_type_keywords = {
            'method_architecture': {
                'keywords': ['architecture', 'framework', 'overview', 'system', 'model', 'illustration of our proposed', 'cross-modal', 'teacher model', 'style-based'],
                'priority': 5
            },
            'results_performance': {
                'keywords': ['comparison', 'performance', 'result', 'evaluation', 'quantitative', 'qualitative'],
                'priority': 4
            },
            'method_detail': {
                'keywords': ['visualization', 'attention', 'map', 'process', 'mechanism', 'detail'],
                'priority': 3
            },
            'results_examples': {
                'keywords': ['example', 'generated', 'output', 'synthesis', 'transfer'],
                'priority': 2
            },
            'problem_illustration': {
                'keywords': ['overfitting', 'artifact', 'problem', 'issue', 'challenge'],
                'priority': 1
            }
        }
    
    def classify_slide_type(self, slide: Dict[str, Any]) -> str:
        """分类幻灯片类型"""
        title = slide.get('title', '').lower()
        content = ' '.join(slide.get('content', [])).lower()
        text = f"{title} {content}"
        
        scores = {}
        for slide_type, keywords in self.slide_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[slide_type] = score
        
        return max(scores, key=scores.get) if scores else 'method'
    
    def classify_figure_type(self, figure: Dict[str, Any]) -> Tuple[str, float]:
        """分类图片类型并计算置信度"""
        caption = figure.get('caption', '').lower()
        description = figure.get('description', '').lower()
        text = f"{caption} {description}"
        
        best_type = None
        best_score = 0
        best_priority = 0
        
        for fig_type, config in self.figure_type_keywords.items():
            keywords = config['keywords']
            priority = config['priority']
            
            # 计算关键词匹配得分
            keyword_score = sum(1 for keyword in keywords if keyword in text)
            
            if keyword_score > 0:
                # 综合得分 = 关键词匹配数 + 优先级权重
                total_score = keyword_score + priority * 0.1
                
                if total_score > best_score or (total_score == best_score and priority > best_priority):
                    best_type = fig_type
                    best_score = total_score
                    best_priority = priority
        
        confidence = min(best_score / 3.0, 1.0) if best_score > 0 else 0.0
        return best_type or 'method_detail', confidence
    
    def calculate_compatibility_score(self, slide: Dict[str, Any], figure: Dict[str, Any]) -> float:
        """计算幻灯片和图片的兼容性得分"""
        slide_title = slide.get('title', '').lower()
        slide_content = ' '.join(slide.get('content', [])).lower()
        slide_text = f"{slide_title} {slide_content}"
        
        figure_caption = figure.get('caption', '').lower()
        figure_description = figure.get('description', '').lower()
        figure_text = f"{figure_caption} {figure_description}"
        
        score = 0.0
        
        # 1. 动态语义匹配（基于内容重叠）
        # 提取重要的技术术语和概念
        slide_important_terms = self._extract_important_terms(slide_text)
        figure_important_terms = self._extract_important_terms(figure_text)
        
        # 计算重要术语的重叠度
        common_important_terms = slide_important_terms & figure_important_terms
        if common_important_terms:
            # 根据共同重要术语数量计算语义匹配得分
            semantic_score = min(len(common_important_terms) * 0.15, 0.4)
            score += semantic_score
        
        # 2. 类型匹配
        slide_type = self.classify_slide_type(slide)
        figure_type, figure_confidence = self.classify_figure_type(figure)
        
        # 添加严格的语义排斥规则
        semantic_exclusions = [
            # 消融实验不应该匹配工作流程演化图
            ('ablation', 'workflow', 'evolution'),
            # 消融实验不应该匹配迭代过程图  
            ('ablation', 'iteration', 'process'),
            # 消融实验应该匹配性能对比，而不是流程图
            ('ablation', 'over', 'iterations')
        ]
        
        # 检查语义排斥
        for slide_keyword, fig_keyword1, fig_keyword2 in semantic_exclusions:
            if (slide_keyword in slide_text and 
                fig_keyword1 in figure_text and 
                fig_keyword2 in figure_text):
                score -= 0.3  # 严重惩罚不合适的匹配
        
        type_compatibility = {
            ('method', 'method_architecture'): 0.3,
            ('method', 'method_detail'): 0.25,
            ('result', 'results_performance'): 0.3,
            ('result', 'results_examples'): 0.25,
            ('background', 'problem_illustration'): 0.2,
            ('ablation', 'results_performance'): 0.15
        }
        
        type_match = type_compatibility.get((slide_type, figure_type), 0.05)
        score += type_match * figure_confidence
        
        # 3. 通用关键词重叠
        slide_words = set(slide_text.split())
        figure_words = set(figure_text.split())
        common_words = slide_words & figure_words
        significant_words = common_words - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        if significant_words:
            overlap_score = min(len(significant_words) * 0.05, 0.2)
            score += overlap_score
        
        # 4. 架构图优先级加成
        if 'architecture' in figure_text or 'framework' in figure_text or 'overview' in figure_text:
            if slide_type == 'method':
                score += 0.1
        
        return min(score, 1.0)
    
    def _extract_important_terms(self, text: str) -> set:
        """提取文本中的重要技术术语"""
        import re
        
        # 移除常见停用词
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'we', 'our', 'use', 'using', 'show', 'shows', 'paper', 'figure', 'table', 'results', 'method'
        }
        
        # 提取重要术语的规则
        important_terms = set()
        
        # 1. 技术缩写词（2-6个大写字母）
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text.upper())
        important_terms.update(term.lower() for term in acronyms)
        
        # 2. 技术术语模式（含连字符的复合词）
        compound_terms = re.findall(r'\b\w+[-]\w+(?:[-]\w+)*\b', text)
        important_terms.update(term.lower() for term in compound_terms)
        
        # 3. 专业词汇（特定领域的关键词）
        domain_keywords = {
            'algorithm', 'model', 'architecture', 'framework', 'network', 'learning', 'training',
            'optimization', 'performance', 'accuracy', 'evaluation', 'dataset', 'benchmark',
            'attention', 'transformer', 'convolution', 'neural', 'deep', 'machine', 'artificial',
            'classification', 'regression', 'clustering', 'segmentation', 'detection', 'recognition',
            'feature', 'embedding', 'representation', 'encoding', 'decoding', 'generation',
            'medical', 'clinical', 'diagnosis', 'treatment', 'patient', 'healthcare', 'therapy',
            'agent', 'workflow', 'system', 'pipeline', 'process', 'methodology', 'approach'
        }
        
        # 4. 提取文本中的专业词汇
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if len(word) >= 4 and word in domain_keywords:
                important_terms.add(word)
        
        # 5. 移除停用词
        important_terms = important_terms - stop_words
        
        return important_terms
    
    def detect_architecture_figures(self, figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测架构图"""
        architecture_figures = []
        for figure in figures:
            caption = figure.get('caption', '').lower()
            description = figure.get('description', '').lower()
            text = f"{caption} {description}"
            
            arch_keywords = ['architecture', 'framework', 'overview', 'system', 'illustration of our proposed']
            if any(keyword in text for keyword in arch_keywords):
                architecture_figures.append(figure)
        
        return architecture_figures
    
    def optimize_figure_assignment(self, slides: List[Dict[str, Any]], figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优化图片分配"""
        if not figures:
            self.logger.warning("没有可用图片进行匹配")
            return slides
        
        logger = self.logger
        logger.info(f"开始智能图片匹配，共{len(slides)}张幻灯片，{len(figures)}张图片")
        
        # 1. 检测架构图
        architecture_figures = self.detect_architecture_figures(figures)
        logger.info(f"检测到{len(architecture_figures)}张架构图")
        
        # 2. 计算兼容性矩阵
        compatibility_matrix = {}
        used_figures = set()
        
        for i, slide in enumerate(slides):
            slide_scores = []
            for j, figure in enumerate(figures):
                score = self.calculate_compatibility_score(slide, figure)
                slide_scores.append((j, figure, score))
            
            # 按得分排序
            slide_scores.sort(key=lambda x: x[2], reverse=True)
            compatibility_matrix[i] = slide_scores
            logger.info(f"幻灯片{i+1}的最佳匹配: {slide_scores[0][1]['id']} (得分: {slide_scores[0][2]:.3f})")
        
        # 3. 贪心算法进行最优分配
        optimized_slides = slides.copy()
        
        # 首先分配架构图（优先级最高）
        for arch_fig in architecture_figures:
            best_slide_idx = None
            best_score = 0
            
            for i, slide in enumerate(optimized_slides):
                slide_type = self.classify_slide_type(slide)
                if slide_type == 'method':  # 架构图优先分配给方法类幻灯片
                    score = self.calculate_compatibility_score(slide, arch_fig)
                    if score > best_score and arch_fig['id'] not in used_figures:
                        best_score = score
                        best_slide_idx = i
            
            if best_slide_idx is not None:
                optimized_slides[best_slide_idx]['includes_figure'] = True
                optimized_slides[best_slide_idx]['figure_reference'] = {
                    'id': arch_fig['id'],
                    'caption': arch_fig['caption'],
                    'filename': arch_fig.get('filename', ''),
                    'path': arch_fig.get('path', '')
                }
                used_figures.add(arch_fig['id'])
                logger.info(f"架构图 {arch_fig['id']} 分配给幻灯片 {best_slide_idx + 1}，得分: {best_score:.3f}")
        
        # 4. 设置匹配阈值，平衡质量和数量
        MIN_COMPATIBILITY_SCORE = 0.25  # 较低阈值，增加图片数量，适度降低质量要求
        
        # 清除所有幻灯片的图片分配（保留架构图分配）
        for i, slide in enumerate(optimized_slides):
            fig_ref = slide.get('figure_reference')
            current_fig_id = fig_ref.get('id', '') if fig_ref else ''
            if current_fig_id not in used_figures:  # 如果不是架构图
                optimized_slides[i]['includes_figure'] = False
                optimized_slides[i]['figure_reference'] = None
        
        # 为每张未使用的图片找到最佳匹配的幻灯片
        for figure in figures:
            if figure['id'] not in used_figures:
                best_slide_idx = None
                best_score = MIN_COMPATIBILITY_SCORE
                
                for i, slide in enumerate(optimized_slides):
                    # 跳过已经有图片的幻灯片
                    if optimized_slides[i].get('includes_figure', False):
                        continue
                        
                    score = self.calculate_compatibility_score(slide, figure)
                    if score > best_score:
                        best_score = score
                        best_slide_idx = i
                
                # 如果找到合适的幻灯片，分配图片
                if best_slide_idx is not None:
                    optimized_slides[best_slide_idx]['includes_figure'] = True
                    optimized_slides[best_slide_idx]['figure_reference'] = {
                        'id': figure['id'],
                        'caption': figure['caption'],  # 保持原始caption，不添加description避免错误
                        'filename': figure.get('filename', ''),
                        'path': figure.get('path', '')
                    }
                    used_figures.add(figure['id'])
                    logger.info(f"图片 {figure['id']} 分配给幻灯片 {best_slide_idx + 1}，得分: {best_score:.3f}")
                else:
                    logger.info(f"图片 {figure['id']} 没有找到合适的幻灯片（最高得分 < {MIN_COMPATIBILITY_SCORE}）")
        
        logger.info(f"智能图片匹配完成，共使用{len(used_figures)}张图片")
        return optimized_slides
