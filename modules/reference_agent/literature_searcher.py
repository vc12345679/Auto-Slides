#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Literature Searcher Module for Reference Agent
文献检索模块，集成Semantic Scholar和arXiv API
"""

import re
import time
import logging
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .citation_extractor import Citation


@dataclass 
class PaperResult:
    """论文检索结果"""
    paper_id: str = ""
    title: str = ""
    authors: List[str] = None
    year: str = ""
    abstract: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    arxiv_id: str = ""
    pdf_url: str = ""
    confidence_score: float = 0.0
    search_strategy: str = ""
    full_text: str = ""
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
    
    def has_pdf_access(self) -> bool:
        """是否有PDF全文访问权限"""
        return bool(self.pdf_url)
    
    def has_full_text(self) -> bool:
        """是否有全文内容"""
        return bool(self.full_text)
    
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return bool(self.title and (self.authors or self.year))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'abstract': self.abstract,
            'venue': self.venue,
            'doi': self.doi,
            'url': self.url,
            'arxiv_id': self.arxiv_id,
            'pdf_url': self.pdf_url,
            'confidence_score': self.confidence_score,
            'search_strategy': self.search_strategy
        }


class RateLimiter:
    """API速率限制器"""
    
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.call_times = []
    
    def __enter__(self):
        now = time.time()
        # 移除过期的调用记录
        self.call_times = [t for t in self.call_times if now - t < self.period]
        
        # 如果达到限制，等待
        if len(self.call_times) >= self.calls:
            sleep_time = self.period - (now - self.call_times[0]) + 1
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()
                self.call_times = [t for t in self.call_times if now - t < self.period]
        
        self.call_times.append(now)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class LiteratureCache:
    """文献缓存管理"""
    
    def __init__(self, cache_dir: str = "literature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def get_cache_key(self, citation: Citation) -> str:
        """生成缓存键"""
        key_str = f"{citation.authors}_{citation.title}_{citation.year}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[PaperResult]:
        """获取缓存的结果"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                import json
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return PaperResult(**data)
        except Exception as e:
            self.logger.warning(f"读取缓存失败: {e}")
        return None
    
    def store(self, cache_key: str, result: PaperResult):
        """存储结果到缓存"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            import json
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"存储缓存失败: {e}")


class SemanticScholarSearcher:
    """Semantic Scholar API搜索器"""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.rate_limiter = RateLimiter(calls=100, period=300)  # 100 calls per 5 min
        self.logger = logging.getLogger(__name__)
        
        # 请求参数配置（移除不支持的doi字段）
        self.search_fields = [
            'title', 'authors', 'year', 'abstract', 'venue', 
            'url', 'openAccessPdf', 'externalIds'
        ]
    
    def search(self, citation: Citation) -> Optional[PaperResult]:
        """搜索论文"""
        try:
            with self.rate_limiter:
                # 构建查询
                query = self._build_query(citation)
                
                # 发送请求
                response = requests.get(
                    f"{self.base_url}/paper/search",
                    params=query,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    papers = data.get('data', [])
                    
                    if papers:
                        # 找到最佳匹配
                        best_match = self._find_best_match(papers, citation)
                        if best_match:
                            return self._create_paper_result(best_match, citation)
                else:
                    self.logger.warning(f"Semantic Scholar API错误: {response.status_code}, 响应: {response.text[:200]}")
                    
        except Exception as e:
            self.logger.error(f"Semantic Scholar搜索失败: {e}")
        
        return None
    
    def _build_query(self, citation: Citation) -> Dict[str, Any]:
        """构建搜索查询"""
        query = {
            'fields': ','.join(self.search_fields),
            'limit': 10
        }
        
        # 优先使用标题搜索
        if citation.title and len(citation.title) > 10:
            query['query'] = citation.title
        elif citation.authors and citation.year:
            # 使用作者和年份
            author_name = citation.authors[0].split(',')[0] if citation.authors else ""
            query['query'] = f"{author_name} {citation.year}"
        else:
            # 最后使用作者名
            query['query'] = citation.authors[0] if citation.authors else ""
        
        return query
    
    def _find_best_match(self, papers: List[Dict], citation: Citation) -> Optional[Dict]:
        """找到最佳匹配的论文"""
        best_score = 0
        best_paper = None
        
        for paper in papers:
            score = self._calculate_match_score(paper, citation)
            if score > best_score:
                best_score = score
                best_paper = paper
        
        # 只返回置信度足够高的结果
        return best_paper if best_score > 0.3 else None
    
    def _calculate_match_score(self, paper: Dict, citation: Citation) -> float:
        """计算匹配分数"""
        score = 0.0
        
        # 标题相似度
        if citation.title and paper.get('title'):
            title_sim = self._string_similarity(citation.title.lower(), paper['title'].lower())
            score += title_sim * 0.5
        
        # 年份匹配
        if citation.year and paper.get('year'):
            if str(citation.year) == str(paper['year']):
                score += 0.3
        
        # 作者匹配
        if citation.authors and paper.get('authors'):
            author_sim = self._author_similarity(citation.authors, paper['authors'])
            score += author_sim * 0.2
        
        return score
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度（简化版）"""
        if not s1 or not s2:
            return 0.0
        
        # 移除标点符号，分词
        words1 = set(re.findall(r'\w+', s1.lower()))
        words2 = set(re.findall(r'\w+', s2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _author_similarity(self, authors1: List[str], authors2: List[Dict]) -> float:
        """计算作者相似度"""
        if not authors1 or not authors2:
            return 0.0
        
        # 提取姓氏
        surnames1 = set()
        for author in authors1:
            # 简单提取第一个和最后一个单词作为姓名
            parts = author.split()
            if parts:
                surnames1.add(parts[-1].lower())  # 姓氏通常在最后
        
        surnames2 = set()
        for author_dict in authors2:
            name = author_dict.get('name', '')
            parts = name.split()
            if parts:
                surnames2.add(parts[-1].lower())
        
        if not surnames1 or not surnames2:
            return 0.0
        
        intersection = len(surnames1 & surnames2)
        return intersection / min(len(surnames1), len(surnames2))
    
    def _create_paper_result(self, paper: Dict, citation: Citation) -> PaperResult:
        """创建论文结果对象"""
        # 提取作者名
        authors = []
        for author in paper.get('authors', []):
            authors.append(author.get('name', ''))
        
        # 提取PDF URL
        pdf_url = ""
        open_access = paper.get('openAccessPdf')
        if open_access and open_access.get('url'):
            pdf_url = open_access['url']
        
        # 提取arXiv ID
        arxiv_id = ""
        external_ids = paper.get('externalIds', {})
        if external_ids and external_ids.get('ArXiv'):
            arxiv_id = external_ids['ArXiv']
        
        return PaperResult(
            paper_id=paper.get('paperId', ''),
            title=paper.get('title', ''),
            authors=authors,
            year=str(paper.get('year', '')),
            abstract=paper.get('abstract', ''),
            venue=paper.get('venue', ''),
            doi=paper.get('doi', ''),
            url=paper.get('url', ''),
            arxiv_id=arxiv_id,
            pdf_url=pdf_url,
            confidence_score=self._calculate_match_score(paper, citation),
            search_strategy="semantic_scholar"
        )


class ArXivSearcher:
    """arXiv API搜索器"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limiter = RateLimiter(calls=30, period=60)  # 30 calls per minute
        self.logger = logging.getLogger(__name__)
    
    def search(self, citation: Citation) -> Optional[PaperResult]:
        """搜索arXiv论文"""
        try:
            with self.rate_limiter:
                # 如果已有arXiv ID，直接查询
                if citation.arxiv_id:
                    return self._search_by_arxiv_id(citation.arxiv_id)
                
                # 否则使用标题或作者搜索
                query = self._build_arxiv_query(citation)
                
                response = requests.get(
                    self.base_url,
                    params=query,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = self._parse_arxiv_response(response.text, citation)
                    return result
                else:
                    self.logger.warning(f"arXiv API错误: {response.status_code}")
                    
        except Exception as e:
            self.logger.error(f"arXiv搜索失败: {e}")
        
        return None
    
    def _search_by_arxiv_id(self, arxiv_id: str) -> Optional[PaperResult]:
        """根据arXiv ID搜索"""
        query = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        try:
            response = requests.get(self.base_url, params=query, timeout=30)
            if response.status_code == 200:
                result = self._parse_arxiv_response(response.text)
                # 直接通过ID找到的结果应该有高置信度
                if result:
                    result.confidence_score = 0.9  # 直接ID匹配给高置信度
                return result
        except Exception as e:
            self.logger.error(f"arXiv ID搜索失败: {e}")
        
        return None
    
    def _build_arxiv_query(self, citation: Citation) -> Dict[str, Any]:
        """构建arXiv查询"""
        query = {
            'max_results': 10,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        # 构建搜索字符串
        search_terms = []
        
        if citation.title and len(citation.title) > 10:
            # 清理标题，移除特殊字符
            clean_title = re.sub(r'[^\w\s]', ' ', citation.title)
            search_terms.append(f'ti:"{clean_title}"')
        
        if citation.authors:
            author = citation.authors[0].split(',')[0].strip()
            search_terms.append(f'au:"{author}"')
        
        if search_terms:
            query['search_query'] = ' AND '.join(search_terms)
        else:
            return None
        
        return query
    
    def _parse_arxiv_response(self, xml_text: str, citation: Citation = None) -> Optional[PaperResult]:
        """解析arXiv XML响应"""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml_text)
            
            # 找到第一个entry
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                result = self._parse_arxiv_entry(entry)
                
                # 如果提供了引用信息，计算匹配分数
                if citation:
                    score = self._calculate_arxiv_match_score(result, citation)
                    result.confidence_score = score
                    
                    # 只返回置信度足够的结果
                    if score > 0.3:
                        return result
                else:
                    return result
            
        except Exception as e:
            self.logger.error(f"解析arXiv响应失败: {e}")
        
        return None
    
    def _parse_arxiv_entry(self, entry) -> PaperResult:
        """解析单个arXiv条目"""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # 提取基本信息
        title = entry.find('atom:title', ns)
        title = title.text.strip() if title is not None else ""
        
        abstract = entry.find('atom:summary', ns)
        abstract = abstract.text.strip() if abstract is not None else ""
        
        published = entry.find('atom:published', ns)
        year = ""
        if published is not None:
            year = published.text[:4]  # 提取年份
        
        # 提取arXiv ID和PDF URL
        arxiv_id = ""
        pdf_url = ""
        
        for link in entry.findall('atom:link', ns):
            href = link.get('href', '')
            if 'arxiv.org/abs/' in href:
                arxiv_id = href.split('/')[-1]
            elif 'arxiv.org/pdf/' in href:
                pdf_url = href
        
        # 提取作者
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text.strip())
        
        return PaperResult(
            paper_id=arxiv_id,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            venue="arXiv",
            arxiv_id=arxiv_id,
            pdf_url=pdf_url,
            search_strategy="arxiv"
        )
    
    def _calculate_arxiv_match_score(self, result: PaperResult, citation: Citation) -> float:
        """计算arXiv结果的匹配分数"""
        score = 0.0
        
        # 标题相似度
        if citation.title and result.title:
            title_sim = self._string_similarity(citation.title.lower(), result.title.lower())
            score += title_sim * 0.6
        
        # 年份匹配
        if citation.year and result.year:
            if str(citation.year) == str(result.year):
                score += 0.3
        
        # 作者匹配
        if citation.authors and result.authors:
            author_sim = self._author_similarity(citation.authors, result.authors)
            score += author_sim * 0.1
        
        return score
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """字符串相似度计算"""
        if not s1 or not s2:
            return 0.0
        
        words1 = set(re.findall(r'\w+', s1.lower()))
        words2 = set(re.findall(r'\w+', s2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _author_similarity(self, authors1: List[str], authors2: List[str]) -> float:
        """作者相似度计算"""
        if not authors1 or not authors2:
            return 0.0
        
        surnames1 = set()
        for author in authors1:
            parts = author.split()
            if parts:
                surnames1.add(parts[-1].lower())
        
        surnames2 = set()
        for author in authors2:
            parts = author.split()
            if parts:
                surnames2.add(parts[-1].lower())
        
        if not surnames1 or not surnames2:
            return 0.0
        
        intersection = len(surnames1 & surnames2)
        return intersection / min(len(surnames1), len(surnames2))


class LiteratureSearcher:
    """文献检索器主类"""
    
    def __init__(self):
        self.searchers = [
            SemanticScholarSearcher(),
            ArXivSearcher()
        ]
        self.cache = LiteratureCache()
        self.logger = logging.getLogger(__name__)
    
    def search_paper(self, citation: Citation) -> Optional[PaperResult]:
        """搜索论文，使用级联策略"""
        self.logger.info(f"搜索论文: {citation.title} ({citation.year})")
        
        # 检查缓存
        cache_key = self.cache.get_cache_key(citation)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.logger.info("使用缓存结果")
            return cached_result
        
        # 按优先级尝试搜索
        best_result = None
        best_score = -1  # 改为-1，这样0.0的结果也能被选中
        
        for searcher in self.searchers:
            try:
                result = searcher.search(citation)
                if result and result.is_valid():
                    # 对于有效结果，如果置信度为0，给予基础分数
                    if result.confidence_score == 0.0:
                        result.confidence_score = 0.1  # 给予基础分数
                    
                    self.logger.info(f"{searcher.__class__.__name__} 找到结果，置信度: {result.confidence_score:.3f}")
                    
                    if result.confidence_score > best_score:
                        best_score = result.confidence_score
                        best_result = result
                    
                    # 如果找到高置信度结果，直接返回
                    if result.confidence_score > 0.8:
                        break
                        
            except Exception as e:
                self.logger.warning(f"{searcher.__class__.__name__} 搜索失败: {e}")
        
        # 缓存最佳结果
        if best_result:
            self.cache.store(cache_key, best_result)
            self.logger.info(f"最终选择: {best_result.search_strategy}, 置信度: {best_result.confidence_score:.3f}")
        
        return best_result
    
    def search_multiple_papers(self, citations: List[Citation], max_results: int = 5) -> List[PaperResult]:
        """批量搜索多篇论文"""
        results = []
        
        for citation in citations[:max_results]:
            result = self.search_paper(citation)
            if result:
                results.append(result)
            
            # 避免API限制，添加短暂延迟
            time.sleep(0.5)
        
        return results


# 测试函数
def test_literature_searcher():
    """测试文献检索器"""
    print("🔍 测试文献检索器...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试引用
    test_citation = Citation(
        anchor="test",
        authors=["OpenAI"],
        title="GPT-4 Technical Report",
        year="2024",
        venue="arXiv",
        arxiv_id="2303.08774"
    )
    
    # 先测试arXiv搜索器（更简单）
    print("\n🧪 测试arXiv搜索器...")
    arxiv_searcher = ArXivSearcher()
    arxiv_result = arxiv_searcher.search(test_citation)
    
    if arxiv_result:
        print("✅ arXiv搜索成功!")
        print(f"   标题: {arxiv_result.title}")
        print(f"   作者: {', '.join(arxiv_result.authors[:2])}...")
        print(f"   年份: {arxiv_result.year}")
        print(f"   arXiv ID: {arxiv_result.arxiv_id}")
        print(f"   有PDF: {'是' if arxiv_result.has_pdf_access() else '否'}")
    else:
        print("❌ arXiv搜索失败")
    
    # 再测试Semantic Scholar
    print("\n🧪 测试Semantic Scholar搜索器...")
    ss_searcher = SemanticScholarSearcher()
    
    # 创建一个更简单的测试
    simple_citation = Citation(
        anchor="test2",
        authors=["Wei"],
        title="Attention Is All You Need",
        year="2017",
        venue="NIPS"
    )
    
    ss_result = ss_searcher.search(simple_citation)
    
    if ss_result:
        print("✅ Semantic Scholar搜索成功!")
        print(f"   标题: {ss_result.title}")
        print(f"   作者: {', '.join(ss_result.authors[:2])}...")
        print(f"   年份: {ss_result.year}")
        print(f"   有PDF: {'是' if ss_result.has_pdf_access() else '否'}")
        return True
    else:
        print("❌ Semantic Scholar搜索失败")
        
    return arxiv_result is not None


if __name__ == "__main__":
    test_literature_searcher()
