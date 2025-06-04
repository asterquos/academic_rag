"""
src/analysis/author_analyzer.py
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import logging
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AuthorAnalyzer:
    """优化的作者分析器 - 批量加载模式"""

    def __init__(self, rag_engine=None, lazy_load=True):
        """
        初始化作者分析器

        Args:
            rag_engine: RAG引擎实例
            lazy_load: 是否延迟加载（只在需要时构建索引）
        """
        self.rag_engine = rag_engine
        self.author_index = {}  # 作者名 -> 文档列表
        self.author_variations = {}  # 处理作者名的不同写法
        self._index_built = False

        if not lazy_load:
            self._build_author_index()

    def _ensure_index(self):
        """确保索引已构建"""
        if not self._index_built:
            logger.info("首次使用，构建作者索引...")
            self._build_author_index()

    def _build_author_index(self):
        """优化的批量构建作者索引"""
        if not self.rag_engine:
            logger.error("RAG引擎未初始化")
            return

        logger.info("开始构建作者索引（优化版）...")

        try:
            # 方法1：尝试直接从向量存储获取所有文档
            if hasattr(self.rag_engine.vector_store, 'get_all') or hasattr(self.rag_engine.vector_store, 'collection'):
                logger.info("尝试批量获取所有文档...")
                all_docs = self._get_all_documents_batch()
            else:
                # 方法2：通过少量高频词获取大部分文档
                logger.info("使用高频词搜索获取文档...")
                all_docs = self._get_documents_by_common_terms()

            logger.info(f"总共获取到 {len(all_docs)} 个文档")

            # 批量处理文档
            self._process_documents_batch(all_docs)

            logger.info(f"作者索引构建完成: {len(self.author_index)} 个作者")
            self._index_built = True

        except Exception as e:
            logger.error(f"构建作者索引失败: {e}")
            self._index_built = False

    def _get_all_documents_batch(self) -> Dict[str, Dict]:
        """尝试批量获取所有文档"""
        all_docs = {}

        try:
            # 对于ChromaDB，尝试获取所有ID然后批量获取
            if hasattr(self.rag_engine.vector_store, 'collection'):
                # 获取collection中的所有文档
                collection = self.rag_engine.vector_store.collection

                # 尝试获取所有文档（ChromaDB的方式）
                try:
                    # 方法1: 使用get()获取所有
                    results = collection.get()

                    if results and 'ids' in results:
                        for i, doc_id in enumerate(results['ids']):
                            doc = {
                                'id': doc_id,
                                'text': results['documents'][i] if 'documents' in results else '',
                                'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                            }
                            all_docs[doc_id] = doc

                except Exception as e:
                    logger.warning(f"直接获取失败，尝试查询方式: {e}")

                    # 方法2: 使用大量查询
                    results = collection.query(
                        query_embeddings=[[0.0] * self.rag_engine.embedding_model.embedding_dim],
                        n_results=10000  # 获取尽可能多的结果
                    )

                    if results and 'ids' in results and len(results['ids']) > 0:
                        for i, doc_id in enumerate(results['ids'][0]):
                            doc = {
                                'id': doc_id,
                                'text': results['documents'][0][i] if 'documents' in results else '',
                                'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                            }
                            all_docs[doc_id] = doc

        except Exception as e:
            logger.warning(f"批量获取失败: {e}")

        return all_docs

    def _get_documents_by_common_terms(self) -> Dict[str, Dict]:
        """通过常见词获取文档"""
        # 使用更少但更通用的搜索词
        search_terms = ['的', '是', '在', '和', '了', '有', '年', '与', '为']

        all_docs = {}

        for term in search_terms:
            try:
                results = self.rag_engine.search(term, top_k=500)

                for doc in results:
                    doc_id = doc.get('id', '')
                    if doc_id and doc_id not in all_docs:
                        all_docs[doc_id] = doc

                logger.info(f"搜索词 '{term}' 获得 {len(results)} 个结果，累计 {len(all_docs)} 个唯一文档")

                # 如果已经获得足够多文档，提前结束
                if len(all_docs) > 5000:
                    logger.info("已获得足够文档，停止搜索")
                    break

            except Exception as e:
                logger.error(f"搜索失败 '{term}': {e}")

        return all_docs

    def _process_documents_batch(self, all_docs: Dict[str, Dict]):
        """批量处理文档构建索引"""
        logger.info("批量处理文档构建作者索引...")

        # 使用进度条
        for doc_id, doc in tqdm(all_docs.items(), desc="处理文档"):
            try:
                self._process_document_authors(doc)
            except Exception as e:
                logger.debug(f"处理文档 {doc_id} 失败: {e}")

    def _process_document_authors(self, doc: Dict):
        """处理单个文档的作者信息"""
        metadata = doc.get('metadata', {})

        # 获取作者字段
        author_field = metadata.get('作者名称', '')

        # 处理各种可能的空值
        if not author_field or pd.isna(author_field):
            return

        author_field = str(author_field).strip()
        if not author_field or author_field.lower() in ['nan', 'none', '']:
            return

        # 分割多个作者
        authors = self._split_author_string(author_field)

        # 为每个作者建立索引
        for author in authors:
            if author:
                # 标准化作者名
                normalized_author = self._normalize_author_name(author)

                if normalized_author not in self.author_index:
                    self.author_index[normalized_author] = []

                self.author_index[normalized_author].append(doc)

                # 记录作者名的变体
                if normalized_author not in self.author_variations:
                    self.author_variations[normalized_author] = set()
                self.author_variations[normalized_author].add(author)

    def _split_author_string(self, author_string: str) -> List[str]:
        """分割作者字符串"""
        # 常见的分隔符
        separators = ['，', ',', '、', ';', '；', ' and ', ' & ', '等', '　']

        authors = [author_string]

        # 逐步分割
        for sep in separators:
            new_authors = []
            for author in authors:
                if sep in author:
                    parts = author.split(sep)
                    new_authors.extend(parts)
                else:
                    new_authors.append(author)
            authors = new_authors

        # 清理每个作者名
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if author and len(author) >= 2:  # 至少2个字符
                cleaned_authors.append(author)

        return cleaned_authors

    def _normalize_author_name(self, author: str) -> str:
        """标准化作者名"""
        # 移除称谓
        titles = ['教授', '博士', '先生', '女士', '老师', 'Prof.', 'Dr.', 'Mr.', 'Ms.']
        for title in titles:
            if author.endswith(title):
                author = author[:-len(title)]

        # 去除额外的空格和标点
        author = re.sub(r'\s+', '', author)  # 移除所有空格
        author = author.strip('.,;:')  # 移除标点符号

        return author

    def find_author(self, query_author: str) -> Tuple[Optional[str], List[Dict], float]:
        """查找作者及其文档"""
        self._ensure_index()

        if not self.author_index:
            logger.warning("作者索引为空")
            return None, [], 0.0

        # 标准化查询作者名
        normalized_query = self._normalize_author_name(query_author)

        # 1. 精确匹配
        if normalized_query in self.author_index:
            docs = self.author_index[normalized_query]
            return normalized_query, docs, 1.0

        # 2. 模糊匹配
        best_match = None
        best_score = 0.0
        best_docs = []

        for indexed_author in self.author_index.keys():
            score = self._calculate_author_similarity(normalized_query, indexed_author)
            if score > best_score and score > 0.6:  # 相似度阈值
                best_score = score
                best_match = indexed_author
                best_docs = self.author_index[indexed_author]

        if best_match:
            return best_match, best_docs, best_score

        return None, [], 0.0

    def _calculate_author_similarity(self, name1: str, name2: str) -> float:
        """计算作者名相似度"""
        if name1 == name2:
            return 1.0

        # 字符级别的相似度
        from difflib import SequenceMatcher
        char_sim = SequenceMatcher(None, name1, name2).ratio()

        # 长度相似度
        len_diff = abs(len(name1) - len(name2))
        max_len = max(len(name1), len(name2))
        len_sim = 1 - (len_diff / max_len) if max_len > 0 else 0

        # 子串包含
        substring_sim = 0
        if name1 in name2 or name2 in name1:
            min_len = min(len(name1), len(name2))
            substring_sim = min_len / max_len if max_len > 0 else 0

        # 综合计算
        final_sim = (char_sim * 0.4 + len_sim * 0.3 + substring_sim * 0.3)
        return final_sim

    def analyze_author(self, author_name: str) -> Dict:
        """分析作者的研究情况"""
        self._ensure_index()

        # 查找作者
        matched_author, docs, confidence = self.find_author(author_name)

        if not docs:
            return {
                'status': 'not_found',
                'query_author': author_name,
                'message': f"未找到作者 '{author_name}' 的文档"
            }

        # 分析文档
        analysis = {
            'status': 'found',
            'query_author': author_name,
            'matched_author': matched_author,
            'match_confidence': confidence,
            'total_publications': len(docs),
            'publications': [],
            'year_distribution': Counter(),
            'category_distribution': Counter(),
            'collaborators': Counter(),
            'research_topics': Counter(),
            'journal_distribution': Counter()
        }

        # 处理每个文档
        for doc in docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')

            # 基本信息
            pub = {
                'title': metadata.get('文章名称+副标题', '无标题'),
                'year': metadata.get('年份'),
                'category': metadata.get('分类', ''),
                'journal': metadata.get('刊号', ''),
                'authors': metadata.get('作者名称', ''),
                'content_preview': text[:200] + '...' if text else ''
            }
            analysis['publications'].append(pub)

            # 统计信息
            if pub['year'] and not pd.isna(pub['year']):
                analysis['year_distribution'][int(pub['year'])] += 1

            if pub['category']:
                analysis['category_distribution'][pub['category']] += 1

            if pub['journal']:
                analysis['journal_distribution'][pub['journal']] += 1

            # 提取合作者
            if pub['authors']:
                authors = self._split_author_string(pub['authors'])
                for co_author in authors:
                    normalized_co = self._normalize_author_name(co_author)
                    if normalized_co != matched_author and normalized_co != author_name:
                        analysis['collaborators'][normalized_co] += 1

            # 提取研究主题
            self._extract_research_topics(pub['title'] + ' ' + text, analysis['research_topics'])

        # 转换Counter为普通dict并排序
        analysis['year_distribution'] = dict(sorted(analysis['year_distribution'].items()))
        analysis['category_distribution'] = dict(analysis['category_distribution'].most_common())
        analysis['collaborators'] = dict(analysis['collaborators'].most_common(10))
        analysis['research_topics'] = dict(analysis['research_topics'].most_common(10))
        analysis['journal_distribution'] = dict(analysis['journal_distribution'].most_common(10))

        return analysis

    def _extract_research_topics(self, text: str, topic_counter: Counter):
        """提取研究主题关键词"""
        if not text:
            return

        # 艺术设计相关关键词
        keywords = [
            '包豪斯', '现代主义', '后现代主义', '极简主义', '构成主义',
            '装饰艺术', '工艺美术', '新艺术运动', '风格派', '立体主义',
            '设计理念', '设计方法', '设计教育', '建筑设计', '平面设计',
            '工业设计', '环境设计', '视觉传达', '产品设计', '空间设计',
            '色彩', '造型', '材料', '工艺', '技术', '创新', '传统',
            '美学', '功能', '形式', '结构', '比例', '节奏', '韵律'
        ]

        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text:
                topic_counter[keyword] += 1

    def get_author_list(self, limit: int = 50) -> List[Tuple[str, int]]:
        """获取作者列表"""
        self._ensure_index()

        if not self.author_index:
            return []

        # 按文档数排序
        author_stats = [(author, len(docs)) for author, docs in self.author_index.items()]
        author_stats.sort(key=lambda x: x[1], reverse=True)

        return author_stats[:limit]

    def search_authors(self, pattern: str) -> List[Tuple[str, int]]:
        """搜索匹配的作者"""
        self._ensure_index()

        if not self.author_index:
            return []

        pattern_lower = pattern.lower()
        matches = []

        for author, docs in self.author_index.items():
            # 检查作者名是否包含搜索模式
            if pattern_lower in author.lower():
                matches.append((author, len(docs)))
            else:
                # 检查作者名变体
                for variant in self.author_variations.get(author, set()):
                    if pattern_lower in variant.lower():
                        matches.append((author, len(docs)))
                        break

        # 按文档数排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_statistics(self) -> Dict:
        """获取统计信息 - 快速版本"""
        # 如果索引未构建，返回基本信息
        if not self._index_built:
            return {
                'total_authors': 0,
                'total_documents': 0,
                'avg_docs_per_author': 0,
                'max_docs_by_author': 0,
                'min_docs_by_author': 0,
                'authors_with_multiple_docs': 0,
                'index_built': False
            }

        # 如果索引已构建，返回详细统计
        if not self.author_index:
            return {
                'total_authors': 0,
                'total_documents': 0,
                'avg_docs_per_author': 0,
                'max_docs_by_author': 0,
                'min_docs_by_author': 0,
                'authors_with_multiple_docs': 0,
                'index_built': True
            }

        total_docs = sum(len(docs) for docs in self.author_index.values())
        doc_counts = [len(docs) for docs in self.author_index.values()]

        return {
            'total_authors': len(self.author_index),
            'total_documents': total_docs,
            'avg_docs_per_author': total_docs / len(self.author_index) if self.author_index else 0,
            'max_docs_by_author': max(doc_counts) if doc_counts else 0,
            'min_docs_by_author': min(doc_counts) if doc_counts else 0,
            'authors_with_multiple_docs': sum(1 for count in doc_counts if count > 1),
            'index_built': True
        }