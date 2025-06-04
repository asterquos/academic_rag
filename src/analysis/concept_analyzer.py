"""
src/analysis/concept_analyzer.py
优化的概念分析器 - 快速启动版本
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from datetime import datetime
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


class ConceptAnalyzer:
    """优化的概念分析器 - 使用缓存和优化搜索策略"""

    def __init__(self, rag_engine=None):
        """
        初始化概念分析器

        Args:
            rag_engine: RAG引擎实例
        """
        self.rag_engine = rag_engine
        self.concept_cache = {}
        self._search_cache = {}  # 缓存搜索结果
        self._max_cache_size = 100  # 最大缓存大小

        # 预定义的艺术设计概念
        self.known_concepts = {
            '包豪斯', '现代主义', '后现代主义', '极简主义', '构成主义',
            '装饰艺术', '工艺美术', '新艺术运动', '风格派', '立体主义',
            '未来主义', '达达主义', '超现实主义', '抽象表现主义',
            '设计', '艺术', '建筑', '美学', '功能主义', '形式主义'
        }

    @lru_cache(maxsize=128)
    def _get_cache_key(self, concept: str, verify: bool = True) -> str:
        """生成缓存键"""
        return f"{concept.lower()}_{verify}"

    def search_concept_documents(self, concept: str, verify_presence: bool = True) -> List[Dict]:
        """
        搜索包含概念的文档 - 优化版本

        Args:
            concept: 概念关键词
            verify_presence: 是否验证概念确实在文档中出现

        Returns:
            相关文档列表
        """
        if not self.rag_engine:
            logger.error("RAG引擎未初始化")
            return []

        # 检查缓存
        cache_key = self._get_cache_key(concept, verify_presence)
        if cache_key in self._search_cache:
            logger.info(f"从缓存返回概念 '{concept}' 的搜索结果")
            return self._search_cache[cache_key]

        logger.info(f"搜索概念: {concept}")

        # 优化的搜索策略：使用更少但更精确的查询
        search_queries = [concept]  # 首先直接搜索概念本身

        # 只对较长的概念添加变体
        if len(concept) > 2:
            search_queries.extend([
                f"{concept}的",
                f"{concept}概念"
            ])

        all_docs = {}
        max_results_per_query = 50  # 限制每个查询的结果数

        for query in search_queries:
            try:
                logger.info(f"执行搜索: {query}")
                results = self.rag_engine.search(query, top_k=max_results_per_query)

                valid_results = 0
                for doc in results:
                    doc_id = doc.get('id', '')
                    if doc_id and doc_id not in all_docs:
                        # 快速验证（可选）
                        if verify_presence:
                            if self._verify_concept_in_document_fast(concept, doc):
                                all_docs[doc_id] = doc
                                valid_results += 1
                        else:
                            all_docs[doc_id] = doc
                            valid_results += 1

                logger.info(f"查询 '{query}' 找到 {valid_results} 个有效结果")

                # 如果已经找到足够的文档，提前结束
                if len(all_docs) >= 100:
                    logger.info(f"已找到足够文档，停止搜索")
                    break

            except Exception as e:
                logger.error(f"搜索查询 '{query}' 失败: {e}")

        final_docs = list(all_docs.values())
        logger.info(f"概念 '{concept}' 最终找到 {len(final_docs)} 个相关文档")

        # 更新缓存（限制缓存大小）
        if len(self._search_cache) >= self._max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]

        self._search_cache[cache_key] = final_docs

        return final_docs

    def _verify_concept_in_document_fast(self, concept: str, doc: Dict) -> bool:
        """快速验证概念是否在文档中出现"""
        # 只检查文档文本，不检查标题（减少字符串操作）
        text = doc.get('text', '').lower()
        concept_lower = concept.lower()

        # 直接返回结果，不做复杂的部分匹配
        return concept_lower in text

    def find_first_appearance(self, concept: str) -> Dict:
        """
        查找概念的首次出现 - 优化版本

        Args:
            concept: 要查找的概念

        Returns:
            首次出现信息
        """
        logger.info(f"查找概念 '{concept}' 的首次出现")

        # 检查结果缓存
        cache_key = f"first_{concept}"
        if cache_key in self.concept_cache:
            logger.info(f"从缓存返回概念 '{concept}' 的首次出现")
            return self.concept_cache[cache_key]

        # 获取相关文档（限制数量以提高性能）
        docs = self.search_concept_documents(concept, verify_presence=True)

        if not docs:
            result = {
                'status': 'not_found',
                'concept': concept,
                'message': f"未找到包含概念 '{concept}' 的文档"
            }
            self.concept_cache[cache_key] = result
            return result

        # 快速筛选有效年份的文档
        docs_with_year = []
        for doc in docs:
            metadata = doc.get('metadata', {})
            year = metadata.get('年份')

            if year and not pd.isna(year):
                try:
                    year_int = int(year)
                    if 1800 <= year_int <= 2030:
                        docs_with_year.append((year_int, doc))
                except (ValueError, TypeError):
                    continue

        if not docs_with_year:
            result = {
                'status': 'no_valid_date',
                'concept': concept,
                'total_docs': len(docs),
                'message': f"找到 {len(docs)} 个相关文档，但都缺少有效年份信息"
            }
            self.concept_cache[cache_key] = result
            return result

        # 找到最早的年份
        docs_with_year.sort(key=lambda x: x[0])
        earliest_year, earliest_doc = docs_with_year[0]

        # 提取上下文
        context = self._extract_concept_context_fast(concept, earliest_doc)

        result = {
            'status': 'found',
            'concept': concept,
            'year': earliest_year,
            'title': earliest_doc.get('metadata', {}).get('文章名称+副标题', '未知'),
            'author': earliest_doc.get('metadata', {}).get('作者名称', '未知'),
            'category': earliest_doc.get('metadata', {}).get('分类', ''),
            'context': context,
            'doc_id': earliest_doc.get('id', ''),
            'total_docs_found': len(docs),
            'docs_with_valid_year': len(docs_with_year)
        }

        self.concept_cache[cache_key] = result
        return result

    def _extract_concept_context_fast(self, concept: str, doc: Dict, window_size: int = 150) -> str:
        """快速提取概念周围的上下文"""
        text = doc.get('text', '')
        title = doc.get('metadata', {}).get('文章名称+副标题', '')

        if not text:
            return f"标题: {title}"

        # 简化的上下文提取
        concept_lower = concept.lower()
        text_lower = text.lower()

        pos = text_lower.find(concept_lower)
        if pos == -1:
            # 没找到，返回文档开头
            return f"标题: {title}\n内容: {text[:window_size*2]}..."

        # 提取上下文
        start = max(0, pos - window_size)
        end = min(len(text), pos + len(concept) + window_size)
        context = text[start:end]

        # 添加省略号
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return f"标题: {title}\n上下文: {context}"

    def analyze_temporal_distribution(self, concept: str) -> pd.DataFrame:
        """
        分析概念的时间分布 - 优化版本

        Args:
            concept: 要分析的概念

        Returns:
            包含年份分布的DataFrame
        """
        logger.info(f"分析概念 '{concept}' 的时间分布")

        # 检查缓存
        cache_key = f"temporal_{concept}"
        if cache_key in self.concept_cache:
            cached_data = self.concept_cache[cache_key]
            if isinstance(cached_data, list):
                return pd.DataFrame(cached_data)

        # 获取相关文档（使用缓存的结果）
        docs = self.search_concept_documents(concept, verify_presence=True)

        if not docs:
            return pd.DataFrame()

        # 快速统计年份分布
        year_counts = Counter()

        for doc in docs:
            metadata = doc.get('metadata', {})
            year = metadata.get('年份')

            if year and not pd.isna(year):
                try:
                    year_int = int(year)
                    if 1800 <= year_int <= 2030:
                        year_counts[year_int] += 1
                except (ValueError, TypeError):
                    continue

        if not year_counts:
            return pd.DataFrame()

        # 创建DataFrame
        df = pd.DataFrame(
            list(year_counts.items()),
            columns=['year', 'count']
        ).sort_values('year')

        # 计算百分比
        total_count = df['count'].sum()
        df['percentage'] = (df['count'] / total_count * 100).round(2)
        df['cumulative_percentage'] = df['percentage'].cumsum().round(2)

        # 缓存结果
        self.concept_cache[cache_key] = df.to_dict('records')

        return df

    def find_related_concepts(self, concept: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        查找相关概念 - 优化版本

        Args:
            concept: 目标概念
            top_n: 返回的相关概念数量

        Returns:
            相关概念列表 [(概念, 共现次数), ...]
        """
        logger.info(f"查找与 '{concept}' 相关的概念")

        # 检查缓存
        cache_key = f"related_{concept}_{top_n}"
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]

        # 获取包含目标概念的文档（限制数量）
        docs = self.search_concept_documents(concept, verify_presence=True)[:50]

        if not docs:
            return []

        # 统计其他概念的共现
        concept_cooccurrence = Counter()

        for doc in docs:
            text = doc.get('text', '')[:1000]  # 只检查前1000字符
            text_lower = text.lower()

            # 只检查预定义的概念（避免过度计算）
            for other_concept in self.known_concepts:
                if other_concept.lower() != concept.lower():
                    if other_concept.lower() in text_lower:
                        concept_cooccurrence[other_concept] += 1

        # 获取结果
        result = concept_cooccurrence.most_common(top_n)

        # 缓存结果
        self.concept_cache[cache_key] = result

        return result

    def analyze_concept_evolution(self, concept: str, time_window: int = 5) -> Dict:
        """
        分析概念的演进历程 - 优化版本

        Args:
            concept: 要分析的概念
            time_window: 时间窗口大小（年）

        Returns:
            演进分析结果
        """
        logger.info(f"分析概念 '{concept}' 的演进历程")

        # 获取时间分布（使用缓存）
        temporal_df = self.analyze_temporal_distribution(concept)

        if temporal_df.empty:
            return {
                'status': 'no_data',
                'concept': concept,
                'message': f"无法获取概念 '{concept}' 的时间分布数据"
            }

        # 基础统计
        min_year = int(temporal_df['year'].min())
        max_year = int(temporal_df['year'].max())
        total_mentions = int(temporal_df['count'].sum())
        peak_idx = temporal_df['count'].idxmax()
        peak_year = int(temporal_df.loc[peak_idx, 'year'])
        peak_count = int(temporal_df['count'].max())

        # 分析趋势
        trend = self._analyze_trend_fast(temporal_df)

        # 按时间窗口分析（简化版）
        periods = []
        for start_year in range(min_year, max_year + 1, time_window):
            end_year = min(start_year + time_window - 1, max_year)

            period_data = temporal_df[
                (temporal_df['year'] >= start_year) &
                (temporal_df['year'] <= end_year)
            ]

            if not period_data.empty:
                period_count = int(period_data['count'].sum())
                period_peak_idx = period_data['count'].idxmax()
                period_peak = int(period_data.loc[period_peak_idx, 'year'])

                periods.append({
                    'period': f"{start_year}-{end_year}",
                    'start_year': start_year,
                    'end_year': end_year,
                    'total_mentions': period_count,
                    'peak_year': period_peak,
                    'avg_mentions_per_year': round(period_count / len(period_data), 2)
                })

        return {
            'status': 'success',
            'concept': concept,
            'overview': {
                'first_year': min_year,
                'last_year': max_year,
                'span_years': max_year - min_year + 1,
                'total_mentions': total_mentions,
                'peak_year': peak_year,
                'peak_count': peak_count,
                'trend': trend
            },
            'periods': periods,
            'temporal_data': temporal_df.to_dict('records')
        }

    def _analyze_trend_fast(self, df: pd.DataFrame) -> str:
        """快速分析概念的整体趋势"""
        if len(df) < 3:
            return "数据不足"

        # 简单的趋势判断：比较前后期平均值
        mid_point = len(df) // 2
        early_avg = df.iloc[:mid_point]['count'].mean()
        late_avg = df.iloc[mid_point:]['count'].mean()

        if late_avg > early_avg * 1.2:
            return "上升趋势"
        elif late_avg < early_avg * 0.8:
            return "下降趋势"
        else:
            return "平稳"

    def generate_concept_report(self, concept: str) -> Dict:
        """
        生成概念的完整分析报告 - 优化版本

        Args:
            concept: 要分析的概念

        Returns:
            完整报告
        """
        logger.info(f"生成概念 '{concept}' 的完整报告")

        report = {
            'concept': concept,
            'generated_at': datetime.now().isoformat(),
            'status': 'success'
        }

        try:
            # 1. 首次出现分析（使用缓存）
            first_appearance = self.find_first_appearance(concept)
            report['first_appearance'] = first_appearance

            if first_appearance['status'] == 'not_found':
                report['status'] = 'not_found'
                return report

            # 2. 时间分布分析（使用缓存）
            temporal_df = self.analyze_temporal_distribution(concept)
            report['temporal_distribution'] = temporal_df.to_dict('records') if not temporal_df.empty else []

            # 3. 相关概念（使用缓存）
            related_concepts = self.find_related_concepts(concept, top_n=10)
            report['related_concepts'] = related_concepts

            # 4. 演进分析
            evolution = self.analyze_concept_evolution(concept)
            report['evolution'] = evolution

            # 5. 统计摘要
            if not temporal_df.empty:
                report['statistics'] = {
                    'total_mentions': int(temporal_df['count'].sum()),
                    'active_years': len(temporal_df),
                    'year_range': f"{temporal_df['year'].min()}-{temporal_df['year'].max()}",
                    'peak_year': int(temporal_df.loc[temporal_df['count'].idxmax(), 'year']),
                    'peak_count': int(temporal_df['count'].max()),
                    'average_mentions_per_year': round(temporal_df['count'].mean(), 2)
                }

        except Exception as e:
            logger.error(f"生成报告时出错: {e}")
            report['status'] = 'error'
            report['error'] = str(e)

        return report

    def get_available_concepts(self) -> List[str]:
        """获取可用的概念列表"""
        return sorted(list(self.known_concepts))

    def add_concept(self, concept: str):
        """添加新概念到已知概念集合"""
        self.known_concepts.add(concept)

    def get_concept_statistics(self) -> Dict:
        """获取概念相关的统计信息"""
        return {
            'known_concepts_count': len(self.known_concepts),
            'known_concepts': sorted(list(self.known_concepts)),
            'cache_size': len(self.concept_cache),
            'search_cache_size': len(self._search_cache)
        }

    def clear_cache(self):
        """清除所有缓存"""
        self.concept_cache.clear()
        self._search_cache.clear()
        logger.info("概念分析器缓存已清除")