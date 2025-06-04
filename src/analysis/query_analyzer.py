"""
src/analysis/query_analyzer.py
查询意图分析器 - 识别用户查询意图并分解任务
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """查询意图类型"""
    # 概念相关
    CONCEPT_FIRST_APPEARANCE = "concept_first_appearance"  # 概念首次出现
    CONCEPT_TIMELINE = "concept_timeline"  # 概念时间线
    CONCEPT_FREQUENCY = "concept_frequency"  # 概念频率分析
    CONCEPT_COMPARISON = "concept_comparison"  # 概念对比分析
    CONCEPT_EVOLUTION = "concept_evolution"  # 概念演进
    CONCEPT_CONTEXT = "concept_context"  # 概念上下文

    # 作者相关
    AUTHOR_RESEARCH = "author_research"  # 作者研究分析
    AUTHOR_TIMELINE = "author_timeline"  # 作者时间线
    AUTHOR_TOPICS = "author_topics"  # 作者研究主题
    AUTHOR_COLLABORATION = "author_collaboration"  # 作者合作关系
    AUTHOR_CITATION = "author_citation"  # 作者引用关系

    # 综合分析
    CONCEPT_RESEARCHERS = "concept_researchers"  # 概念的研究者分析
    MULTI_ANALYSIS = "multi_analysis"  # 多维度综合分析

    # 基础搜索
    SIMPLE_SEARCH = "simple_search"  # 简单搜索

    # 可视化
    VISUALIZATION = "visualization"  # 可视化需求


@dataclass
class QueryTask:
    """查询任务"""
    intent: QueryIntent
    entities: Dict[str, List[str]]  # 实体类型 -> 实体列表
    requirements: List[str]  # 具体要求
    visualization: bool = False  # 是否需要可视化
    comparison: bool = False  # 是否需要对比


class QueryAnalyzer:
    """查询意图分析器"""

    def __init__(self):
        # 意图识别关键词
        self.intent_keywords = {
            QueryIntent.CONCEPT_FIRST_APPEARANCE: [
                '首次出现', '第一次', '最早', '起源', '开始'
            ],
            QueryIntent.CONCEPT_TIMELINE: [
                '时间线', '时间', '历史', '演变', '发展'
            ],
            QueryIntent.CONCEPT_FREQUENCY: [
                '频率', '次数', '统计', '分布'
            ],
            QueryIntent.CONCEPT_COMPARISON: [
                '对比', '比较', '和', '与', '一起'
            ],
            QueryIntent.CONCEPT_RESEARCHERS: [
                '研究者', '学者', '作者', '谁研究', '研究人员'
            ],
            QueryIntent.AUTHOR_RESEARCH: [
                '研究成果', '发表', '文章', '论文'
            ],
            QueryIntent.AUTHOR_TOPICS: [
                '主题', '研究主题', '研究方向', '研究内容'
            ],
            QueryIntent.AUTHOR_COLLABORATION: [
                '合作', '共同', '一起', '协作'
            ],
            QueryIntent.AUTHOR_CITATION: [
                '引用', '互引', '引证', '参考'
            ],
            QueryIntent.VISUALIZATION: [
                '可视化', '图表', '图形', '展示', '绘制'
            ]
        }

        # 实体类型识别
        self.entity_patterns = {
            'concept': [
                r'"([^"]+)"',  # 双引号中的内容
                r'「([^」]+)」',  # 中文引号
                r'概念[：:]?\s*([^\s，。,]+)',
                r'关于([^\s，。,]+)',
            ],
            'author': [
                r'作者[：:]?\s*([^\s，。,]+)',
                r'研究者[：:]?\s*([^\s，。,]+)',
                r'学者[：:]?\s*([^\s，。,]+)',
            ],
            'time': [
                r'(\d{4}年)',
                r'(\d+年代)',
                r'(最近|近期|早期|晚期)',
            ]
        }

    def analyze(self, query: str) -> List[QueryTask]:
        """
        分析查询意图

        Args:
            query: 用户查询

        Returns:
            查询任务列表
        """
        logger.info(f"分析查询: {query}")

        # 1. 提取实体
        entities = self._extract_entities(query)

        # 2. 识别意图
        intents = self._identify_intents(query, entities)

        # 3. 构建任务
        tasks = self._build_tasks(query, entities, intents)

        # 4. 如果没有识别到明确意图，默认为简单搜索
        if not tasks:
            tasks.append(QueryTask(
                intent=QueryIntent.SIMPLE_SEARCH,
                entities=entities,
                requirements=[query]
            ))

        logger.info(f"识别到 {len(tasks)} 个任务")
        return tasks

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """提取查询中的实体"""
        entities = {
            'concept': [],
            'author': [],
            'time': []
        }

        # 提取概念
        for pattern in self.entity_patterns['concept']:
            matches = re.findall(pattern, query)
            entities['concept'].extend(matches)

        # 特殊处理：如果没有找到引号中的概念，尝试识别关键概念词
        if not entities['concept']:
            # 常见的艺术设计概念
            known_concepts = [
                '包豪斯', '现代主义', '后现代主义', '极简主义', '构成主义',
                '装饰艺术', '工艺美术', '新艺术运动', '工业设计', '平面设计'
            ]
            for concept in known_concepts:
                if concept in query:
                    entities['concept'].append(concept)

        # 提取作者
        for pattern in self.entity_patterns['author']:
            matches = re.findall(pattern, query)
            entities['author'].extend(matches)

        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))

        logger.info(f"提取的实体: {entities}")
        return entities

    def _identify_intents(self, query: str, entities: Dict[str, List[str]]) -> List[QueryIntent]:
        """识别查询意图"""
        intents = []
        query_lower = query.lower()

        # 检查每种意图的关键词
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intents.append(intent)
                    break

        # 特殊规则
        if '研究者' in query and entities['concept']:
            intents.append(QueryIntent.CONCEPT_RESEARCHERS)

        if '时间' in query and '频率' in query:
            intents.append(QueryIntent.CONCEPT_TIMELINE)
            intents.append(QueryIntent.CONCEPT_FREQUENCY)

        if len(entities['concept']) > 1 and any(word in query for word in ['对比', '比较', '和', '与']):
            intents.append(QueryIntent.CONCEPT_COMPARISON)

        # 去重
        intents = list(set(intents))

        logger.info(f"识别的意图: {[intent.value for intent in intents]}")
        return intents

    def _build_tasks(self, query: str, entities: Dict[str, List[str]], intents: List[QueryIntent]) -> List[QueryTask]:
        """构建查询任务"""
        tasks = []

        # 检查是否需要可视化
        need_viz = any(word in query for word in ['可视化', '图表', '图形', '展示'])

        # 根据意图构建任务
        for intent in intents:
            task = QueryTask(
                intent=intent,
                entities=entities,
                requirements=self._extract_requirements(query, intent),
                visualization=need_viz
            )

            # 特殊处理
            if intent == QueryIntent.CONCEPT_COMPARISON:
                task.comparison = True

            tasks.append(task)

        # 合并相关任务
        tasks = self._merge_related_tasks(tasks)

        return tasks

    def _extract_requirements(self, query: str, intent: QueryIntent) -> List[str]:
        """提取具体要求"""
        requirements = []

        if intent == QueryIntent.CONCEPT_RESEARCHERS:
            if '时间顺序' in query:
                requirements.append('chronological_order')
            if '研究成果' in query or '主题' in query:
                requirements.append('research_topics')
            if '互引' in query or '引用' in query:
                requirements.append('citations')
            if '相关性' in query:
                requirements.append('correlation')

        elif intent == QueryIntent.CONCEPT_FIRST_APPEARANCE:
            if '作者信息' in query:
                requirements.append('author_info')
            if '高亮' in query:
                requirements.append('highlight')
            if '上下文' in query:
                requirements.append('context')

        elif intent == QueryIntent.CONCEPT_TIMELINE:
            if '频率' in query:
                requirements.append('frequency')
            if '文献名称' in query:
                requirements.append('document_titles')

        return requirements

    def _merge_related_tasks(self, tasks: List[QueryTask]) -> List[QueryTask]:
        """合并相关任务"""
        # 简单实现：暂不合并
        return tasks

    def format_task_description(self, task: QueryTask) -> str:
        """格式化任务描述"""
        desc_parts = []

        # 意图描述
        intent_desc = {
            QueryIntent.CONCEPT_FIRST_APPEARANCE: "查找概念首次出现",
            QueryIntent.CONCEPT_TIMELINE: "分析概念时间线",
            QueryIntent.CONCEPT_FREQUENCY: "统计概念频率",
            QueryIntent.CONCEPT_COMPARISON: "对比分析概念",
            QueryIntent.CONCEPT_RESEARCHERS: "分析概念研究者",
            QueryIntent.AUTHOR_RESEARCH: "分析作者研究",
            QueryIntent.AUTHOR_TOPICS: "分析研究主题",
            QueryIntent.AUTHOR_COLLABORATION: "分析合作关系",
            QueryIntent.AUTHOR_CITATION: "分析引用关系",
            QueryIntent.VISUALIZATION: "生成可视化",
            QueryIntent.SIMPLE_SEARCH: "基础搜索"
        }

        desc_parts.append(intent_desc.get(task.intent, "未知任务"))

        # 实体描述
        if task.entities['concept']:
            desc_parts.append(f"概念: {', '.join(task.entities['concept'])}")
        if task.entities['author']:
            desc_parts.append(f"作者: {', '.join(task.entities['author'])}")

        # 要求描述
        if task.requirements:
            req_desc = {
                'chronological_order': '按时间顺序',
                'research_topics': '研究主题',
                'citations': '引用关系',
                'correlation': '相关性分析',
                'author_info': '作者信息',
                'highlight': '高亮显示',
                'context': '上下文',
                'frequency': '频率统计',
                'document_titles': '文献标题'
            }
            reqs = [req_desc.get(r, r) for r in task.requirements]
            desc_parts.append(f"要求: {', '.join(reqs)}")

        if task.visualization:
            desc_parts.append("需要可视化")

        return " | ".join(desc_parts)
