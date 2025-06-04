"""
src/preprocessing/text_processor.py
文本处理器 - 支持多种chunk策略
"""

import re
import jieba
import jieba.posseg
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class ChunkStrategy(Enum):
    """Chunk划分策略"""
    FULL_ARTICLE = "full_article"      # 完整文章
    PARAGRAPH_BASED = "paragraph_based" # 基于段落
    FIXED_SIZE = "fixed_size"          # 固定大小
    SEMANTIC = "semantic"              # 语义分割


@dataclass
class Chunk:
    """文本块数据结构"""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict
    strategy: ChunkStrategy
    position: int
    length: int

    def to_dict(self) -> Dict:
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'text': self.text,
            'metadata': self.metadata,
            'strategy': self.strategy.value,
            'position': self.position,
            'length': self.length
        }


class TextProcessor:
    """文本处理器"""

    def __init__(self, config_path: Optional[str] = None):
        # 加载配置
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # 默认配置
            config = self._get_default_config()

        self.config = config['preprocessing']['text']

        # 初始化jieba
        self._init_jieba()

        # 编译正则表达式
        self.paragraph_pattern = re.compile(r'\n\s*\n|\n(?=\s{2,})')
        self.sentence_pattern = re.compile(r'[。！？.!?]+')

    def set_strategy(self, strategy: Union[str, ChunkStrategy, List[Union[str, ChunkStrategy]]]):
        """
        设置chunk策略

        Args:
            strategy: 策略名称、ChunkStrategy枚举或策略列表
                     可选值: 'full_article', 'paragraph_based', 'fixed_size', 'semantic'
                     或 ChunkStrategy.FULL_ARTICLE 等

        Examples:
            # 使用单一策略
            processor.set_strategy('full_article')
            processor.set_strategy(ChunkStrategy.FULL_ARTICLE)

            # 使用多种策略
            processor.set_strategy(['full_article', 'paragraph_based'])
        """
        # 重置所有策略
        for key in self.config['chunking_strategies']:
            self.config['chunking_strategies'][key]['enabled'] = False

        # 处理输入
        if isinstance(strategy, (str, ChunkStrategy)):
            strategies = [strategy]
        else:
            strategies = strategy

        # 启用指定策略
        for s in strategies:
            if isinstance(s, str):
                strategy_key = s
            elif isinstance(s, ChunkStrategy):
                strategy_key = s.value
            else:
                raise ValueError(f"不支持的策略类型: {type(s)}")

            if strategy_key in self.config['chunking_strategies']:
                self.config['chunking_strategies'][strategy_key]['enabled'] = True
                logger.info(f"启用策略: {strategy_key}")
            else:
                raise ValueError(f"未知的策略: {strategy_key}")

    def get_enabled_strategies(self) -> List[str]:
        """获取当前启用的策略名称列表"""
        enabled = []
        for key, config in self.config['chunking_strategies'].items():
            if config.get('enabled', False):
                enabled.append(key)
        return enabled
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'preprocessing': {
                'text': {
                    'chunking_strategies': {
                        'full_article': {'enabled': True, 'max_length': 8000},
                        'paragraph_based': {
                            'enabled': False,  # 默认关闭
                            'min_paragraph_length': 200,
                            'max_paragraph_length': 1500,
                            'paragraph_overlap': 100
                        },
                        'fixed_size': {
                            'enabled': False,
                            'chunk_size': 800,
                            'chunk_overlap': 100
                        }
                    },
                    'art_terms': ['包豪斯', '现代主义', '装饰艺术']
                }
            }
        }

    def _init_jieba(self):
        """初始化jieba分词器"""
        # 添加艺术设计专业词汇
        art_terms = self.config.get('art_terms', [])
        for term in art_terms:
            jieba.add_word(term, freq=10000)

    def create_chunks(self, doc: Dict[str, Any], strategies: Optional[List[ChunkStrategy]] = None) -> List[Chunk]:
        """
        创建文档chunks

        Args:
            doc: 文档数据，包含 'doc_id', 'text', 'metadata' 等字段
            strategies: 使用的策略列表，None则使用配置中启用的策略

        Returns:
            List[Chunk]: chunk列表
        """
        if strategies is None:
            strategies = self._get_enabled_strategies()

        chunks = []
        doc_id = doc.get('doc_id', '')
        text = doc.get('text', '')
        metadata = doc.get('metadata', {})

        # 文本预处理
        text = self.clean_text(text)

        # 根据不同策略创建chunks
        for strategy in strategies:
            if strategy == ChunkStrategy.FULL_ARTICLE:
                chunks.extend(self._create_full_article_chunks(doc_id, text, metadata))
            elif strategy == ChunkStrategy.PARAGRAPH_BASED:
                chunks.extend(self._create_paragraph_chunks(doc_id, text, metadata))
            elif strategy == ChunkStrategy.FIXED_SIZE:
                chunks.extend(self._create_fixed_size_chunks(doc_id, text, metadata))
            elif strategy == ChunkStrategy.SEMANTIC:
                chunks.extend(self._create_semantic_chunks(doc_id, text, metadata))

        return chunks

    def _get_enabled_strategies(self) -> List[ChunkStrategy]:
        """获取启用的策略"""
        strategies = []
        strategy_config = self.config.get('chunking_strategies', {})

        if strategy_config.get('full_article', {}).get('enabled', True):
            strategies.append(ChunkStrategy.FULL_ARTICLE)
        if strategy_config.get('paragraph_based', {}).get('enabled', True):
            strategies.append(ChunkStrategy.PARAGRAPH_BASED)
        if strategy_config.get('fixed_size', {}).get('enabled', False):
            strategies.append(ChunkStrategy.FIXED_SIZE)

        return strategies

    def _create_full_article_chunks(self, doc_id: str, text: str, metadata: Dict) -> List[Chunk]:
        """创建完整文章chunk"""
        max_length = self.config['chunking_strategies']['full_article']['max_length']

        # 如果文章不超过最大长度，作为一个chunk
        if len(text) <= max_length:
            chunk = Chunk(
                chunk_id=f"{doc_id}_full",
                doc_id=doc_id,
                text=text,
                metadata={**metadata, 'chunk_type': 'full_article'},
                strategy=ChunkStrategy.FULL_ARTICLE,
                position=0,
                length=len(text)
            )
            return [chunk]
        else:
            # 如果太长，退化为段落分割
            logger.warning(f"文档 {doc_id} 超过最大长度 {max_length}，使用段落分割")
            return self._create_paragraph_chunks(doc_id, text, metadata)

    def _create_paragraph_chunks(self, doc_id: str, text: str, metadata: Dict) -> List[Chunk]:
        """基于段落创建chunks"""
        config = self.config['chunking_strategies']['paragraph_based']
        min_length = config['min_paragraph_length']
        max_length = config['max_paragraph_length']
        overlap = config['paragraph_overlap']

        # 分割段落
        paragraphs = self.paragraph_pattern.split(text)
        chunks = []
        current_position = 0

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para or len(para) < min_length:
                continue

            # 如果段落太长，进一步分割
            if len(para) > max_length:
                sub_chunks = self._split_long_paragraph(para, max_length, overlap)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk = Chunk(
                        chunk_id=f"{doc_id}_p{i}_s{j}",
                        doc_id=doc_id,
                        text=sub_chunk,
                        metadata={**metadata, 'chunk_type': 'paragraph', 'paragraph_index': i},
                        strategy=ChunkStrategy.PARAGRAPH_BASED,
                        position=current_position,
                        length=len(sub_chunk)
                    )
                    chunks.append(chunk)
                    current_position += len(sub_chunk) - overlap
            else:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_p{i}",
                    doc_id=doc_id,
                    text=para,
                    metadata={**metadata, 'chunk_type': 'paragraph', 'paragraph_index': i},
                    strategy=ChunkStrategy.PARAGRAPH_BASED,
                    position=current_position,
                    length=len(para)
                )
                chunks.append(chunk)
                current_position += len(para)

        return chunks

    def _split_long_paragraph(self, paragraph: str, max_length: int, overlap: int) -> List[str]:
        """分割长段落"""
        # 先尝试按句子分割
        sentences = self.sentence_pattern.split(paragraph)

        chunks = []
        current_chunk = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # 如果加上这个句子不会超过最大长度
            if len(current_chunk) + len(sent) + 1 <= max_length:
                current_chunk = current_chunk + " " + sent if current_chunk else sent
            else:
                # 保存当前chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # 开始新chunk，包含重叠部分
                if overlap > 0 and chunks:
                    # 从上一个chunk取最后的overlap字符
                    overlap_text = chunks[-1][-overlap:]
                    current_chunk = overlap_text + " " + sent
                else:
                    current_chunk = sent

        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_fixed_size_chunks(self, doc_id: str, text: str, metadata: Dict) -> List[Chunk]:
        """创建固定大小的chunks"""
        config = self.config['chunking_strategies']['fixed_size']
        chunk_size = config['chunk_size']
        overlap = config['chunk_overlap']

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # 尝试在句子边界结束
            if end < len(text):
                last_period = chunk_text.rfind('。')
                if last_period > chunk_size * 0.8:  # 如果句号在后80%
                    end = start + last_period + 1
                    chunk_text = text[start:end]

            chunk = Chunk(
                chunk_id=f"{doc_id}_c{chunk_index}",
                doc_id=doc_id,
                text=chunk_text,
                metadata={**metadata, 'chunk_type': 'fixed_size', 'chunk_index': chunk_index},
                strategy=ChunkStrategy.FIXED_SIZE,
                position=start,
                length=len(chunk_text)
            )
            chunks.append(chunk)

            start = end - overlap if end < len(text) else end
            chunk_index += 1

        return chunks

    def _create_semantic_chunks(self, doc_id: str, text: str, metadata: Dict) -> List[Chunk]:
        """基于语义创建chunks（高级功能，需要额外的语义分析）"""
        # 这是一个占位实现，实际的语义分割需要更复杂的算法
        # 比如使用主题模型或语义相似度
        logger.info("语义分割功能尚未完全实现，使用段落分割作为替代")
        return self._create_paragraph_chunks(doc_id, text, metadata)

    def clean_text(self, text: str) -> str:
        """清理文本"""
        if pd.isna(text):
            return ""

        text = str(text)

        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 去除多余的空白字符（但保留段落结构）
        text = re.sub(r'[ \t]+', ' ', text)  # 空格和制表符
        text = re.sub(r'\n{3,}', '\n\n', text)  # 多个换行符

        return text.strip()

    def process_dataframe_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理DataFrame，为每个文档创建chunks

        Args:
            df: 包含文档的DataFrame

        Returns:
            包含所有chunks的DataFrame
        """
        all_chunks = []

        for idx, row in df.iterrows():
            # 构建文档对象
            doc = {
                'doc_id': row.get('doc_id', str(idx)),
                'text': row.get('全文', ''),
                'metadata': {
                    '年份': row.get('年份'),
                    '作者名称': row.get('作者名称'),
                    '文章名称+副标题': row.get('文章名称+副标题'),
                    '分类': row.get('分类'),
                    '刊号': row.get('刊号')
                }
            }

            # 创建chunks
            chunks = self.create_chunks(doc)

            # 转换为字典列表
            for chunk in chunks:
                chunk_dict = chunk.to_dict()
                # 展开metadata
                for key, value in chunk_dict['metadata'].items():
                    chunk_dict[f'meta_{key}'] = value
                all_chunks.append(chunk_dict)

        # 创建DataFrame
        chunks_df = pd.DataFrame(all_chunks)

        logger.info(f"创建了 {len(chunks_df)} 个chunks，来自 {len(df)} 个文档")

        # 统计不同策略的chunks
        strategy_counts = chunks_df['strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            logger.info(f"  {strategy}: {count} chunks")

        return chunks_df


# 测试
if __name__ == "__main__":
    # 创建处理器
    processor = TextProcessor()

    # 测试文档
    test_doc = {
        'doc_id': 'test_001',
        'text': """
包豪斯（Bauhaus）是1919年在德国魏玛成立的一所设计学校，由建筑师瓦尔特·格罗皮乌斯创立。
这所学校在现代设计史上具有里程碑意义，它不仅是一所学校，更是一场设计革命的发源地。

包豪斯的核心理念是将艺术与技术相结合，强调功能性和简洁性。学校倡导"形式追随功能"的设计原则，
反对过度装饰，主张设计应该服务于大众而非精英。这种理念深刻影响了20世纪的建筑、工业设计、
平面设计等多个领域。

在教学方法上，包豪斯采用了革命性的工作坊制度。学生们在大师的指导下，通过实践来学习设计。
学校设有金属工作坊、织物工作坊、陶瓷工作坊等，让学生能够亲手制作作品，理解材料和工艺的特性。

包豪斯的影响延续至今。它培养的设计师们将现代主义设计理念传播到世界各地，
特别是在美国，许多包豪斯教师移居美国后，继续传播包豪斯的设计理念，
对美国乃至全球的现代设计发展产生了深远影响。
        """,
        'metadata': {
            '年份': 2023,
            '作者名称': '张三',
            '文章名称+副标题': '包豪斯的历史与影响'
        }
    }

    print("=== Chunk策略使用示例 ===\n")

    # 示例1: 使用完整文章策略
    print("1. 完整文章策略:")
    processor.set_strategy('full_article')
    chunks = processor.create_chunks(test_doc)
    print(f"   生成 {len(chunks)} 个chunks")
    for chunk in chunks:
        print(f"   - {chunk.chunk_id}: {len(chunk.text)} 字符")

    # 示例2: 使用段落分割策略
    print("\n2. 段落分割策略:")
    processor.set_strategy('paragraph_based')
    chunks = processor.create_chunks(test_doc)
    print(f"   生成 {len(chunks)} 个chunks")
    for chunk in chunks:
        print(f"   - {chunk.chunk_id}: {len(chunk.text)} 字符")
        print(f"     开头: {chunk.text[:50]}...")

    # 示例3: 使用固定大小策略
    print("\n3. 固定大小策略:")
    processor.set_strategy('fixed_size')
    # 临时启用fixed_size策略
    processor.config['chunking_strategies']['fixed_size']['enabled'] = True
    chunks = processor.create_chunks(test_doc)
    print(f"   生成 {len(chunks)} 个chunks")
    for chunk in chunks:
        print(f"   - {chunk.chunk_id}: {len(chunk.text)} 字符")

    # 示例4: 使用多种策略组合
    print("\n4. 多策略组合 (完整文章 + 段落分割):")
    processor.set_strategy(['full_article', 'paragraph_based'])
    chunks = processor.create_chunks(test_doc)
    print(f"   生成 {len(chunks)} 个chunks")
    strategies_used = set(chunk.strategy.value for chunk in chunks)
    print(f"   使用的策略: {strategies_used}")

    # 示例5: 直接在create_chunks中指定策略
    print("\n5. 在create_chunks中直接指定策略:")
    chunks = processor.create_chunks(test_doc, strategies=[ChunkStrategy.FULL_ARTICLE])
    print(f"   生成 {len(chunks)} 个chunks")

    # 示例6: 查看当前启用的策略
    print("\n6. 当前启用的策略:")
    enabled = processor.get_enabled_strategies()
    print(f"   {enabled}")