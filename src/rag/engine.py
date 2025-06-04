"""
src/rag/engine.py
简化的RAG引擎 - 组件协调和统一接口
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .embedding import EmbeddingModel
from .vector_store import VectorStore
from .retriever import UnifiedRetriever
from .generator import AnswerGenerator

# 导入统一的模型配置
try:
    from model_config import get_embedding_model_config
except ImportError:
    # 如果找不到model_config，使用默认配置
    def get_embedding_model_config(**kwargs):
        return kwargs

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG引擎，协调各个组件"""

    def __init__(
        self,
        collection_name: str = "art_design_docs",
        persist_directory: str = "data/chroma",
        embedding_model: str = None,
        gemini_api_key: str = None,
        enable_bm25: bool = True
    ):
        """
        初始化RAG引擎

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_model: 嵌入模型名称
            gemini_api_key: Gemini API密钥
            enable_bm25: 是否启用BM25
        """
        logger.info("初始化RAG引擎...")

        # 初始化嵌入模型
        if embedding_model and isinstance(embedding_model, str):
            # 如果传入的是模型名称字符串，使用统一配置
            embedding_kwargs = get_embedding_model_config()
            self.embedding_model = EmbeddingModel(
                model_name=embedding_model,
                **embedding_kwargs
            )
        elif embedding_model is None:
            # 使用默认配置
            embedding_kwargs = get_embedding_model_config()
            self.embedding_model = EmbeddingModel(**embedding_kwargs)
        else:
            # 如果传入的已经是 EmbeddingModel 实例
            self.embedding_model = embedding_model

        # 初始化向量存储
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        # 初始化检索器
        self.retriever = UnifiedRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            enable_bm25=enable_bm25,
            persist_directory=persist_directory
        )

        # 初始化生成器
        self.generator = AnswerGenerator(
            api_key=gemini_api_key
        )

        # 搜索历史（用于统计）
        self.search_history = []

        logger.info("RAG引擎初始化完成")

    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        搜索接口（向后兼容）

        Args:
            query: 查询文本
            top_k: 返回结果数
            method: 检索方法
            **kwargs: 其他参数

        Returns:
            搜索结果列表
        """
        results, stats = self.hybrid_search(query, top_k, method=method, **kwargs)
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        rerank: bool = True,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        混合搜索

        Args:
            query: 查询文本
            top_k: 返回结果数
            method: 检索方法 ("vector", "bm25", "hybrid")
            bm25_weight: BM25权重
            vector_weight: 向量权重
            rerank: 是否重排序

        Returns:
            (搜索结果, 搜索统计)
        """
        # 调用检索器
        results, stats = self.retriever.search(
            query=query,
            top_k=top_k,
            method=method,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rerank=rerank
        )

        # 记录历史
        self.search_history.append(stats)

        return results, stats

    def generate_answer(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        生成答案

        Args:
            query: 查询
            context: 上下文

        Returns:
            生成的答案
        """
        return self.generator.generate(query, context)

    def generate_answer_with_citations(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        生成带引用的答案

        Args:
            query: 查询
            context: 上下文

        Returns:
            带引用的答案
        """
        return self.generator.generate_with_citations(query, context)

    def build_bm25_index(self, documents: List[Dict] = None, force_rebuild: bool = False):
        """
        构建BM25索引

        Args:
            documents: 文档列表
            force_rebuild: 是否强制重建
        """
        # 检查是否需要重建
        if not force_rebuild and self.retriever.bm25_index:
            logger.info("BM25索引已存在，跳过构建")
            return

        if documents is None:
            logger.error("未提供文档数据")
            return

        self.retriever.build_bm25_index(documents)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total_documents': self.vector_store.count(),
            'collection_name': self.vector_store.collection_name,
            'embedding_model': self.embedding_model.model_name,
            'embedding_dim': self.embedding_model.embedding_dim,
            'bm25_enabled': self.retriever.enable_bm25,
            'bm25_indexed': self.retriever.bm25_index is not None,
            'generator_available': self.generator.is_available(),
            'search_history_count': len(self.search_history)
        }

        if self.retriever.bm25_index:
            stats['bm25_documents'] = len(self.retriever.documents)

        return stats

    @property
    def llm(self):
        """兼容属性：获取LLM"""
        return self.generator.llm

    @property
    def bm25_index(self):
        """兼容属性：获取BM25索引"""
        return self.retriever.bm25_index

    @property
    def documents(self):
        """兼容属性：获取文档列表"""
        return self.retriever.documents


# 向后兼容的别名
ImprovedRAGEngine = RAGEngine
SimpleRAGEngine = RAGEngine  # 兼容刚才的修改