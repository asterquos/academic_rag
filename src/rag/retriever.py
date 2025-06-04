"""
src/rag/retriever.py
整合的检索器 - 支持向量检索、BM25检索和混合检索
"""

import logging
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from rank_bm25 import BM25Okapi
import jieba
from datetime import datetime
from tqdm import tqdm

from .embedding import EmbeddingModel
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class UnifiedRetriever:
    """统一的检索器，整合向量检索和BM25检索"""

    def __init__(
            self,
            embedding_model: EmbeddingModel,
            vector_store: VectorStore,
            enable_bm25: bool = True,
            persist_directory: str = None
    ):
        """
        初始化检索器

        Args:
            embedding_model: 嵌入模型
            vector_store: 向量存储
            enable_bm25: 是否启用BM25
            persist_directory: 持久化目录
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.enable_bm25 = enable_bm25
        self.persist_directory = persist_directory

        # BM25相关
        self.bm25_index = None
        self.documents = []
        self.tokenized_docs = []

        # 尝试加载BM25索引
        if self.enable_bm25 and persist_directory:
            self._load_bm25_index()

    def search(
            self,
            query: str,
            top_k: int = 10,
            method: str = "hybrid",
            bm25_weight: float = 0.3,
            vector_weight: float = 0.7,
            rerank: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行搜索

        Args:
            query: 查询文本
            top_k: 返回结果数
            method: 检索方法 ("vector", "bm25", "hybrid")
            bm25_weight: BM25权重（混合检索时）
            vector_weight: 向量权重（混合检索时）
            rerank: 是否重排序

        Returns:
            (搜索结果, 搜索统计)
        """
        start_time = datetime.now()

        search_stats = {
            'query': query,
            'timestamp': start_time.isoformat(),
            'method': method
        }

        # 根据方法选择检索策略
        if method == "vector":
            results = self._vector_search(query, top_k)
            search_stats['vector_results'] = len(results)

        elif method == "bm25":
            if not self.bm25_index:
                logger.warning("BM25索引不可用，回退到向量检索")
                results = self._vector_search(query, top_k)
                search_stats['method'] = 'vector_fallback'
            else:
                results = self._bm25_search(query, top_k)
                search_stats['bm25_results'] = len(results)

        else:  # hybrid
            # 向量检索
            vector_results = self._vector_search(query, top_k * 2)
            search_stats['vector_results'] = len(vector_results)

            # BM25检索（如果可用）
            if self.bm25_index:
                bm25_results = self._bm25_search(query, top_k * 2)
                search_stats['bm25_results'] = len(bm25_results)

                # 融合结果
                results = self._merge_results(
                    bm25_results,
                    vector_results,
                    bm25_weight,
                    vector_weight
                )
            else:
                results = vector_results
                search_stats['method'] = 'vector_only'

        # 重排序
        if rerank and len(results) > 0:
            results = self._rerank_results(query, results)

        # 截取top_k
        results = results[:top_k]

        # 计算耗时
        search_time = (datetime.now() - start_time).total_seconds()
        search_stats['search_time'] = search_time
        search_stats['final_results'] = len(results)

        return results, search_stats

    def build_bm25_index(self, documents: List[Dict[str, Any]], save: bool = True):
        """
        构建BM25索引

        Args:
            documents: 文档列表
            save: 是否保存索引
        """
        if not self.enable_bm25:
            logger.info("BM25未启用，跳过索引构建")
            return

        logger.info(f"构建BM25索引，共{len(documents)}个文档")

        self.documents = documents

        # 分词处理
        self.tokenized_docs = []
        for doc in tqdm(documents, desc="分词处理"):
            text = doc.get('text', '')
            tokens = list(jieba.cut(text.lower()))
            self.tokenized_docs.append(tokens)

        # 构建BM25索引
        self.bm25_index = BM25Okapi(self.tokenized_docs)

        # 保存索引
        if save and self.persist_directory:
            self._save_bm25_index()

        logger.info("BM25索引构建完成")

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索"""
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode_queries(query)

        # 调试信息
        logger.debug(f"Query embedding type: {type(query_embedding)}")
        logger.debug(f"Query embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'N/A'}")

        # 确保 query_embedding 是一维数组
        if isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim > 1:
                # 如果是二维数组（批处理结果），取第一个向量
                logger.debug(f"Converting from shape {query_embedding.shape} to {query_embedding[0].shape}")
                query_embedding = query_embedding[0]
            # 确保是 float32 类型
            query_embedding = query_embedding.astype(np.float32)

        # 调用向量存储的搜索方法
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k
        )

        # 添加检索方法标记
        for r in results:
            r['retrieval_method'] = 'vector'

        return results

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25检索"""
        # 查询分词
        query_tokens = list(jieba.cut(query.lower()))

        # BM25评分
        scores = self.bm25_index.get_scores(query_tokens)

        # 获取top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0 and idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(scores[idx])
                doc['retrieval_method'] = 'bm25'
                results.append(doc)

        return results


    def _merge_results(
            self,
            bm25_results: List[Dict],
            vector_results: List[Dict],
            bm25_weight: float,
            vector_weight: float
    ) -> List[Dict[str, Any]]:
        """融合BM25和向量检索结果"""
        # 构建ID到结果的映射
        merged_dict = {}

        # 归一化分数
        def normalize_scores(results):
            if not results:
                return
            scores = [r['score'] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                for r in results:
                    r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score)
            else:
                for r in results:
                    r['normalized_score'] = 0.5

        normalize_scores(bm25_results)
        normalize_scores(vector_results)

        # 合并结果
        for result in bm25_results:
            doc_id = result.get('id', result.get('doc_id', ''))
            if doc_id:
                merged_dict[doc_id] = {
                    **result,
                    'bm25_score': result['normalized_score'],
                    'vector_score': 0,
                    'combined_score': bm25_weight * result['normalized_score']
                }

        for result in vector_results:
            doc_id = result.get('id', result.get('doc_id', ''))
            if doc_id:
                if doc_id in merged_dict:
                    merged_dict[doc_id]['vector_score'] = result['normalized_score']
                    merged_dict[doc_id]['combined_score'] = (
                            bm25_weight * merged_dict[doc_id]['bm25_score'] +
                            vector_weight * result['normalized_score']
                    )
                else:
                    merged_dict[doc_id] = {
                        **result,
                        'bm25_score': 0,
                        'vector_score': result['normalized_score'],
                        'combined_score': vector_weight * result['normalized_score']
                    }

        # 按组合分数排序
        merged_list = list(merged_dict.values())
        merged_list.sort(key=lambda x: x['combined_score'], reverse=True)

        return merged_list

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """重排序结果"""
        query_terms = set(jieba.cut(query.lower()))

        for doc in results:
            text = doc.get('text', '').lower()
            doc_terms = set(jieba.cut(text))

            # 查询词覆盖率
            if query_terms:
                coverage = len(query_terms & doc_terms) / len(query_terms)
            else:
                coverage = 0

            # 精确匹配加分
            exact_bonus = 1.2 if query.lower() in text else 1.0

            # 长度惩罚
            text_len = len(text)
            if text_len < 100:
                len_penalty = 0.8
            elif text_len > 5000:
                len_penalty = 0.9
            else:
                len_penalty = 1.0

            # 计算重排序分数
            doc['rerank_score'] = (
                    doc.get('combined_score', doc.get('score', 0)) *
                    exact_bonus *
                    (1 + coverage * 0.3) *
                    len_penalty
            )

        # 按重排序分数排序
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return results

    def _get_bm25_cache_path(self) -> Path:
        """获取BM25缓存路径"""
        if not self.persist_directory:
            return None
        cache_dir = Path(self.persist_directory) / "bm25_cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{self.vector_store.collection_name}_bm25.pkl"

    def _save_bm25_index(self):
        """保存BM25索引"""
        cache_path = self._get_bm25_cache_path()
        if not cache_path:
            return

        try:
            cache_data = {
                'bm25_index': self.bm25_index,
                'documents': self.documents,
                'tokenized_docs': self.tokenized_docs,
                'timestamp': datetime.now().isoformat()
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f"BM25索引已保存到: {cache_path}")

        except Exception as e:
            logger.error(f"保存BM25索引失败: {e}")

    def _load_bm25_index(self) -> bool:
        """加载BM25索引"""
        cache_path = self._get_bm25_cache_path()
        if not cache_path or not cache_path.exists():
            return False

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.bm25_index = cache_data['bm25_index']
            self.documents = cache_data['documents']
            self.tokenized_docs = cache_data['tokenized_docs']

            logger.info(f"成功加载BM25索引，共{len(self.documents)}个文档")
            return True

        except Exception as e:
            logger.error(f"加载BM25索引失败: {e}")
            return False