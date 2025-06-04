"""
src/rag/embedding.py
嵌入模型 - 支持高维度中文模型
"""

import numpy as np
from typing import List, Union, Optional, Dict
import torch
from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache
import gc

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """嵌入模型 - 针对5070Ti优化"""

    # 推荐的高质量中文模型
    RECOMMENDED_MODELS = {
        # BGE系列 - 最推荐
        'bge-large-zh': {
            'name': 'BAAI/bge-large-zh-v1.5',
            'dimension': 1024,
            'max_seq_length': 512,
            'description': 'BGE大模型，最佳中文效果'
        },
        'bge-base-zh': {
            'name': 'BAAI/bge-base-zh-v1.5',
            'dimension': 768,
            'max_seq_length': 512,
            'description': 'BGE基础模型，平衡效果和速度'
        },

        # Text2Vec系列
        'text2vec-large': {
            'name': 'GanymedeNil/text2vec-large-chinese',
            'dimension': 1024,
            'max_seq_length': 512,
            'description': 'Text2Vec大模型'
        },

        # M3E系列
        'm3e-large': {
            'name': 'moka-ai/m3e-large',
            'dimension': 1024,
            'max_seq_length': 512,
            'description': 'M3E大模型，支持中英文'
        },

        # 轻量级选项
        'paraphrase-multilingual': {
            'name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'dimension': 384,
            'max_seq_length': 512,
            'description': '轻量级多语言模型'
        }
    }

    def __init__(
        self,
        model_name: str = None,
        model_type: str = 'bge-large-zh',
        device: str = None,
        cache_folder: str = None,
        use_fp16: bool = True,  # 使用半精度
        batch_size: int = 64    # 5070Ti可以处理更大批次
    ):
        """
        初始化嵌入模型

        Args:
            model_name: 具体模型名称
            model_type: 模型类型（从RECOMMENDED_MODELS选择）
            device: 设备类型
            cache_folder: 模型缓存目录
            use_fp16: 是否使用半精度（节省显存）
            batch_size: 批处理大小
        """
        # 确定使用的模型
        if model_name is None:
            if model_type in self.RECOMMENDED_MODELS:
                model_info = self.RECOMMENDED_MODELS[model_type]
                model_name = model_info['name']
                self.dimension = model_info['dimension']
                self.max_seq_length = model_info['max_seq_length']
            else:
                raise ValueError(f"未知的模型类型: {model_type}")
        else:
            # 从名称推断维度
            self._infer_model_info(model_name)

        self.model_name = model_name
        self.model_type = model_type
        self.batch_size = batch_size

        # 设备配置
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # GPU优化设置
        if self.device == 'cuda':
            # 为5070Ti优化
            torch.cuda.empty_cache()
            if use_fp16:
                torch.set_float32_matmul_precision('high')

        logger.info(f"加载嵌入模型: {model_name}")
        logger.info(f"模型维度: {self.dimension}")
        logger.info(f"设备: {device}")

        # 加载模型
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder
        )

        # 使用半精度
        if use_fp16 and device == 'cuda':
            self.model = self.model.half()
            logger.info("使用FP16半精度模式")

        # 设置为评估模式
        self.model.eval()

        # 获取实际的嵌入维度
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # 内存优化：启用梯度检查点
        if hasattr(self.model[0].auto_model, 'gradient_checkpointing_enable'):
            self.model[0].auto_model.gradient_checkpointing_enable()

    def _infer_model_info(self, model_name: str):
        """从模型名称推断信息"""
        if 'large' in model_name.lower():
            self.dimension = 1024
        elif 'base' in model_name.lower():
            self.dimension = 768
        else:
            self.dimension = 512  # 默认
        self.max_seq_length = 512

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 文本或文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条
            normalize: 是否归一化向量
            convert_to_numpy: 是否转换为numpy数组

        Returns:
            嵌入向量
        """
        if batch_size is None:
            batch_size = self.batch_size

        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]

        # 对长文本进行截断
        texts = [self._truncate_text(text) for text in texts]

        try:
            # 批量编码
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize,
                convert_to_numpy=convert_to_numpy,
                device=self.device
            )

            # 如果使用了FP16，需要转换回FP32以兼容ChromaDB
            if hasattr(self, 'model') and self.device == 'cuda':
                # 确保是numpy数组
                if torch.is_tensor(embeddings):
                    embeddings = embeddings.cpu().numpy()
                # 转换为float32
                embeddings = embeddings.astype(np.float32)

            return embeddings

        except Exception as e:
            logger.error(f"编码失败: {e}")
            # 降级到更小的批次
            if batch_size > 1:
                logger.info(f"尝试更小的批次大小: {batch_size // 2}")
                return self.encode(
                    texts,
                    batch_size=batch_size // 2,
                    show_progress_bar=show_progress_bar,
                    normalize=normalize,
                    convert_to_numpy=convert_to_numpy
                )
            else:
                raise

    def _truncate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """截断文本到最大长度"""
        if max_length is None:
            max_length = self.max_seq_length

        # 简单的字符级截断
        # 实际使用时应该用tokenizer来精确截断
        if len(text) > max_length * 2:  # 假设平均每个token 2个字符
            return text[:max_length * 2]
        return text

    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        编码查询文本（可能使用特殊的查询前缀）

        Args:
            queries: 查询文本
            **kwargs: 其他参数传递给encode

        Returns:
            查询嵌入向量
        """
        # 记录原始输入是否为字符串
        is_single_query = isinstance(queries, str)

        if is_single_query:
            queries = [queries]

        # BGE模型需要特殊的查询前缀
        if 'bge' in self.model_name.lower():
            queries = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]

        # 编码
        embeddings = self.encode(queries, **kwargs)

        # 如果原始输入是单个查询字符串，返回一维数组
        if is_single_query and embeddings.ndim > 1:
            return embeddings[0]

        return embeddings

    def encode_corpus(self, corpus: List[str], **kwargs) -> np.ndarray:
        """
        编码文档语料

        Args:
            corpus: 文档列表
            **kwargs: 其他参数传递给encode

        Returns:
            文档嵌入向量
        """
        # BGE模型的文档不需要特殊前缀
        return self.encode(corpus, **kwargs)

    @lru_cache(maxsize=10000)
    def encode_cached(self, text: str) -> np.ndarray:
        """缓存的编码函数（用于重复查询）"""
        return self.encode(text, show_progress_bar=False)

    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        计算两组嵌入的相似度矩阵

        Args:
            embeddings1: 第一组嵌入 [n, dim]
            embeddings2: 第二组嵌入 [m, dim]

        Returns:
            相似度矩阵 [n, m]
        """
        # 确保是numpy数组
        if torch.is_tensor(embeddings1):
            embeddings1 = embeddings1.cpu().numpy()
        if torch.is_tensor(embeddings2):
            embeddings2 = embeddings2.cpu().numpy()

        # 归一化
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        # 计算余弦相似度
        similarity = np.matmul(embeddings1, embeddings2.T)

        return similarity

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
            'max_seq_length': self.max_seq_length,
            'device': self.device,
            'batch_size': self.batch_size
        }

    def clear_cache(self):
        """清理缓存"""
        self.encode_cached.cache_clear()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


# 便捷函数
def load_embedding_model(model_type: str = 'bge-large-zh', **kwargs) -> EmbeddingModel:
    """加载嵌入模型的便捷函数"""
    return EmbeddingModel(model_type=model_type, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 测试不同的模型
    print("测试嵌入模型...")

    # 测试文本
    test_texts = [
        "包豪斯是1919年在德国魏玛成立的一所设计学校",
        "现代主义设计强调功能性和简洁性",
        "中国传统工艺美术具有悠久的历史"
    ]

    # 测试BGE大模型
    print("\n1. 测试BGE-Large模型:")
    model = EmbeddingModel(model_type='bge-large-zh')
    embeddings = model.encode(test_texts)
    print(f"嵌入形状: {embeddings.shape}")
    print(f"嵌入维度: {model.embedding_dim}")

    # 测试查询编码
    query = "包豪斯的设计理念"
    query_embedding = model.encode_queries(query)

    # 计算相似度
    similarities = model.compute_similarity(query_embedding, embeddings)
    print(f"\n查询 '{query}' 与文档的相似度:")
    for i, (text, sim) in enumerate(zip(test_texts, similarities[0])):
        print(f"  {i+1}. {sim:.3f} - {text[:30]}...")

    # 显示模型信息
    print(f"\n模型信息:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")