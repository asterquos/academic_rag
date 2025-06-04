"""
src/rag/vector_store.py
ChromaDB向量存储 - 精简版
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB向量存储"""
    
    def __init__(
        self,
        collection_name: str = "art_design_docs",
        persist_directory: Optional[str] = None
    ):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        # 创建客户端
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )
            
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"加载集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            logger.info(f"创建集合: {collection_name}")
            
        self.collection_name = collection_name
        
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """添加文档"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
            
        # 确保嵌入是列表格式
        embeddings_list = [
            emb.tolist() if isinstance(emb, np.ndarray) else emb 
            for emb in embeddings
        ]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        return ids


    def search(
            self,
            query_embedding: np.ndarray,
            n_results: int = 10,
            where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """向量搜索"""
        # 确保查询嵌入是正确的格式
        if isinstance(query_embedding, np.ndarray):
            # 确保是一维数组
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()

            # 转换为float32（ChromaDB要求）
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)

            # 转换为列表
            query_emb = query_embedding.tolist()
        else:
            query_emb = query_embedding

        # 确保查询嵌入是一个列表（不是嵌套列表）
        if isinstance(query_emb, list) and len(query_emb) > 0 and isinstance(query_emb[0], list):
            # 如果是嵌套列表，取第一个
            query_emb = query_emb[0]

        results = self.collection.query(
            query_embeddings=[query_emb],  # ChromaDB需要列表包装
            n_results=n_results,
            where=where
        )

        # 格式化结果
        formatted_results = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'score': 1 / (1 + results['distances'][0][i])
                })

        return formatted_results
        
    def get(self, ids: List[str]) -> Dict[str, Any]:
        """获取文档"""
        return self.collection.get(ids=ids)
        
    def delete(self, ids: List[str]):
        """删除文档"""
        self.collection.delete(ids=ids)
        
    def count(self) -> int:
        """文档数量"""
        return self.collection.count()
