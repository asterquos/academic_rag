"""
debug_embedding.py
深入调试嵌入格式问题
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.rag.engine import RAGEngine
from src.rag.embedding import EmbeddingModel
from model_config import get_embedding_model_config


def debug_retrieval_chain():
    """调试整个检索链"""
    print("调试检索链...")

    # 1. 初始化RAG引擎
    print("\n1. 初始化RAG引擎")
    rag = RAGEngine(
        collection_name="art_design_docs_v2",
        persist_directory="data/chroma_v2",
        enable_bm25=True
    )

    # 2. 测试查询
    query = "包豪斯"
    print(f"\n2. 测试查询: '{query}'")

    # 3. 测试嵌入模型
    print("\n3. 测试嵌入模型直接编码")
    embedding = rag.embedding_model.encode_queries(query)
    print(f"   类型: {type(embedding)}")
    print(f"   形状: {embedding.shape if hasattr(embedding, 'shape') else 'N/A'}")
    print(f"   维度: {embedding.ndim if hasattr(embedding, 'ndim') else 'N/A'}")
    if hasattr(embedding, 'shape'):
        print(f"   前5个值: {embedding.flatten()[:5]}")

    # 4. 测试retriever
    print("\n4. 测试retriever的encode_queries")
    retriever_embedding = rag.retriever.embedding_model.encode_queries(query)
    print(f"   类型: {type(retriever_embedding)}")
    print(f"   形状: {retriever_embedding.shape if hasattr(retriever_embedding, 'shape') else 'N/A'}")

    # 5. 手动测试向量搜索
    print("\n5. 手动测试向量搜索")
    try:
        # 确保是一维数组
        if retriever_embedding.ndim > 1:
            search_embedding = retriever_embedding[0]
        else:
            search_embedding = retriever_embedding

        print(f"   搜索嵌入形状: {search_embedding.shape}")
        print(f"   搜索嵌入类型: {type(search_embedding)}")

        # 直接调用vector_store
        results = rag.vector_store.search(
            query_embedding=search_embedding,
            n_results=5
        )
        print(f"   搜索成功，找到 {len(results)} 个结果")

    except Exception as e:
        print(f"   搜索失败: {e}")
        print(f"   错误类型: {type(e)}")

        # 尝试不同的格式
        print("\n6. 尝试不同的嵌入格式")

        # 尝试1: 转换为列表
        try:
            if hasattr(search_embedding, 'tolist'):
                list_embedding = search_embedding.tolist()
                print(f"   列表格式类型: {type(list_embedding)}")
                print(f"   列表长度: {len(list_embedding) if isinstance(list_embedding, list) else 'N/A'}")
        except:
            pass


def test_direct_chromadb():
    """直接测试ChromaDB"""
    print("\n7. 直接测试ChromaDB集合")

    import chromadb
    client = chromadb.PersistentClient(path="data/chroma_v2")

    try:
        collection = client.get_collection("art_design_docs_v2")
        print(f"   集合存在，文档数: {collection.count()}")

        # 创建测试嵌入
        test_embedding = np.random.rand(1024).astype(np.float32)
        print(f"   测试嵌入形状: {test_embedding.shape}")

        # 测试查询
        results = collection.query(
            query_embeddings=[test_embedding.tolist()],
            n_results=1
        )
        print(f"   查询成功")

    except Exception as e:
        print(f"   ChromaDB测试失败: {e}")


if __name__ == "__main__":
    debug_retrieval_chain()
    test_direct_chromadb()