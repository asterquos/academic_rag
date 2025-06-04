"""
build_rag.py
构建RAG系统 - 数据预处理和向量化

# 使用默认配置
python build_rag.py

# 重建向量库
python build_rag.py --rebuild

# 指定配置文件
python build_rag.py --config config.yaml
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import time
import json
import yaml
import argparse
import chromadb
from chromadb.config import Settings

# 导入自定义模块
from src.preprocessing.text_processor import TextProcessor, ChunkStrategy
from src.rag.embedding import EmbeddingModel
from src.preprocessing.excel_parser import ExcelParser
from src.preprocessing.data_validator import DataValidator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGBuilder:
    """RAG系统构建器"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化构建器"""
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化组件
        self.excel_parser = ExcelParser()
        self.text_processor = TextProcessor()
        self.data_validator = DataValidator()

        # 设置默认使用full_article策略
        self.text_processor.set_strategy('full_article')
        logger.info("使用chunk策略: full_article")

        # 初始化嵌入模型（使用高维度模型）
        embedding_config = self.config['embedding']
        model_type = embedding_config.get('current_model', 'bge_large')
        model_config = embedding_config['models'][model_type]

        self.embedding_model = EmbeddingModel(
            model_name=model_config['model_name'],
            device=model_config.get('device', 'cuda'),
            use_fp16=embedding_config.get('mixed_precision', True),
            batch_size=embedding_config.get('batch_size', 64)
        )

        logger.info(f"使用嵌入模型: {model_config['model_name']} ({model_config['dimension']}维)")

        # 初始化向量数据库
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self):
        """初始化ChromaDB向量存储"""
        persist_directory = self.config['vector_store']['persist_directory']
        collection_name = self.config['vector_store']['collection_name']

        # 创建目录
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # 创建ChromaDB客户端
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 处理集合的创建或重建
        try:
            if hasattr(self, 'rebuild') and self.rebuild:
                # 如果需要重建，先尝试删除旧集合
                try:
                    self.chroma_client.delete_collection(collection_name)
                    logger.info(f"✓ 删除旧集合: {collection_name}")
                except Exception as e:
                    logger.warning(f"删除集合失败（可能不存在）: {e}")

                # 创建新集合
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "艺术设计文献向量库"}
                )
                logger.info(f"✓ 创建新集合: {collection_name}")
            else:
                # 不重建时，尝试获取现有集合
                try:
                    collection = self.chroma_client.get_collection(collection_name)
                    logger.info(f"✓ 使用现有集合: {collection_name} (包含 {collection.count()} 个文档)")
                except Exception:
                    # 如果集合不存在，创建新的
                    collection = self.chroma_client.create_collection(
                        name=collection_name,
                        metadata={"description": "艺术设计文献向量库"}
                    )
                    logger.info(f"✓ 创建新集合: {collection_name}")
        except Exception as e:
            logger.error(f"初始化向量存储失败: {e}")
            raise

        return collection

    def process_data(self, input_file: str, output_dir: str = "data/processed"):
        """
        处理数据的主流程

        Args:
            input_file: 输入的Excel文件路径
            output_dir: 输出目录
        """
        logger.info("=" * 70)
        logger.info("开始数据处理流程")
        logger.info("=" * 70)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: 加载和解析Excel数据
        logger.info("\n[Step 1/6] 加载Excel数据...")
        df = self.excel_parser.process(input_file, filter_selected=False)
        logger.info(f"✓ 加载 {len(df)} 条记录")

        # Step 2: 数据验证
        logger.info("\n[Step 2/6] 验证数据质量...")
        validation_report = self.data_validator.validate_dataframe(df)
        logger.info(f"✓ 数据质量得分: {validation_report['data_quality_score']:.1f}/100")

        # 保存验证报告
        with open(output_path / "validation_report.json", 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)

        # Step 3: 创建文本chunks
        logger.info("\n[Step 3/6] 创建文本chunks...")
        chunks_data = self._create_chunks(df)
        logger.info(f"✓ 创建 {len(chunks_data)} 个chunks")

        # Step 4: 生成嵌入向量
        logger.info("\n[Step 4/6] 生成嵌入向量...")
        embeddings = self._generate_embeddings(chunks_data)
        logger.info(f"✓ 生成 {len(embeddings)} 个嵌入向量，维度: {embeddings[0].shape}")

        # Step 5: 存储到向量数据库
        logger.info("\n[Step 5/6] 存储到向量数据库...")
        self._store_to_vectordb(chunks_data, embeddings)

        # Step 6: 保存处理结果
        logger.info("\n[Step 6/6] 保存处理结果...")
        self._save_processed_data(df, chunks_data, output_path)

        logger.info("\n" + "=" * 70)
        logger.info("✅ 数据处理完成！")
        logger.info("=" * 70)

        # Step 7: 构建BM25索引
        logger.info("\n[Step 7/7] 构建BM25索引...")
        self._build_bm25_index(chunks_data)

        # 打印统计信息
        self._print_statistics(df, chunks_data)

    def _create_chunks(self, df: pd.DataFrame) -> List[Dict]:
        """创建文本chunks"""
        chunks_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="创建chunks"):
            # 构建文档对象
            doc = {
                'doc_id': row.get('doc_id', f"doc_{idx}"),
                'text': row.get('全文', ''),
                'metadata': {
                    '年份': row.get('年份'),
                    '作者名称': row.get('作者名称'),
                    '文章名称+副标题': row.get('文章名称+副标题'),
                    '分类': row.get('分类'),
                    '刊号': row.get('刊号'),
                    '是否入选': row.get('是否入选', False)
                }
            }

            # 创建chunks（使用默认的full_article策略）
            chunks = self.text_processor.create_chunks(doc)

            # 转换为字典格式
            for chunk in chunks:
                chunk_dict = {
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'strategy': chunk.strategy.value,
                    'position': chunk.position,
                    'length': chunk.length
                }
                chunks_data.append(chunk_dict)

        return chunks_data

    def _generate_embeddings(self, chunks_data: List[Dict]) -> List[np.ndarray]:
        """批量生成嵌入向量"""
        texts = [chunk['text'] for chunk in chunks_data]
        embeddings = []

        # 批量处理
        batch_size = self.embedding_model.batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode_corpus(
                batch_texts,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

        return embeddings

    def _store_to_vectordb(self, chunks_data: List[Dict], embeddings: List[np.ndarray]):
        """存储到向量数据库"""
        # 准备数据
        ids = [chunk['chunk_id'] for chunk in chunks_data]
        documents = [chunk['text'] for chunk in chunks_data]
        metadatas = [chunk['metadata'] for chunk in chunks_data]

        # 添加额外的元数据
        for i, metadata in enumerate(metadatas):
            metadata['chunk_strategy'] = chunks_data[i]['strategy']
            metadata['chunk_position'] = chunks_data[i]['position']
            metadata['chunk_length'] = chunks_data[i]['length']
            # 确保所有值都是字符串（ChromaDB要求）
            for key, value in metadata.items():
                if value is None:
                    metadata[key] = ""
                elif isinstance(value, (int, float, bool)):
                    metadata[key] = str(value)

        # 批量添加到向量数据库
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc="存储到向量库"):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            self.vector_store.add(
                ids=batch_ids,
                embeddings=[emb.tolist() for emb in batch_embeddings],
                documents=batch_documents,
                metadatas=batch_metadatas
            )

        logger.info(f"✓ 成功存储 {len(ids)} 个向量")

    def _save_processed_data(self, df: pd.DataFrame, chunks_data: List[Dict], output_path: Path):
        """保存处理后的数据"""
        # 保存原始处理后的DataFrame
        df.to_parquet(output_path / "processed_documents.parquet")
        logger.info(f"✓ 保存文档数据: processed_documents.parquet")

        # 保存chunks数据
        chunks_df = pd.DataFrame(chunks_data)
        chunks_df.to_parquet(output_path / "chunks_data.parquet")
        logger.info(f"✓ 保存chunks数据: chunks_data.parquet")

        # 保存配置信息
        process_info = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_documents': len(df),
            'total_chunks': len(chunks_data),
            'embedding_model': self.embedding_model.get_model_info(),
            'chunk_strategies': self.text_processor.get_enabled_strategies(),
            'vector_store': {
                'collection_name': self.config['vector_store']['collection_name'],
                'persist_directory': self.config['vector_store']['persist_directory']
            }
        }

        with open(output_path / "process_info.json", 'w', encoding='utf-8') as f:
            json.dump(process_info, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 保存处理信息: process_info.json")

    def _build_bm25_index(self, chunks_data: List[Dict]):
        """构建BM25索引"""
        # 准备文档列表
        documents = []
        for chunk in chunks_data:
            doc = {
                'id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'text': chunk['text'],
                'metadata': chunk['metadata']
            }
            documents.append(doc)

        # 创建临时的RAG引擎来构建索引
        from src.rag.engine import RAGEngine
        temp_rag = RAGEngine(
            collection_name=self.config['vector_store']['collection_name'],
            persist_directory=self.config['vector_store']['persist_directory'],
            enable_bm25=True
        )

        # 构建BM25索引
        temp_rag.build_bm25_index(documents, force_rebuild=True)
        logger.info(f"✓ BM25索引构建完成，包含 {len(documents)} 个文档")

    def _print_statistics(self, df: pd.DataFrame, chunks_data: List[Dict]):
        """打印统计信息"""
        print("\n📊 处理统计:")
        print(f"  原始文档数: {len(df)}")
        print(f"  生成chunks数: {len(chunks_data)}")
        print(f"  平均每文档chunks数: {len(chunks_data) / len(df):.1f}")

        # Chunk策略统计
        strategy_counts = {}
        for chunk in chunks_data:
            strategy = chunk['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        print("\n  Chunk策略分布:")
        for strategy, count in strategy_counts.items():
            print(f"    {strategy}: {count} ({count / len(chunks_data) * 100:.1f}%)")

        # 长度统计
        lengths = [chunk['length'] for chunk in chunks_data]
        print(f"\n  Chunk长度统计:")
        print(f"    平均: {np.mean(lengths):.0f} 字符")
        print(f"    最小: {np.min(lengths)} 字符")
        print(f"    最大: {np.max(lengths)} 字符")
        print(f"    中位数: {np.median(lengths):.0f} 字符")

    def test_retrieval(self, query: str, top_k: int = 5):
        """测试检索功能"""
        logger.info(f"\n🔍 测试检索: '{query}'")

        # 生成查询嵌入
        query_embedding = self.embedding_model.encode_queries(query)

        # 检索
        results = self.vector_store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # 显示结果
        if results['ids'][0]:
            print(f"\n找到 {len(results['ids'][0])} 个相关结果:")
            for i, (doc_id, distance, document, metadata) in enumerate(zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['documents'][0],
                    results['metadatas'][0]
            )):
                print(f"\n[{i + 1}] 相似度: {1 - distance:.3f}")
                print(f"  标题: {metadata.get('文章名称+副标题', 'N/A')}")
                print(f"  作者: {metadata.get('作者名称', 'N/A')}")
                print(f"  年份: {metadata.get('年份', 'N/A')}")
                print(f"  内容预览: {document[:200]}...")
        else:
            print("未找到相关结果")




def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建RAG系统')
    parser.add_argument('--input', type=str, default='data/raw/applied_arts.xlsx',
                        help='输入的Excel文件路径')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--rebuild', action='store_true',
                        help='是否重建向量库')
    parser.add_argument('--test', action='store_true',
                        help='构建后进行测试')
    parser.add_argument('--chunk-strategy', type=str, default='full_article',
                        choices=['full_article', 'paragraph_based', 'fixed_size'],
                        help='Chunk划分策略')

    args = parser.parse_args()

    # 创建构建器
    builder = RAGBuilder(args.config)
    builder.rebuild = args.rebuild

    # 设置chunk策略
    if args.chunk_strategy != 'full_article':
        builder.text_processor.set_strategy(args.chunk_strategy)
        logger.info(f"切换chunk策略为: {args.chunk_strategy}")

    # 处理数据
    builder.process_data(args.input, args.output)

    # 测试检索
    if args.test:
        print("\n" + "=" * 70)
        print("测试检索功能")
        print("=" * 70)

        test_queries = [
            "包豪斯的设计理念",
            "现代主义建筑",
            "工业设计的发展",
            "中国传统工艺"
        ]

        for query in test_queries:
            builder.test_retrieval(query, top_k=3)
            print("\n" + "-" * 50)


if __name__ == "__main__":
    main()